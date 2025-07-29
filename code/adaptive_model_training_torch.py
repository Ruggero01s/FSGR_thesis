import torch
import numpy as np
import os
from os import path
import datetime
import wandb
from sklearn.metrics import accuracy_score
import random
from plan import Plan
from grnet_model import PlanModel

from model_training_torch import (
    PlanDataset, train_model, CustomEarlyStopping, 
    get_model_predictions, print_metrics
)
from utils_torch import load_from_folder
from plan_generator_torch import PlanGeneratorMultiPerc, PlanGeneratorMultiPercAugmented
from torch.utils.data import DataLoader


def evaluate_plan(model, plan_data, goals_dict, device, 
                  confidence_threshold=0.08, precision_threshold=0.7,
                  precision_dict=None):
    """
    Evaluate a single plan using a metacognitive approach based on confidence
    and experience metrics.
    
    Args:
        model: The current model
        plan_data: Tuple of (x, y) containing plan data and target
        goals_dict: Dictionary mapping goals to indices
        device: Computing device (CPU/GPU)
        confidence_threshold: Minimum difference between top two goal scores
        precision_threshold: Minimum precision required for predicted goals
        precision_dict: Dictionary of historical precision data for goals/fluents
        
    Returns:
        passed: Boolean indicating if plan passed evaluation
        metrics: Dictionary with confidence and precision metrics
    """
    model.eval()
    x, y_true = plan_data #fix #todo why x is list of tensors? 
    x_tensor = torch.tensor(x, dtype=torch.long).to(device)
    y_true_tensor = torch.tensor(y_true, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        y_pred = model(x_tensor)
        y_pred_np = y_pred.cpu().numpy()[0]  # Get predictions as numpy array
        
        # Get scores and sort them to find the top two
        scores = sorted([(i, score) for i, score in enumerate(y_pred_np)], 
                       key=lambda x: x[1], reverse=True)
        
        # Calculate confidence (difference between top two scores)
        confidence = scores[0][1] - scores[1][1] if len(scores) > 1 else scores[0][1]
        top_goal_idx = scores[0][0]
        
        precision = 0.0
        
        # Get all fluents that are part of the true goal (where y_true == 1)
        true_goal_indices = [i for i, val in enumerate(y_true[0]) if val == 1]
        
        if precision_dict is not None and true_goal_indices:
            # Calculate mean precision across all fluents in the goal
            goal_precisions = [precision_dict.get(idx, 0.0) for idx in true_goal_indices]
            precision = sum(goal_precisions) / len(goal_precisions)
            
            # Log detailed precision info for debugging
            # print(f"Goal fluents: {true_goal_indices}")
            # print(f"Individual precisions: {goal_precisions}")
            # print(f"Mean precision: {precision:.4f}")
        
        # Calculate standard binary metrics for logging/debugging
        y_pred_binary = (y_pred > 0.5).float() #todo threshold for considering 1 / 0 rounding
        accuracy = torch.mean((y_pred_binary == y_true_tensor).float()).item()
        
        # Calculate positive and negative confidences as in original function
        conf_pos = torch.mean(y_pred[y_true_tensor == 1]).item() if torch.sum(y_true_tensor) > 0 else 0
        conf_neg = torch.mean(1 - y_pred[y_true_tensor == 0]).item() if torch.sum(y_true_tensor == 0) > 0 else 0
        avg_confidence = (conf_pos + conf_neg) / 2
    
    # Plan passes if BOTH confidence and precision meet thresholds
    is_confident = confidence >= confidence_threshold
    is_precise = precision >= precision_threshold
    passed = is_confident and is_precise
    
    # Return result and metrics for logging
    metrics = {
        "top_goal_idx": int(top_goal_idx),
        "confidence": float(confidence),
        "precision": float(precision),
        "goal_fluents_count": len(true_goal_indices),
        "accuracy": float(accuracy),
        "avg_confidence": float(avg_confidence),
        "is_confident": bool(is_confident),
        "is_precise": bool(is_precise)
    }
    
    return passed, metrics


def build_precision_history(model, val_loader, device):
    """
    Build a precision history dictionary from validation data to use in metacognition.
    Precision = True Positives / (True Positives + False Positives)
    """
    model.eval()
    true_positives = {}
    false_positives = {}
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            y_pred_binary = (output > 0.5).float()  # Convert to binary predictions
            
            # For each goal/fluent, track precision components
            for i in range(target.shape[1]):
                if i not in true_positives:
                    true_positives[i] = 0
                    false_positives[i] = 0
                
                # Where model predicted 1
                pred_pos_mask = y_pred_binary[:, i] == 1
                if torch.any(pred_pos_mask):
                    # True positives: model predicted 1 and actual was 1
                    true_positives[i] += torch.sum((target[pred_pos_mask, i] == 1).float()).item()
                    # False positives: model predicted 1 but actual was 0
                    false_positives[i] += torch.sum((target[pred_pos_mask, i] == 0).float()).item()
    
    # Calculate precision for each goal
    precision_dict = {
        i: true_positives[i] / (true_positives[i] + false_positives[i]) if (true_positives[i] + false_positives[i]) > 0 else 0
        for i in true_positives
    }
    
    return precision_dict


def adaptive_incremental_training(
    train_plans,
    val_plans,
    test_plans,
    action_dict,
    goals_dict,
    target_dir,
    device,
    config
):
    """
    Run adaptive incremental training that focuses on difficult examples.
    
    Args:
        train_plans: List of training plans
        val_plans: List of validation plans
        test_plans: List of test plans
        action_dict: Dictionary mapping actions to indices
        goals_dict: Dictionary mapping goals to indices
        target_dir: Directory to save models and results
        device: Computing device (CPU/GPU)
        config: Configuration dictionary with training parameters
    """
    # Extract parameters from config
    increment = config.get("increment", 16)
    batch_size = config.get("batch_size", 64)
    epochs = config.get("epochs", 30)
    patience = config.get("patience", 5)
    max_dim = config.get("max_dim", 32)
    min_perc = config.get("min_perc", 0.3)
    max_perc = config.get("max_perc", 1.0)
    confidence_threshold = config.get("confidence_threshold", 0.04)
    experience_threshold = config.get("experience_threshold", 0.7)
    review_frequency = config.get("review_frequency", 0)
    review_sample_size = config.get("review_sample_size", batch_size)
    
    # Create validation loader (fixed throughout training)
    val_generator = PlanGeneratorMultiPerc(
        val_plans,
        action_dict,
        goals_dict,
        batch_size=batch_size,
        max_dim=max_dim,
        min_perc=min_perc,
        max_perc=max_perc,
        shuffle=False,
    )
    val_dataset = PlanDataset(val_generator)
    val_loader = DataLoader(val_dataset, batch_size=None, shuffle=False)
    
    # Create test loader if needed
    if test_plans:
        test_generator = PlanGeneratorMultiPerc(
            test_plans,
            action_dict,
            goals_dict,
            batch_size=batch_size,
            max_dim=max_dim,
            min_perc=min_perc,
            max_perc=max_perc,
            shuffle=False,
        )
        test_dataset = PlanDataset(test_generator)
        test_loader = DataLoader(test_dataset, batch_size=None, shuffle=False)
    
    # Initialize model
    model = PlanModel(
        vocab_size=len(action_dict),
        goal_size=len(goals_dict),
        max_dim=max_dim,
        embedding_dim=85,
        lstm_hidden=446,
        dropouti=0.2,
        dropoutw=0.2,
        dropouto=0.2,
    )
    model.to(device)
    
    # Log model architecture
    wandb.log({
        "model_architecture": str(model),
        "model_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
    })
    wandb.watch(model, log="all")
    
    # Initialize trackers
    remaining_plans = train_plans.copy()  # Plans not yet passed evaluation
    passed_plans = []  # Plans that passed evaluation
    trained_plans = []  # Plans that were actually used for training
    iteration = 0
    
    # Initialize precision history
    precision_dict = None
    
    # Main training loop
    while len(remaining_plans) > 0:
        print(f"\n=== Iteration {iteration} ===")
        print(f"Remaining plans: {len(remaining_plans)}, Passed plans: {len(passed_plans)}")
        
        
        # Sample plans for review 
        review_plans = []
        if review_frequency > 0 and iteration > 0 and len(passed_plans) > review_sample_size:
            if iteration % review_frequency == 0:
                review_plans = random.sample(list(passed_plans), review_sample_size)
                print(f"Reviewing {len(review_plans)} previously passed plans")
        
        # Collect failed plans for this iteration
        failed_plans = []
        plans_to_evaluate = remaining_plans.copy()
        
        # Add in any plans for review if needed
        plans_to_evaluate = plans_to_evaluate + review_plans
        
        # Shuffle to ensure randomness in collection
        random.shuffle(plans_to_evaluate)
        
        # Initialize generator for evaluation
        temp_generator = PlanGeneratorMultiPerc(
            plans_to_evaluate,
            action_dict,
            goals_dict,
            batch_size=1,
            max_dim=max_dim,
            min_perc=min_perc,
            max_perc=max_perc,
            shuffle=False,
        )
        
        # Evaluate plans until we have enough failed plans or exhaust the list
        target_fail_count = increment * batch_size
        newly_passed_plans = []
        plan_evaluations = []  # Track evaluation results for logging
        failed_review_counter = 0
        
        # Update precision history every iteration using validation data
        print("Updating precision history from validation data...")
        precision_dict = build_precision_history(model, val_loader, device)
        
        print(f"Evaluating plans until finding {target_fail_count} failures...")
        
        # Evaluate plans with metacognitive approach
        for i, plan_idx in enumerate(range(len(plans_to_evaluate))):
            if len(failed_plans) >= target_fail_count:
                break
                
            plan = plans_to_evaluate[plan_idx]
            plan_data = temp_generator[plan_idx]
            
            passed, metrics = evaluate_plan(
                model, 
                plan_data, 
                goals_dict, 
                device, 
                confidence_threshold=confidence_threshold,
                precision_threshold=experience_threshold,
                precision_dict=precision_dict
            )
            
            # Track metrics 
            plan_evaluations.append({
                "plan_id": plan.plan_name,
                "passed": passed,
                "top_goal": metrics["top_goal_idx"],
                "confidence": metrics["confidence"],
                "precision": metrics["precision"],
                "is_confident": metrics["is_confident"],
                "is_precise": metrics["is_precise"]
            })
            
            if passed:
                if plan in remaining_plans:
                    remaining_plans.remove(plan)
                    passed_plans.append(plan)
                    newly_passed_plans.append(plan)
            else:
                if plan in remaining_plans:
                    failed_plans.append(plan)
                    remaining_plans.remove(plan)
                elif plan in passed_plans and plan in review_plans:
                    # Plan previously passed but now fails review
                    failed_review_counter +=1
                    passed_plans.remove(plan)  # Remove from passed plans
                    remaining_plans.append(plan)  # Add back to remaining plans
                    failed_plans.append(plan)  # Add to failed plans for this iteration
            
            if i % 100 == 0:
                print(f"  Evaluated {i}/{len(plans_to_evaluate)} plans, " 
                      f"failed: {len(failed_plans)}, newly passed: {len(newly_passed_plans)}")
        
        print(f"Finished evaluating plans.")
        print(f"Newly passed plans: {len(newly_passed_plans)}, Failed plans: {len(failed_plans)}")
        print(f"Plans that failed review: {failed_review_counter}")
        
        # Log evaluation statistics
        pass_rate = len(newly_passed_plans) / (len(newly_passed_plans) + len(failed_plans)) if newly_passed_plans or failed_plans else 0
        wandb.log({
            f"iteration_{iteration:03d}/plans_evaluated": len(plan_evaluations),
            f"iteration_{iteration:03d}/plans_passed": len(newly_passed_plans),
            f"iteration_{iteration:03d}/plans_failed": len(failed_plans),
            f"iteration_{iteration:03d}/plans_failed_review": failed_review_counter,
            f"iteration_{iteration:03d}/pass_rate": pass_rate,
            f"iteration_{iteration:03d}/remaining_plans": len(remaining_plans),
            f"iteration_{iteration:03d}/total_passed_plans": len(passed_plans),
            "pass_rate": pass_rate,
        })
                 
        # Set up plans for training - start with failed plans
        plans_for_training = failed_plans.copy()
        
        # Check if we have enough failed plans for training
        if len(plans_for_training) < target_fail_count:
            print(f"Warning: Only found {len(plans_for_training)} failed plans, less than target {target_fail_count}")
            print("Stopping training as we don't have enough failed plans to make an increment.")
            break  # Exit the training loop if we don't have enough failed plans
        
        # If we have enough failed plans, include some previously used plans based on old_plans_factor
        old_plans_factor = config.get("old_plans_factor", 1)  # Default to 20% if not specified
        
        if iteration > 0 and old_plans_factor > 0:
            # Calculate how many old plans to include
            old_plans_count = int(old_plans_factor * len(plans_for_training))
            
            # FIXED: Only consider plans that were actually used in training before
            if trained_plans:  # Check if we have any previously trained plans
                # Sample from previously trained plans
                old_plans_sample = random.sample(trained_plans, min(old_plans_count, len(trained_plans)))
                plans_for_training.extend(old_plans_sample)
                print(f"Added {len(old_plans_sample)} previously used plans ({old_plans_factor:.1%} of current increment)")
                
                # Log information about plan mixing
                wandb.log({
                    f"iteration_{iteration:03d}/failed_plans": len(failed_plans),
                    f"iteration_{iteration:03d}/old_plans_added": len(old_plans_sample),
                    f"iteration_{iteration:03d}/total_training_plans": len(plans_for_training)
                })
        
        # Calculate learning rate for this iteration
        lr = np.linspace(0.001, 0.0001, 20)[min(iteration, 19)]  # Cap at 20 iterations for lr schedule
        
        # Prepare training data with augmentation
        train_generator = PlanGeneratorMultiPercAugmented(
            plans_for_training,
            action_dict,
            goals_dict,
            num_plans=config.get("augmentation_plans", 4),
            batch_size=batch_size,
            max_dim=max_dim,
            min_perc=min_perc,
            max_perc=max_perc,
            add_complete=config.get("use_full_plan", True),
            shuffle=True,
        )
        
        print(f"Training on {len(train_generator.plans)} plans (including augmentation) "
              f"({len(train_generator.plans)/config.get('augmentation_plans', 4)} not augmented)")
        
        train_dataset = PlanDataset(train_generator)
        train_loader = DataLoader(train_dataset, batch_size=None, shuffle=True)
        
        # Train model
        early_stopping = CustomEarlyStopping(patience=patience, iteration=iteration)
        model = train_model(
            model,
            train_loader,
            val_loader,
            epochs,
            device,
            lr,
            early_stopping,
            iteration=iteration,
        )
        
        # Save model
        model_path = path.join(target_dir, f"model_{iteration}.pth")
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
        
        # Evaluate on test set
        if test_plans:
            print("Evaluating on test set for metrics reporting...")
            y_pred, y_true = get_model_predictions(model, test_loader, device)
            scores = print_metrics(
                y_true=y_true,
                y_pred=y_pred,
                dizionario_goal=goals_dict,
                save_dir=target_dir,
                filename=f"metrics_{iteration}",
            )
            
            wandb.log({
                f"iteration_{iteration:03d}/test_accuracy": scores[0],
                f"iteration_{iteration:03d}/test_hamming_loss": scores[1],
            })
        
        # After training, add all plans used in this iteration to trained_plans
        trained_plans.extend([p for p in plans_for_training if p not in trained_plans])
        
        iteration += 1

    # Final report
    print(f"\n=== Training Complete ===")
    print(f"Completed {iteration} iterations")
    final_pass_rate = len(passed_plans)/len(train_plans)
    print(f"Final pass rate: {final_pass_rate:.1%} ({len(passed_plans)}/{len(train_plans)})")
    
    # Log final cumulative pass rate
    wandb.log({
        "final_cumulative_pass_rate": final_pass_rate
    })
    
    return model


if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set seed for reproducibility
    torch.manual_seed(420)
    np.random.seed(420)
    
    # Parameters
    plans_dir = './datasets/gr_logistics/pickles'
    dict_dir = "./datasets/gr_logistics/pickles"
    target_dir = path.join(
        "./datasets/gr_logistics/results/adaptive_incremental",
        datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
    )
    os.makedirs(target_dir, exist_ok=True)
    
    # Configuration
    config = {
        "increment": 16,
        "batch_size": 64,
        "min_perc": 0.3,
        "max_perc": 1.0,
        "max_dim": 32,
        "epochs": 30,
        "patience": 3,
        "augmentation_plans": 4,
        "use_full_plan": True,
        "review_frequency": 1,  # Review passed examples every N iterations
        "review_sample_size": 256,  # Number of passed examples to review
        "confidence_threshold": 0.04,  # difference between top two goal scores must be minimum this value
        "experience_threshold": 0.7,    # avg precision of predicted goal must be minimum this value
        "old_plans_factor": 1,  # %(in relation to size of incremnet) of previously used plans to add to each increment e.g. 0.5 means increment will be 150% standard size where 50% are old plans
    }
    
    # Initialize wandb
    wandb.init(
        project="fsgr-plan-training",
        name=f"adaptive_incremental_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config=config
    )
    
    # Load data
    [action_dict, goals_dict] = load_from_folder(
        dict_dir, ["action_dict.pkl", "goal_dict.pkl"]
    )
    [train_plans, val_plans, test_plans] = load_from_folder(
        plans_dir, ["train_plans", "val_plans", "test_plans"]
    )
    
    print(f"Loaded {len(train_plans)} training plans, {len(val_plans)} validation plans, {len(test_plans)} test plans")
    
    # Log dataset info
    wandb.log({
        "vocab_size": len(action_dict),
        "goal_size": len(goals_dict),
        "train_plans_count": len(train_plans),
        "val_plans_count": len(val_plans),
        "test_plans_count": len(test_plans),
    })
    
    # Run adaptive incremental training
    model = adaptive_incremental_training(
        train_plans,
        val_plans,
        test_plans,
        action_dict,
        goals_dict,
        target_dir,
        device,
        config
    )
    
    # Save final model (actually a copy of the last iteration)
    final_model_path = path.join(target_dir, "model_final.pth")
    torch.save(model.state_dict(), final_model_path)
    
    # Close wandb
    wandb.finish()