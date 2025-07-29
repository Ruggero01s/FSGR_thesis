import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from os import path
import datetime
import warnings
from sklearn.metrics import accuracy_score, hamming_loss, classification_report
from utils_torch import load_from_folder
from plan import Plan
from plan_generator_torch import PlanGeneratorMultiPerc, PlanGeneratorMultiPercAugmented

from grnet_model import PlanModel

import wandb

class PlanDataset(Dataset):
    def __init__(self, generator):
        self.generator = generator
        self.length = len(generator)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x, y = self.generator[idx]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.float32)


def custom_hamming_loss(y_pred, y_true):
    """Custom Hamming Loss equivalent to TensorFlow version"""
    y_pred = y_pred.float()
    y_true = y_true.float()
    diff = torch.abs(y_true - y_pred)
    mismatches = (diff > 0.5).float() #todo threshold to consider 1/0 rounding
    return torch.mean(mismatches)


def custom_precision(y_pred, y_true, threshold=0.5):
    """Custom precision calculation"""
    y_pred_binary = (y_pred > threshold).float()
    y_true_binary = y_true.float()

    true_positives = torch.sum(y_pred_binary * y_true_binary)
    predicted_positives = torch.sum(y_pred_binary)

    if predicted_positives == 0:
        return torch.tensor(0.0)

    precision = true_positives / predicted_positives
    return precision


class CustomEarlyStopping:
    def __init__(self, patience=5, iteration=0):
        self.patience = patience
        self.iteration = iteration
        self.max_prec = -1
        self.min_val_loss = 1000
        self.best_weights = None
        self.increased_epochs = 0

    def __call__(self, val_loss, val_precision, loss, model):
        new_best = False

        if self.iteration > 5:
            # Prioritize precision after iteration 5
            if val_precision > self.max_prec and val_loss < 1 and loss < 1:
                message = f"New best model found with precision {val_precision:.4f} and loss {val_loss:.4f}"
                print(message)
                wandb.log({"early_stopping_event": message})
                self.max_prec = val_precision
                self.min_val_loss = val_loss
                self.best_weights = model.state_dict().copy()
                self.increased_epochs = 0
                new_best = True
            elif (
                abs(val_precision - self.max_prec) < 1e-5
                and val_loss < self.min_val_loss
            ):
                message = f"New best model found with precision {val_precision:.4f} and loss {val_loss:.4f}"
                print(message)
                wandb.log({"early_stopping_event": message})
                self.min_val_loss = val_loss
                self.best_weights = model.state_dict().copy()
                self.increased_epochs = 0
                new_best = True
        else:
            # Prioritize loss for first iterations
            if val_loss < self.min_val_loss and loss < 1 and val_loss < 1:
                message = f"New best model found with loss {val_loss:.4f}"
                print(message)
                wandb.log({"early_stopping_event": message})
                self.min_val_loss = val_loss
                self.best_weights = model.state_dict().copy()
                self.increased_epochs = 0
                new_best = True

        if not new_best:
            self.increased_epochs += 1

        return self.increased_epochs >= self.patience


def train_model(
    model, train_loader, val_loader, epochs, device, lr, early_stopping=None, iteration=None
):
    """Train the model"""
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_hamming = 0.0
        train_precision = 0.0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_hamming += custom_hamming_loss(output, target).item()
            train_precision += custom_precision(output, target).item()

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_hamming = 0.0
        val_precision = 0.0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                val_hamming += custom_hamming_loss(output, target).item()
                val_precision += custom_precision(output, target).item()

        # Calculate averages
        train_loss /= len(train_loader)
        train_hamming /= len(train_loader)
        train_precision /= len(train_loader)
        val_loss /= len(val_loader)
        val_hamming /= len(val_loader)
        val_precision /= len(val_loader)

        print(f"Epoch {epoch+1}/{epochs}:")
        print(
            f"  Train Loss: {train_loss:.4f}, Train Hamming: {train_hamming:.4f}, Train Precision: {train_precision:.4f}"
        )
        print(
            f"  Val Loss: {val_loss:.4f}, Val Hamming: {val_hamming:.4f}, Val Precision: {val_precision:.4f}"
        )

        # Log to wandb
        log_dict = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_hamming": train_hamming,
            "train_precision": train_precision,
            "val_loss": val_loss,
            "val_hamming": val_hamming,
            "val_precision": val_precision,
            "learning_rate": lr,
        }
        
        if iteration is not None:
            log_dict["iteration"] = iteration
            
        wandb.log(log_dict)

        # Early stopping
        if early_stopping and early_stopping(
            val_loss, val_precision, train_loss, model
        ):
            print(f"Early stopping triggered at epoch {epoch+1}")
            wandb.log({"early_stopping_epoch": epoch + 1})
            if early_stopping.best_weights is not None:
                model.load_state_dict(early_stopping.best_weights)
                print("Best model weights restored")
            break

    return model


def get_model_predictions(model, test_loader, device):
    """Get predictions from the model"""
    model.eval()
    y_pred = []
    y_true = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            y_pred.extend(output.cpu().numpy())
            y_true.extend(target.cpu().numpy())

    return y_pred, y_true


def print_metrics(y_true, y_pred, dizionario_goal, save_dir=None, filename="metrics"):
    """Print and save metrics"""
    # Convert predictions to binary
    for i, y in enumerate(y_pred):
        y_pred[i] = [0 if pred < 0.5 else 1 for pred in y]

    labels = list(dizionario_goal.keys())
    to_print = []
    accuracy = accuracy_score(y_true, y_pred)
    hamming_loss_score = hamming_loss(y_true, y_pred)
    to_print.append(f"Accuracy: {accuracy}\n")
    to_print.append(f"Hamming Loss: {hamming_loss_score}\n")
    to_print.append(classification_report(y_true, y_pred, target_names=labels))

    if save_dir is None:
        for line in to_print:
            print(line)
    else:
        with open(path.join(save_dir, f"{filename}.txt"), "w") as file:
            for line in to_print:
                file.write(line)

    return [accuracy, hamming_loss_score]


if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set seed for reproducibility
    torch.manual_seed(420)
    np.random.seed(420)
    warnings.filterwarnings("ignore")

    # Parameters
    plans_dir = './datasets/gr_logistics/pickles'
    # plans_dir = "datasets/logistics/optimal_plans/plans_max-plan-dim=30_train_percentage=0.8"
    dict_dir = "./datasets/gr_logistics/pickles"
    target_dir = path.join(
        "./datasets/gr_logistics/results/pytorch_incremental",
        datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
    )
    logs_dir = path.join(target_dir, "logs")
    temp_dir = path.join(target_dir, "temp")

    # Control flow flags:
    test = False  # Whether to run a quick test of training logic
    train = True  # Enable the training loop
    results = False  # Whether to collect final results over saved iterations
    live_test = True  # Perform evaluation on test set after each iteration
    random_model = False  # If True, use random model for baseline
    incremental = True  # Enable incremental training

    # Incremental training parameters:
    increment = 16  # Number of batches per iteration
    batch_size = 64  # Number of samples per batch
    old_plans_percentage = 1  # Fraction of previous plans to include
    min_perc = 0.3  # Minimum fraction of plan to use
    max_perc = 1  # Maximum fraction of plan to use
    max_dim = 32  # Maximum plan length #! needs to be set appropriately for the dataset
    epochs = 30  # Epochs per iteration
    patience = 3  # Early stopping patience

    # Data augmentation:
    augmentation_plans = 4  # Number of plans to augment per batch
    use_full_plan = True  # Whether to include complete plan in augmentation

    
    
    # Initialize wandb
    wandb.init(
        project="fsgr-plan-training",
        name=f"incremental_{'test' if test else 'full'}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config={
            "increment": increment,
            "batch_size": batch_size,
            "old_plans_percentage": old_plans_percentage,
            "min_perc": min_perc,
            "max_perc": max_perc,
            "max_dim": max_dim,
            "epochs": epochs,
            "patience": patience,
            "augmentation_plans": augmentation_plans,
            "use_full_plan": use_full_plan,
            "incremental": incremental,
            "test_mode": test,
            "device": str(device),
            "random_model": random_model,
        }
    )

    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)

    # Load data
    [action_dict, goals_dict] = load_from_folder(
        dict_dir, ["action_dict.pkl", "goal_dict.pkl"]
    )

    # Log dataset info
    wandb.log({
        "vocab_size": len(action_dict),
        "goal_size": len(goals_dict),
    })
    if live_test or results:
        [test_plans] = load_from_folder(plans_dir, ["test_plans"])
        test_generator = PlanGeneratorMultiPerc(
            test_plans,
            action_dict,
            goals_dict,
            batch_size,
            max_dim,
            min_perc,
            max_perc,
            shuffle=False,
        )
        test_dataset = PlanDataset(test_generator)
        test_loader = DataLoader(test_dataset, batch_size=None, shuffle=False)

    if train:
        [train_plans, val_plans] = load_from_folder(
            plans_dir, ["train_plans", "val_plans"]
        )
        print(
            f"Training on {len(train_plans)} plans, validation on {len(val_plans)} plans"
        )

        # Log dataset sizes
        wandb.log({
            "train_plans_count": len(train_plans),
            "val_plans_count": len(val_plans),
        })

        # Create validation loader
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

        model = None
        if test:
            iterations = 2
        else:
            iterations = len(train_plans) // (increment * batch_size)

        print(f"Iterations: {iterations}")
        wandb.log({"total_iterations": iterations})

        if incremental:
            for iteration in range(iterations):
                print(f"Iteration: {iteration}")

                # Calculate subset of training plans
                start_index = iteration * increment * batch_size
                end_index = start_index + increment * batch_size
                train_plans_subset = train_plans[start_index:end_index]

                # Learning rate schedule
                lr = np.linspace(0.001, 0.0001, iterations)[iteration]

                # Log iteration info
                wandb.log({
                    "iteration": f"{iteration:03d}",
                    "current_learning_rate": lr,
                    "train_plans_subset_size": len(train_plans_subset),
                })


                # Add old plans if not first iteration
                if start_index > 0:
                    old_plans_count = int(old_plans_percentage * increment * batch_size)
                    old_plans = np.random.choice(
                        train_plans[0:start_index], old_plans_count, replace=False
                    )
                    train_plans_subset.extend(old_plans)

                np.random.shuffle(train_plans_subset)

                # Create training generator
                train_generator = PlanGeneratorMultiPercAugmented(
                    train_plans_subset,
                    action_dict,
                    goals_dict,
                    num_plans=augmentation_plans,
                    batch_size=batch_size,
                    max_dim=max_dim,
                    min_perc=min_perc,
                    max_perc=max_perc,
                    add_complete=use_full_plan,
                    shuffle=True,
                )

                print(
                    f"Iteration {iteration} - Training on {len(train_generator.plans)} plans, in {train_generator.num_batches} batches"
                )

                # Log training info
                wandb.log({
                    f"iteration_{iteration:03d}/total_plans": len(train_generator.plans),
                    f"iteration_{iteration:03d}/num_batches": train_generator.num_batches,
                })

                # Create or load model
                if iteration == 0:
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
                    print(f"Model created")
                    
                    wandb.log({
                        "model_architecture": str(model),
                        "model_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                    })
                    
                    # Log model info
                    wandb.watch(model, log="all")
                else:
                    # Load previous model
                    model_path = path.join(target_dir, f"model_{iteration-1}.pth")
                    model.load_state_dict(torch.load(model_path))
                    print(f"Model loaded from {model_path}")

                # Create data loader
                train_dataset = PlanDataset(train_generator)
                train_loader = DataLoader(train_dataset, batch_size=None, shuffle=True)

                # Train model
                early_stopping = CustomEarlyStopping(
                    patience=patience, iteration=iteration
                )
                if not random_model:
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

                # Live test
                if live_test:
                    y_pred, y_true = get_model_predictions(model, test_loader, device)
                    scores = print_metrics(
                        y_true=y_true,
                        y_pred=y_pred,
                        dizionario_goal=goals_dict,
                        save_dir=target_dir,
                        filename=f"metrics_{iteration}",
                    )
                    
                    # Log test metrics
                    wandb.log({
                        f"iteration_{iteration:03d}/test_accuracy": scores[0],
                        f"iteration_{iteration:03d}/test_hamming_loss": scores[1],
                    })
        else:
            # Non-incremental training
            lr = 0.0001
            np.random.shuffle(train_plans)

            train_generator = PlanGeneratorMultiPerc(
                train_plans,
                action_dict,
                goals_dict,
                batch_size=batch_size,
                max_dim=max_dim,
                min_perc=min_perc,
                max_perc=max_perc,
                shuffle=True,
            )

            model = PlanModel(
                vocab_size=len(action_dict), goal_size=len(goals_dict), max_dim=max_dim
            )
            model.to(device)

            train_dataset = PlanDataset(train_generator)
            train_loader = DataLoader(train_dataset, batch_size=None, shuffle=True)

            early_stopping = CustomEarlyStopping(patience=patience, iteration=0)
            if not random_model:
                model = train_model(
                    model, train_loader, val_loader, epochs, device, lr, early_stopping
                )

            # Save model
            model_path = path.join(target_dir, "model_FULL.pth")
            torch.save(model.state_dict(), model_path)

            if live_test:
                y_pred, y_true = get_model_predictions(model, test_loader, device)
                scores = print_metrics(
                    y_true=y_true,
                    y_pred=y_pred,
                    dizionario_goal=goals_dict,
                    save_dir=target_dir,
                    filename="metrics_FULL",
                )

    if results:
        start_iteration = 0
        end_iteration = 18
        model_path_template = path.join(
            target_dir, "model_{}.pth"
        )  # Update this path as needed

        for iteration in range(start_iteration, end_iteration + 1):
            model_path = model_path_template.format(iteration)
            model = PlanModel(
                vocab_size=len(action_dict), goal_size=len(goals_dict), max_dim=max_dim
            )
            model.load_state_dict(torch.load(model_path))
            model.to(device)

            y_pred, y_true = get_model_predictions(model, test_loader, device)
            scores = print_metrics(
                y_true=y_true,
                y_pred=y_pred,
                dizionario_goal=goals_dict,
                save_dir=path.dirname(model_path),
                filename=f"metrics_{iteration}",
            )
    
    # Close wandb run
    wandb.finish()