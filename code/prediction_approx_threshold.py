import click
import torch
import numpy as np
import matplotlib.pyplot as plt
from os import path
import os
import seaborn as sns
from utils_torch import load_from_folder
from plan_generator_torch import PlanGenerator
from get_predictions_and_results import PlanModel

def analyze_prediction_distributions(model, data_generator, device, num_samples=100):
    """
    Analyze the distribution of prediction values to help determine an optimal threshold.
    
    Args:
        model: The trained model
        data_generator: Generator providing data samples
        device: Device to run inference on
        num_samples: Number of samples to analyze
    
    Returns:
        true_positive_values: List of prediction values for actual positives
        true_negative_values: List of prediction values for actual negatives
    """
    true_positive_values = []
    true_negative_values = []
    
    model.eval()
    samples_processed = 0
    
    print("Analyzing prediction distributions...")
    
    for i in range(min(num_samples, len(data_generator))):
        x, y_true = data_generator.__getitem__(i)
        x_tensor = torch.tensor(x, dtype=torch.long).to(device)
        
        with torch.no_grad():
            y_pred = model(x_tensor)
            y_pred_np = y_pred.cpu().numpy()[0]
        
        # For each element in the prediction vector
        for j in range(len(y_pred_np)):
            if y_true[0][j] == 1:  # True positive
                true_positive_values.append(y_pred_np[j])
            else:  # True negative
                true_negative_values.append(y_pred_np[j])
        
        samples_processed += 1
        if samples_processed >= num_samples:
            break
    
    print(f"Analysis complete. Collected {len(true_positive_values)} positive values and {len(true_negative_values)} negative values.")
    
    return true_positive_values, true_negative_values

def plot_distributions(true_positive_values, true_negative_values, save_path=None, title=None):
    """Plot the distribution of positive and negative prediction values"""
    plt.figure(figsize=(12, 6))
    
    sns.histplot(true_positive_values, kde=True, stat="density", color="green", alpha=0.5, label="True Positives")
    sns.histplot(true_negative_values, kde=True, stat="density", color="red", alpha=0.5, label="True Negatives")
    
    plt.xlabel('Prediction Value')
    plt.ylabel('Density')
    plot_title = title if title else 'Distribution of Model Prediction Values'
    plt.title(plot_title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    
    plt.show()

def calculate_metrics_by_threshold(true_positive_values, true_negative_values):
    """Calculate precision, recall, F1 score, and accuracy for different thresholds"""
    all_values = np.array(true_negative_values + true_positive_values)
    all_labels = np.array([0] * len(true_negative_values) + [1] * len(true_positive_values))
    
    thresholds = np.arange(0.01, 1.0, 0.01)
    metrics = []
    
    for threshold in thresholds:
        predictions = (all_values >= threshold).astype(int)
        
        # Calculate metrics
        tp = np.sum((predictions == 1) & (all_labels == 1))
        fp = np.sum((predictions == 1) & (all_labels == 0))
        tn = np.sum((predictions == 0) & (all_labels == 0))
        fn = np.sum((predictions == 0) & (all_labels == 1))
        
        accuracy = np.mean(predictions == all_labels)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics.append({
            'threshold': threshold,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
    
    return metrics

def plot_metrics(metrics, save_path=None, title=None):
    """Plot metrics against threshold values"""
    thresholds = [m['threshold'] for m in metrics]
    accuracy = [m['accuracy'] for m in metrics]
    precision = [m['precision'] for m in metrics]
    recall = [m['recall'] for m in metrics]
    f1 = [m['f1'] for m in metrics]
    
    plt.figure(figsize=(12, 8))
    
    plt.plot(thresholds, accuracy, 'b-', label='Accuracy')
    plt.plot(thresholds, precision, 'g-', label='Precision')
    plt.plot(thresholds, recall, 'r-', label='Recall')
    plt.plot(thresholds, f1, 'y-', label='F1 Score')
    
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plot_title = title if title else 'Performance Metrics vs Threshold'
    plt.title(plot_title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Find best thresholds
    best_accuracy_idx = np.argmax(accuracy)
    best_f1_idx = np.argmax(f1)
    
    plt.axvline(x=thresholds[best_accuracy_idx], color='blue', linestyle='--', alpha=0.5,
                label=f'Best Accuracy: {thresholds[best_accuracy_idx]:.2f}')
    plt.axvline(x=thresholds[best_f1_idx], color='yellow', linestyle='--', alpha=0.5,
                label=f'Best F1: {thresholds[best_f1_idx]:.2f}')
    
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Metrics plot saved to {save_path}")
    
    plt.show()
    
    return thresholds[best_accuracy_idx], thresholds[best_f1_idx]

def print_threshold_recommendations(metrics, set_name=None):
    """Print recommended thresholds based on different metrics"""
    best_accuracy_idx = np.argmax([m['accuracy'] for m in metrics])
    best_precision_idx = np.argmax([m['precision'] for m in metrics])
    best_recall_idx = np.argmax([m['recall'] for m in metrics])
    best_f1_idx = np.argmax([m['f1'] for m in metrics])
    
    prefix = f"[{set_name}] " if set_name else ""
    print(f"\n{prefix}Recommended thresholds:")
    print(f"{prefix}For best accuracy ({metrics[best_accuracy_idx]['accuracy']:.4f}): {metrics[best_accuracy_idx]['threshold']:.4f}")
    print(f"{prefix}For best precision ({metrics[best_precision_idx]['precision']:.4f}): {metrics[best_precision_idx]['threshold']:.4f}")
    print(f"{prefix}For best recall ({metrics[best_recall_idx]['recall']:.4f}): {metrics[best_recall_idx]['threshold']:.4f}")
    print(f"{prefix}For best F1 score ({metrics[best_f1_idx]['f1']:.4f}): {metrics[best_f1_idx]['threshold']:.4f}")
    
    # Find balanced threshold (good combination of precision and recall)
    balanced_score = [(m['precision'] + m['recall']) / 2 for m in metrics]
    balanced_idx = np.argmax(balanced_score)
    print(f"{prefix}For balanced precision/recall: {metrics[balanced_idx]['threshold']:.4f}")
    
    return {
        'accuracy': metrics[best_accuracy_idx]['threshold'],
        'precision': metrics[best_precision_idx]['threshold'],
        'recall': metrics[best_recall_idx]['threshold'],
        'f1': metrics[best_f1_idx]['threshold'],
        'balanced': metrics[balanced_idx]['threshold']
    }

def analyze_test_set(model, test_set, dizionario, dizionario_goal, 
                     device, max_dim, perc_action, num_samples, 
                     output_dir=None, set_name=None):
    """Analyze a single test set"""
    print(f"\n{'='*80}\nAnalyzing test set: {set_name}\n{'='*80}")
    
    # Create data generator
    gen = PlanGenerator(
        test_set,
        dizionario,
        dizionario_goal,
        1,
        max_dim,
        perc_action,
        shuffle=True,
    )
    
    print(f"Data generator created with {len(gen)} samples")
    
    # Calculate distribution statistics
    true_positives, true_negatives = analyze_prediction_distributions(
        model, gen, device, num_samples=min(num_samples, len(gen))
    )
    
    # Skip if we didn't get any predictions
    if not true_positives or not true_negatives:
        print(f"Warning: Not enough data in test set {set_name}. Skipping analysis.")
        return None
    
    # Print statistics
    print(f"\n[{set_name}] True positive predictions - Mean: {np.mean(true_positives):.4f}, Median: {np.median(true_positives):.4f}")
    print(f"[{set_name}] Min: {min(true_positives):.4f}, Max: {max(true_positives):.4f}")
    print(f"[{set_name}] 5th percentile: {np.percentile(true_positives, 5):.4f}, 95th percentile: {np.percentile(true_positives, 95):.4f}")
    
    print(f"\n[{set_name}] True negative predictions - Mean: {np.mean(true_negatives):.4f}, Median: {np.median(true_negatives):.4f}")
    print(f"[{set_name}] Min: {min(true_negatives):.4f}, Max: {max(true_negatives):.4f}")
    print(f"[{set_name}] 5th percentile: {np.percentile(true_negatives, 5):.4f}, 95th percentile: {np.percentile(true_negatives, 95):.4f}")
    
    # Calculate metrics for different thresholds
    metrics = calculate_metrics_by_threshold(true_positives, true_negatives)
    threshold_recommendations = print_threshold_recommendations(metrics, set_name)
    
    # Plotting
    if output_dir:
        dist_plot_path = path.join(output_dir, f"distribution_model_set_{set_name}_perc_{perc_action:.1f}.png")
        metrics_plot_path = path.join(output_dir, f"metrics_model_set_{set_name}_perc_{perc_action:.1f}.png")
        
        # Plot distributions
        plot_distributions(
            true_positives, 
            true_negatives, 
            save_path=dist_plot_path, 
            title=f"Distribution of Model Prediction Values - Set {set_name}"
        )
        
        # Plot metrics
        best_acc_thresh, best_f1_thresh = plot_metrics(
            metrics, 
            save_path=metrics_plot_path,
            title=f"Performance Metrics vs Threshold - Set {set_name}"
        )
    else:
        # Plot distributions
        plot_distributions(
            true_positives, 
            true_negatives, 
            title=f"Distribution of Model Prediction Values - Set {set_name}"
        )
        
        # Plot metrics
        best_acc_thresh, best_f1_thresh = plot_metrics(
            metrics,
            title=f"Performance Metrics vs Threshold - Set {set_name}"
        )
    
    return {
        'set_name': set_name,
        'true_positives': true_positives,
        'true_negatives': true_negatives,
        'metrics': metrics,
        'recommendations': threshold_recommendations,
        'best_accuracy_threshold': best_acc_thresh,
        'best_f1_threshold': best_f1_thresh
    }

def plot_combined_metrics(all_results, output_dir=None, model_number=None, perc_action=None):
    """Plot combined metrics for all test sets"""
    plt.figure(figsize=(14, 10))
    
    # Plot F1 scores for each test set
    for result in all_results:
        set_name = result['set_name']
        metrics = result['metrics']
        thresholds = [m['threshold'] for m in metrics]
        f1_scores = [m['f1'] for m in metrics]
        plt.plot(thresholds, f1_scores, label=f"Set {set_name} F1")
    
    # Add legend and labels
    plt.xlabel('Threshold')
    plt.ylabel('F1 Score')
    plt.title('F1 Scores Across All Test Sets')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if output_dir:
        save_path = path.join(output_dir, f"combined_metrics_model_{model_number}_perc_{perc_action:.1f}.png")
        plt.savefig(save_path)
        print(f"Combined metrics plot saved to {save_path}")
    
    plt.show()

def aggregate_results(all_results):
    """Aggregate results from all test sets to provide overall recommendations"""
    # Combine all true positives and true negatives
    all_true_positives = []
    all_true_negatives = []
    
    for result in all_results:
        all_true_positives.extend(result['true_positives'])
        all_true_negatives.extend(result['true_negatives'])
    
    # Calculate overall metrics
    overall_metrics = calculate_metrics_by_threshold(all_true_positives, all_true_negatives)
    
    # Get recommendations
    recommendations = print_threshold_recommendations(overall_metrics, set_name="OVERALL")
    
    # Summarize individual test set recommendations
    print("\nSummary of threshold recommendations by test set:")
    for result in all_results:
        set_name = result['set_name']
        recs = result['recommendations']
        print(f"Set {set_name}: Accuracy={recs['accuracy']:.4f}, F1={recs['f1']:.4f}, Balanced={recs['balanced']:.4f}")
        
    return all_true_positives, all_true_negatives, overall_metrics, recommendations

@click.command()
@click.option('--model-path', '-m', type=click.Path(exists=True), required=True, 
              help='Path to the folder containing the model')
@click.option('--model-number', '-n', type=int, required=True,
              help='Model number to analyze')
@click.option('--dict-dir', '-d', type=click.Path(exists=True), required=True,
              help='Directory containing dictionaries (action_dict.pkl and goal_dict.pkl)')
@click.option('--test-plans-dir', '-t', type=click.Path(exists=True), required=True,
              help='Directory containing test plan files')
@click.option('--num-samples', '-s', type=int, default=50,
              help='Number of samples to analyze per test set')
@click.option('--max-dim', type=int, default=100,
              help='Maximum plan dimension')
@click.option('--perc-action', '-p', type=float, default=0.5,
              help='Percentage of plan actions to use (0.3, 0.5, 0.7)')
@click.option('--output-dir', '-o', type=click.Path(), default=None,
              help='Directory to save plots (optional)')
def main(model_path, model_number, dict_dir, test_plans_dir, num_samples, 
         max_dim, perc_action, output_dir):
    """
    Analyze model prediction thresholds to optimize binary classification across all test sets.
    
    This script loads a trained model and evaluates its predictions on all available test data
    to find optimal thresholds for converting continuous predictions to binary values.
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dictionaries
    try:
        [dizionario, dizionario_goal] = load_from_folder(
            dict_dir, ["action_dict.pkl", "goal_dict.pkl"]
        )
        print("Dictionaries loaded")
    except Exception as e:
        print(f"Error loading dictionaries: {e}")
        return
    
    # Load model
    model_file = path.join(model_path, f"model_{model_number}.pth")
    try:
        model = PlanModel(
            vocab_size=len(dizionario),
            goal_size=len(dizionario_goal),
            max_dim=max_dim,
        )
        model.load_state_dict(torch.load(model_file, map_location=device))
        model.to(device)
        model.eval()
        print(f"Model loaded successfully: {model_file}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Load test plans
    filenames = os.listdir(test_plans_dir)
    try:
        test_plans = load_from_folder(test_plans_dir, filenames)
        print(f"Test plans loaded: {len(test_plans)} files")
    except Exception as e:
        print(f"Error loading test plans: {e}")
        return
    
    # Create output directory if needed
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Analyze each test set
    all_results = []
    for i, test_set in enumerate(test_plans):
        if len(test_set) == 0:
            print(f"Skipping empty test set {i}")
            continue
            
        set_name = filenames[i].split('.')[0] if i < len(filenames) else f"set_{i}"
        #if want test set plot pass output_dir
        result = analyze_test_set(model=model,
                                  test_set=test_set,
                                  dizionario=dizionario,
                                  dizionario_goal=dizionario_goal,
                                  device=device,
                                  max_dim=max_dim,
                                  perc_action=perc_action,
                                  num_samples=num_samples,
                                #   output_dir=output_dir,
                                  set_name=set_name)
        
        if result:
            all_results.append(result)
    
    # Plot combined metrics
    if len(all_results) > 1:
        plot_combined_metrics(all_results, output_dir, model_number, perc_action)
    
    # Aggregate and summarize results
    all_true_positives, all_true_negatives, overall_metrics, recommendations = aggregate_results(all_results)
    
    # Plot overall distributions and metrics
    if output_dir:
        dist_plot_path = path.join(output_dir, f"distribution_model_{model_number}_OVERALL_perc_{perc_action:.1f}.png")
        metrics_plot_path = path.join(output_dir, f"metrics_model_{model_number}_OVERALL_perc_{perc_action:.1f}.png")
        
        plot_distributions(
            all_true_positives, 
            all_true_negatives, 
            save_path=dist_plot_path, 
            title="Distribution of Model Prediction Values - All Test Sets"
        )
        
        best_acc_thresh, best_f1_thresh = plot_metrics(
            overall_metrics, 
            save_path=metrics_plot_path,
            title="Performance Metrics vs Threshold - All Test Sets"
        )
    else:
        plot_distributions(
            all_true_positives, 
            all_true_negatives, 
            title="Distribution of Model Prediction Values - All Test Sets"
        )
        
        best_acc_thresh, best_f1_thresh = plot_metrics(
            overall_metrics,
            title="Performance Metrics vs Threshold - All Test Sets"
        )
    
    print("\nFINAL SUMMARY:")
    print(f"Analyzed {len(all_true_positives)} positive predictions and {len(all_true_negatives)} negative predictions across all test sets")
    print(f"Best accuracy threshold: {recommendations['accuracy']:.4f}")
    print(f"Best F1 score threshold: {recommendations['f1']:.4f}")
    print(f"Best balanced threshold: {recommendations['balanced']:.4f}")

if __name__ == "__main__":
    main()