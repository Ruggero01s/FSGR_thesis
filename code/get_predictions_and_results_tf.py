import argparse
from plan_generator import PlanGeneratorMultiPerc, PlanGenerator
import tensorflow as tf
from keras.models import load_model
from utils import load_from_folder, AttentionWeights, ContextVector
from incremental_model_training import Custom_Hamming_Loss1
import os
import json
import numpy as np
from plan import Plan
import time
from os import path

def get_precision_test(model_path: str, model_number: int):
    """Load precision test results from metrics file"""
    try:
        metrics_file = path.join(
            model_path, f"metrics_{model_number}.txt"
        )
        with open(metrics_file, "r") as f:
            lines = f.readlines()

        start = False
        values = list()
        for line in lines:
            if line.strip() == "":
                if start:
                    break
                else:
                    start = True
                    continue
            if start:
                values.append(float(line.rsplit()[-4]))
        return values
    except FileNotFoundError:
        print(f"Metrics file not found: {metrics_file}")
        return None


def get_score(prediction: np.ndarray, possible_goal: list) -> tuple:
    """
    Returns the score for a possible goal.

    Args:
        prediction: An array that contains the model prediction.
        possible_goal: A list that contains the possible goal indexes.

    Returns:
        A tuple (score, count) representing the score and count of non-zero predictions.
    """
    score = 0
    count = 0
    for index in possible_goal:
        p = prediction[int(index)]
        if p != 0:
            count += 1
        score += p
    return score, count


def get_max(scores: np.ndarray) -> list:
    """
    Returns a list with the index (or indexes) of the highest scores.

    Args:
        scores: An array that contains the scores as floats.

    Returns:
        A list that contains the indexes of the highest score.
    """
    max_element = -1
    index_max = list()
    for i in range(len(scores)):
        if scores[i] > max_element:
            max_element = scores[i]
            index_max = [i]
        elif scores[i] == max_element:
            index_max.append(i)
    return index_max


def get_scores(
    prediction: np.ndarray, possible_goals: dict, normalize: bool = False
) -> np.ndarray:
    """
    Returns the scores for all possible goals.

    Args:
        prediction: An array that contains the model prediction.
        possible_goals: A dict of possible goals; each possible goal is represented as a list.
        normalize: Whether to normalize scores by count.

    Returns:
        An array that contains the score of each of the possible goals.
    """
    try:
        scores = np.zeros((max(possible_goals) + 1,), dtype=float)
        for index in possible_goals:
            scores[index], count = get_score(prediction, list(possible_goals[index]))
            if normalize and count > 0:
                scores[index] /= count
        return scores
    except (IndexError, ValueError) as e:
        print(f"Error in get_scores: {e}")
        print(f"Prediction shape: {prediction.shape}")
        print(f"Possible goals: {possible_goals}")
        return None


def get_result(scores: np.ndarray, correct_goal: int) -> bool:
    """
    Computes if the goal recognition task is successful.

    Args:
        scores: An array of floats that contains a score for each possible goal.
        correct_goal: An integer that represents the index of the correct goal.

    Returns:
        True if the maximum score index corresponds to the correct goal index, False otherwise.
    """
    idx_max_list = get_max(scores)
    if len(idx_max_list) == 1:
        idx_max = idx_max_list[0]
    else:
        print(f"Algorithm chose randomly one of {len(idx_max_list)} equal candidates.")
        idx_max = idx_max_list[np.random.randint(0, len(idx_max_list))]
    return idx_max == correct_goal


def get_correct_goal_idx(correct_goal: list, possible_goals: list) -> int:
    """
    Computes the correct goal index.

    Args:
        correct_goal: A list of strings that contains the correct goal fluents.
        possible_goals: A list of possible goals; each possible goal is represented as a list.

    Returns:
        The index of the correct goal in the possible goals list.
        None if the possible goal list does not contain the correct goal.
    """
    for index, possible_goal in enumerate(possible_goals):
        possible_goal = np.sort(possible_goal)
        correct_goal = np.sort(correct_goal)
        if np.all(possible_goal == correct_goal):
            return index
    return None


def save_results_json(
    target_file, results, filename, model_name, percentage, time, goals
):
    """Save results to JSON file"""
    scores = results[2]
    pred_scores = results[3]
    scores_dict = dict()
    pred_dict = dict()
    goals_dict = dict()

    if scores is not None:
        for i, s in enumerate(scores):
            scores_dict[i] = f"{s:.5f}"
    else:
        scores_dict = None

    if pred_scores is not None:
        for i, s in enumerate(pred_scores):
            pred_dict[i] = f"{s:.5f}"
    else:
        pred_dict = None

    if goals is not None:
        for k in goals:
            s = "".join(f"{str(e)} " for e in goals[k])
            goals_dict[k] = s.strip()
    else:
        goals_dict = None

    json_dict = {
        "INSTANCE": filename.split(",", 1)[0],
        "DOMAIN": filename + f"_constraint{percentage}",
        "MODEL": model_name,
        "PREDICTED": results[0],
        "ACTUAL": results[1],
        "SCORES": scores_dict,
        "PRED": pred_dict,
        "TOTALRUNTIME": time,
        "GOALS": goals_dict,
    }

    with open(target_file, "a") as af:
        af.writelines(f"{json.dumps(json_dict)}\n")


def get_goal(plan: Plan, goals_dict: dict) -> tuple:
    """Extract goal from plan"""
    goals = plan.goals
    new_goal = list()
    for g in goals:
        goals_vector = goals_dict[g]
        index = np.argmax(goals_vector)
        new_goal.append(index)
    new_goal = np.sort(new_goal)
    return tuple(new_goal)


def get_goal_number(plan_name: str) -> int:
    """Extract goal number from plan name"""
    goal_number = plan_name.split("hyp=hyp-")[1]
    goal_number = int(goal_number.split(".", 1)[0])
    return goal_number


def get_goals(plans: list, dizionario_goal: dict) -> dict:
    """Get all goals from plans"""
    goals_dict = dict()
    for plan in plans:
        plan_name = plan.plan_name
        goal_number = get_goal_number(plan_name)
        new_goal = get_goal(plan, dizionario_goal)
        goals_dict[goal_number] = new_goal
    return goals_dict


def get_all_model_numbers(model_path):
    """Discover all model files in the model_path and return their numbers."""
    model_numbers = []
    for file in os.listdir(model_path):
        if file.startswith("model_") and file.endswith(".keras"):
            try:
                model_num = int(file.replace("model_", "").replace(".keras", ""))
                model_numbers.append(model_num)
            except ValueError:
                continue
    return sorted(model_numbers)


def run(
    model_path,
    model_number,
    read_dict_dir,
    read_test_plans_dir,
    target_dir,
    problems_dir,
    max_dim,
    max_plan_perc,
    threshold,
    batch_size=32,  # Add batch processing
):
    """Main function to run goal recognition predictions"""

    # Set up GPU memory growth
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
            print(f"GPU detected: {physical_devices[0].name}")
        except:
            print("Invalid device or cannot modify virtual devices once initialized")
    else:
        print("No GPU detected! Running on CPU will be much slower.")

    # Load dictionaries
    try:
        [dizionario, dizionario_goal] = load_from_folder(
            read_dict_dir, ["action_dict.pkl", "goal_dict.pkl"]
        )
        print("Dictionaries loaded")
    except Exception as e:
        print(f"Error loading dictionaries: {e}")
        return

    print("Model number:", model_number)
    if model_number == "ALL":
        # Get all models in the directory
        model_numbers = get_all_model_numbers(model_path)
        if not model_numbers:
            print("No models found in the specified directory.")
            raise FileNotFoundError("Model files not found in the specified directory.")
    else:
        # Use the specified model number
        model_numbers = [int(model_number)]
        if not path.exists(path.join(model_path, f"model_{model_numbers[0]}.keras")):
            print(f"Model file model_{model_numbers[0]}.keras does not exist in the specified directory.")
            raise FileNotFoundError(f"Model file model_{model_numbers[0]}.keras not found.")
    
    # Load test plans
    filenames = os.listdir(read_test_plans_dir)
    try:
        test_plans = load_from_folder(read_test_plans_dir, filenames)
        print(f"Test plans loaded: {len(test_plans)} files")
    except Exception as e:
        print(f"Error loading test plans: {e}")
        return
        
    # Calculate max dimension based on percentage
    adj_max_dim = int(max_plan_perc * max_dim)
    
    # Process each model
    for curr_model_number in model_numbers:
        print(f"Processing model {curr_model_number}...")
        
        # Load model
        model_file = path.join(model_path, f"model_{curr_model_number}.keras")
        try:
            model = load_model(model_file, 
                             custom_objects={
                                 'Custom_Hamming_Loss1': Custom_Hamming_Loss1,
                                 'AttentionWeights': AttentionWeights,
                                 'ContextVector': ContextVector
                             })
            
            # Optimize the model for inference
            model_inference = tf.function(
                model.predict, 
                input_signature=[tf.TensorSpec(shape=(None, adj_max_dim), dtype=tf.int32)]
            )
            print(f"Model {curr_model_number} loaded and optimized for inference")
        except Exception as e:
            print(f"Error loading model {curr_model_number}: {e}")
            continue

        # Create output directory for this model
        curr_target_dir = path.join(target_dir, f"model_{curr_model_number}")
        os.makedirs(curr_target_dir, exist_ok=True)
        
        # Get precision test results once, outside the loop
        precision_test = get_precision_test(model_path, curr_model_number)
        
        for i, plans in enumerate(test_plans):
            all_goals = get_goals(plans, dizionario_goal)
            filename = f"network_results_{filenames[i]}.txt"

            sel_filenames = os.listdir(problems_dir)
            sel_filenames = [
                f.rsplit(".", 1)[0]
                for f in sel_filenames
                if f.endswith(".pddl") and not f.startswith("domain")
            ]
            
            target_file = path.join(curr_target_dir, filename)
            
            for perc_action in [0.3, 0.5, 0.7]:
                gen = PlanGenerator(
                    plans,
                    dizionario,
                    dizionario_goal,
                    batch_size,  # Use larger batch size
                    adj_max_dim,
                    perc_action,
                    shuffle=False,
                )
                
                print(f"Processing {len(gen.plans)} plans with {perc_action} action percentage")
                
                # Process plans in batches for much better GPU utilization
                for j in range(0, len(gen.plans), batch_size):
                    batch_end = min(j + batch_size, len(gen.plans))
                    batch_plans = []
                    batch_indices = []
                    
                    # Gather plans for this batch
                    for k in range(j, batch_end):
                        plan_name = gen.plans[k].plan_name
                        plan_filename = plan_name.split("-", 2)[2].split(".", 1)[0]
                        
                        if plan_filename in sel_filenames:
                            batch_plans.append((k, plan_name))
                            batch_indices.append(k)
                    
                    if not batch_plans:
                        continue
                    
                    # Get batch input data
                    batch_x = []
                    for idx in batch_indices:
                        x, _ = gen.__getitem__(idx)
                        batch_x.append(x)
                    
                    # Convert to numpy array
                    batch_x = np.vstack(batch_x)
                    
                    # Time the batch prediction
                    start = time.time()
                    y_pred_batch = model.predict(batch_x, verbose=0, batch_size=len(batch_x))
                    elapsed_batch = time.time() - start
                    
                    print(f"Batch prediction time for {len(batch_x)} samples: {elapsed_batch:.4f}s " 
                          f"({elapsed_batch/len(batch_x):.4f}s per sample)")
                    
                    # Process individual results from the batch prediction
                    for idx, (plan_idx, plan_name) in enumerate(batch_plans):
                        y_pred = y_pred_batch[idx:idx+1]  # Keep the batch dimension
                        goal_number = get_goal_number(plan_name)
                        
                        if precision_test is not None:
                            precision_pred = np.multiply(
                                precision_test, [1 if y > threshold else 0 for y in y_pred[0]]
                            )
                        else:
                            precision_pred = y_pred[0]

                        scores = get_scores(y_pred[0], all_goals)
                        pred_scores = get_scores(precision_pred, all_goals, True)

                        if scores is not None:
                            out = get_result(scores, goal_number)
                        else:
                            out = False

                        res = [out, goal_number, scores, pred_scores]
                        
                        save_results_json(
                            target_file,
                            res,
                            path.basename(plan_name),
                            f"model_{curr_model_number}",
                            perc_action,
                            elapsed_batch / len(batch_plans),  # Approximate per-plan time
                            all_goals,
                        )

        print(f"Completed processing for model {curr_model_number}")


def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Run goal recognition predictions for TensorFlow models")
    
    parser.add_argument(
        "--model-path", "-m", 
        required=True,
        help="Folder containing the models",
        type=str
    )
    
    parser.add_argument(
        "--model-number", "-n",
        default="ALL",
        help="Number of the model to use (default: process all models)",
        type=str
    )
    
    parser.add_argument(
        "--read-dict-dir", "-d",
        required=True,
        help="Folder containing the dictionaries",
        type=str
    )
    
    parser.add_argument(
        "--read-test-plans-dir", "-s",
        required=True,
        help="Folder containing the testsets plans",
        type=str
    )
    
    parser.add_argument(
        "--target-dir", "-t",
        required=True,
        help="Folder where to save the results",
        type=str
    )
    
    parser.add_argument(
        "--max-plan-dim",
        default=100,
        help="Maximum number of actions in a plan (default: 100)",
        type=int
    )
    
    parser.add_argument(
        "--max-plan-perc",
        default=1.0,
        help="Maximum percentage of actions in a plan (default: 1.0)",
        type=float
    )
    
    parser.add_argument(
        "--problems-dir", "-p",
        required=True,
        help="Folder containing the problems",
        type=str
    )
    
    parser.add_argument(
        "--threshold", "-th",
        default=0.5,
        help="Threshold for rounding 1/0 (default: 0.5)",
        type=float
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    run(
        model_path=args.model_path,
        model_number=args.model_number,
        read_dict_dir=args.read_dict_dir,
        read_test_plans_dir=args.read_test_plans_dir,
        target_dir=args.target_dir,
        problems_dir=args.problems_dir,
        max_dim=args.max_plan_dim,
        max_plan_perc=args.max_plan_perc,
        threshold=args.threshold,
        batch_size=128, 
    )