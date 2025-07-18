import click
from plan_generator_torch import PlanGeneratorMultiPerc, PlanGenerator
import torch
import torch.nn as nn
from utils_torch import load_from_folder
import os
import json
import numpy as np
from plan import Plan
import time
from os import path


class AttentionWeights(nn.Module):
    def __init__(self, max_dim):
        super(AttentionWeights, self).__init__()
        self.max_dim = max_dim
        self.attention = nn.Linear(446, 1)

    def forward(self, lstm_output):
        attention_scores = self.attention(lstm_output)
        attention_weights = torch.softmax(attention_scores, dim=1)
        return attention_weights


class ContextVector(nn.Module):
    def __init__(self):
        super(ContextVector, self).__init__()

    def forward(self, lstm_output, attention_weights):
        context = torch.sum(lstm_output * attention_weights, dim=1)
        return context


class PlanModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        goal_size,
        max_dim,
        embedding_dim=85,
        lstm_hidden=446,
        dropout=0.30,
    ):
        super(PlanModel, self).__init__()
        self.max_dim = max_dim

        self.embedding = nn.Embedding(vocab_size + 1, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim, lstm_hidden, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.attention_weights = AttentionWeights(max_dim)
        self.context_vector = ContextVector()
        self.output_layer = nn.Linear(lstm_hidden, goal_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (batch_size, seq_len)
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)

        # Create mask for padded elements
        mask = (x != 0).float()  # (batch_size, seq_len)

        lstm_output, _ = self.lstm(embedded)  # (batch_size, seq_len, lstm_hidden)

        # Apply mask to LSTM output
        lstm_output = lstm_output * mask.unsqueeze(-1)
        
        lstm_output = self.dropout(lstm_output)
        
        attention_weights = self.attention_weights(
            lstm_output
        )  # (batch_size, seq_len, 1)
        context = self.context_vector(
            lstm_output, attention_weights
        )  # (batch_size, lstm_hidden)

        output = self.output_layer(context)  # (batch_size, goal_size)
        output = self.sigmoid(output)

        return output


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


@click.command()
@click.option(
    "--model-path",
    "-m",
    "model_path",
    type=click.Path(exists=True),
    help="Folder containing the models",
    required=True,
)
@click.option(
    "--model-number",
    "-n",
    "model_number",
    type=click.INT,
    help="Number of the model to use",
    required=True,
)
@click.option(
    "--read-dict-dir",
    "-d",
    "read_dict_dir",
    type=click.Path(exists=True),
    help="Folder containing the dictionaries",
    required=True,
)
@click.option(
    "--read-test-plans-dir",
    "-s",
    "read_test_plans_dir",
    type=click.Path(exists=True),
    help="Folder containing the testsets plans",
    required=True,
)
@click.option(
    "--target-dir",
    "target_dir",
    "-t",
    type=click.Path(exists=False),
    help="Folder where to save the results",
    required=True,
)
@click.option(
    "--max-plan-dim",
    "max_dim",
    type=click.INT,
    help="Maximum number of actions in a plan",
    default=100,
)
@click.option(
    "--max-plan-perc",
    "max_plan_perc",
    type=click.FLOAT,
    help="Maximum percentage of actions in a plan",
    default=1,
)
@click.option(
    "--problems-dir",
    "problems_dir",
    "-p",
    type=click.Path(exists=True),
    help="Folder containing the problems",
    required=True,
)
def run(
    model_path,
    model_number,
    read_dict_dir,
    read_test_plans_dir,
    target_dir,
    problems_dir,
    max_dim,
    max_plan_perc,
):
    """Main function to run goal recognition predictions"""

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dictionaries
    try:
        [dizionario, dizionario_goal] = load_from_folder(
            read_dict_dir, ["action_dict.pkl", "goal_dict.pkl"]
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
            max_dim=max_dim,  # Adjust based on your model
        )
        model.load_state_dict(torch.load(model_file, map_location=device))
        model.to(device)
        model.eval()
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Load test plans
    filenames = os.listdir(read_test_plans_dir)
    try:
        test_plans = load_from_folder(read_test_plans_dir, filenames)
        print(f"Test plans loaded: {len(test_plans)} files")
    except Exception as e:
        print(f"Error loading test plans: {e}")
        return

    
    
    max_dim = int(max_plan_perc * max_dim)
    
    target_dir = path.join(target_dir, f"model_{model_number}")
    os.makedirs(target_dir, exist_ok=True)
    
    for set in test_plans:
        print(len(set))
    for i, plans in enumerate(test_plans):
        all_goals = get_goals(plans, dizionario_goal)
        filename = f"network_results_{filenames[i]}.txt"

        sel_filenames = os.listdir(problems_dir)
        sel_filenames = [
            f.rsplit(".", 1)[0]
            for f in sel_filenames
            if f.endswith(".pddl") and not f.startswith("domain")
        ]
        
        target_file = path.join(target_dir, filename)
        
        for perc_action in [0.3, 0.5, 0.7]:
            gen = PlanGenerator(
                plans,
                dizionario,
                dizionario_goal,
                1,
                max_dim,
                perc_action,
                shuffle=False,
            )

            for j in range(gen.__len__()):
                plan_name = gen.plans[j].plan_name
                if plan_name.split("-", 2)[2].split(".", 1)[0] in sel_filenames:
                    goal_number = get_goal_number(plan_name)
                    x, y = gen.__getitem__(j)

                    # Convert to PyTorch tensor
                    x_tensor = torch.tensor(x, dtype=torch.long).to(device)

                    start = time.time()
                    with torch.no_grad():
                        y_pred = model(x_tensor)
                        y_pred_np = y_pred.cpu().numpy()

                    precision_test = get_precision_test(model_path, model_number)
                    if precision_test is not None:
                        precision_pred = np.multiply(
                            precision_test, [1 if y > 0.5 else 0 for y in y_pred_np[0]]
                        )
                    else:
                        precision_pred = y_pred_np[0]

                    scores = get_scores(y_pred_np[0], all_goals)
                    pred_scores = get_scores(precision_pred, all_goals, True)

                    if scores is not None:
                        out = get_result(scores, goal_number)
                    else:
                        out = False

                    elapsed = time.time() - start
                    res = [out, goal_number, scores, pred_scores]
                    
                    save_results_json(
                        target_file,
                        res,
                        path.basename(plan_name),
                        f"model_{model_number}",
                        perc_action,
                        elapsed,
                        all_goals,
                    )

                    # print(
                    #     f"Processed: {plan_name}, Result: {out}, Time: {elapsed:.4f}s"
                    # )


if __name__ == "__main__":
    run()
