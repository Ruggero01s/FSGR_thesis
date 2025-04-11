import click
from plan_generator import PlanGeneratorMultiPerc, PlanGenerator
from tensorflow.keras.models import load_model
from goal_rec_utils.attention_layers import AttentionWeights, ContextVector
from incremental_model_training import Custom_Hamming_Loss1
from utils_unibs.files import load_from_folder
import os
import json
import numpy as np
from goal_rec_utils.plan import Plan
import time
from os import path


def get_precision_test(model_path: str, model_number: int):
    [test_metrics] = load_from_folder(os.path.dirname(model_path), [f'metrics_{model_number}.txt'])
    start = False
    values = list()
    for line in test_metrics:
        if line.strip() == '':
            if start:
                break
            else:
                start = True
                continue
        if start:
            values.append(float(line.rsplit()[-4]))
    return values

def get_score(prediction: np.ndarray, possible_goal: list) -> float:
    '''
    Returns the score for a possible goal.
    
    Args:
        prediction:
            An array that contains the model prediction.
        
        possible_goal:
            A list that contains the possible goal indexes.
        
    Returns:
        An float that represents the score of the possible goal.
    '''
    
    score=0
    count=0
    for index in possible_goal:
        p = prediction[0][int(index)]
        if  p != 0:
            count += 1
        score += p
    return score, count


def get_max(scores: np.ndarray) -> list:
    '''
    Returns a list with the index (or indexes) of the highest scores.
    
    Args:
        scores:
            An array that contains the scores as floats.
    
    Returns:
        A list thet contains the indexes of the highest score.
    '''
    max_element = -1
    index_max = list()
    for i in range(len(scores)):
        if scores[i] > max_element:
            max_element = scores[i]
            index_max = [i]
        elif scores[i] == max_element:
            index_max.append(i)

    return index_max

def get_scores(prediction: np.ndarray, possible_goals: dict, normalize: bool = False) -> np.ndarray:
    '''
    Returns the scores for all possible goals.
    
    Args:
        prediction:
            An array that contains the model prediction.
        
        possible_goals:
            A list of possible goals; each possible goal is represented as a
            list
        
    Returns:
        An array that contains the score of each of the possible goals.
    '''
    try:
        scores = np.zeros((max(possible_goals)+1,), dtype=float)
        for index in possible_goals:
            scores[index], count = get_score(prediction, list(possible_goals[index]))
            if normalize and count > 0:
                scores[index] /= count
        return scores
    except IndexError:
        print(prediction)
        print(possible_goals)
        return None
    
    
def get_result(scores: np.ndarray, correct_goal: int) -> bool:
    '''
    Computes if the goal recognition task is successfull.
    
    Args:
        scores:
            An array of floats that contains a score for 
            each possible goal
        correct_goal: 
            An integer that represents the index of the 
            correct goal
            
    Returns:
        True if the maximum score index corresponds to the 
        correct goal index, False otherwise.
    '''
    idx_max_list = get_max(scores)
    if len(idx_max_list) == 1:
        idx_max = idx_max_list[0]
    else:
        print(f'Algorithm chose randomly one of {len(idx_max_list)} equals candidates.')
        idx_max = idx_max_list[np.random.randint(0, len(idx_max_list))]
    if idx_max == correct_goal:
        return True
    else:
        return False
    
def get_correct_goal_idx(correct_goal: list, possible_goals: list) -> int:
    '''
    Conputes the correct goal index.
    
    Args:
        correct_goal:
            A list of strings that contains the correct goal
            fluents.
        possible_goals:
            A list of possible goals; each possible goal is represented as a
            list.
    
    Returns:
        The index of the correct goal in the possible goals list.
        None if the possible goal list does not contain the correct goal.
    '''
    
    for index, possible_goal in enumerate(possible_goals):
        possible_goal = np.sort(possible_goal)
        correct_goal = np.sort(correct_goal)
        if np.all(possible_goal == correct_goal):
            return index
    return None


def save_results_json(target_file, results, filename, model_name, percentage, time, goals):
    scores = results[2]
    pred_scores = results[3]
    scores_dict = dict()
    pred_dict = dict()
    goals_dict = dict()
    if scores is not None:
        for i,s in enumerate(scores):
            scores_dict[i] = f'{s:.5f}'
    else:
        scores_dict = None
    
    if pred_scores is not None:
        for i,s in enumerate(pred_scores):
            pred_dict[i] = f'{s:.5f}'
    else:
        pred_dict = None
    
    if goals is not None:
        for k in goals:
            s =''.join(f'{str(e)} ' for e in goals[k])
            goals_dict[k] = s.strip()
    else:
        goals_dict = None
    json_dict = {
        'INSTANCE' : filename.split(',', 1)[0],
        'DOMAIN': filename + f'_constraint{percentage}',
        'MODEL': model_name,
        'PREDICTED': results[0],
        'ACTUAL' : results[1],
        'SCORES' : scores_dict,
        'PRED' : pred_dict,
        'TOTALRUNTIME': time,
        'GOALS' : goals_dict
    }
    
    with open(target_file, 'a') as af:
        af.writelines(f'{json.dumps(json_dict)}\n')


def get_goal(plan: Plan, goals_dict: dict) -> tuple:
    goals = plan.goals
    new_goal = list()
    for g in goals:
        goals_vector = goals_dict[g]
        index = np.argmax(goals_vector)
        new_goal.append(index)
    new_goal = np.sort(new_goal)
    return tuple(new_goal)


def get_goal_number(plan_name: str) -> int:
    goal_number = plan_name.split('hyp=hyp-')[1]
    #goal_number = int(goal_number.split('_', 1)[0])
    goal_number = int(goal_number.split('.', 1)[0])
    return goal_number

def get_goals(plans: list, dizionario_goal: dict) -> dict:
    goals_dict = dict()
    for plan in plans:
        plan_name = plan.plan_name
        goal_number = get_goal_number(plan_name)
        new_goal = get_goal(plan, dizionario_goal)
        goals_dict[goal_number] = new_goal
    return goals_dict




@click.command()
@click.option('--model-path', '-m', 'model_path', type=click.Path(exists=True), help='Folder containing the models', required=True)
@click.option('--model-number', '-n', 'model_number', type=click.INT, help='Number of the model to use', required=True)
@click.option('--read-dict-dir', '-d', 'read_dict_dir', type=click.Path(exists=True), help='Folder containing the dictionaries', required=True)
@click.option('--read-test-plans-dir', '-s', 'read_test_plans_dir', type=click.Path(exists=True), help='Folder containing the testsets plans', required=True)
@click.option('--target-dir', 'target_dir', '-t', type=click.Path(exists=False), help='Folder where to save the results', required=True)
@click.option('--max-plan-dim', 'max_dim', type=click.INT, help='Maximum number of actions in a plan', default=100)
@click.option('--max-plan-perc', 'max_plan_perc', type=click.FLOAT, help='Maximum percentage of actions in a plan', default=1)
@click.option('--problems-dir', 'problems_dir', '-p', type=click.Path(exists=True), help='Folder containing the problems', required=True)
def run(model_path, model_number, read_dict_dir, read_test_plans_dir, target_dir, problems_dir, max_dim, max_plan_perc):
    model_path = os.path.join(model_path, f'model_{model_number}')
    try:
        model = load_model(model_path,  custom_objects={'AttentionWeights': AttentionWeights,
                                                        'ContextVector': ContextVector,
                                                        'Custom_Hamming_Loss1': Custom_Hamming_Loss1})
        print('Model loaded')
        print(model.summary())
    except OSError as e:
        print(e)
        print('Error while loading the model.\n'
              'Please check the -r parameter is correct')
        model = None
    
    filenames = os.listdir(read_test_plans_dir)
    [dizionario, dizionario_goal] = load_from_folder(read_dict_dir, ['dizionario', 'dizionario_goal'])
    test_plans = load_from_folder(read_test_plans_dir, filenames)

    max_dim = int(max_plan_perc*max_dim)

    if model is None or test_plans is None or dizionario is None or dizionario_goal is None:
        print('Could not create the file')
    else:
        os.makedirs(target_dir, exist_ok=True)
        for i, plans in enumerate(test_plans):

            all_goals = get_goals(plans, dizionario_goal)

            filename = f'network_results_{filenames[i]}.txt'

            sel_filenames = os.listdir(problems_dir)

            sel_filenames = [f.rsplit('.', 1)[0] for f in sel_filenames if f.endswith('.pddl') and not(f.startswith('domain'))]

            target_file = path.join(target_dir, filename)
            for perc_action in [0.3, 0.5, 0.7]:#[0.1, 0.3, 0.5, 0.7, 1]:
                gen = PlanGenerator(plans, dizionario, dizionario_goal, 1, max_dim, perc_action, shuffle=False)
                for i in range(gen.__len__()):
                    plan_name = gen.plans[i].plan_name
                    #print(plan_name, plan_name.split('-',2)[2].split('.',1)[0], sel_filenames[0])
                    if plan_name.split('-',2)[2].split('.',1)[0] in sel_filenames:
                        goal_number = get_goal_number(plan_name)
                        x, y = gen.__getitem__(i)
                        start = time.time()
                        y_pred = model.predict(x)
                        precision_test = get_precision_test(model_path, model_number)
                        precision_pred = [np.multiply(precision_test, [1 if y > 0.5 else 0 for y in y_pred[0]])]
                        scores = get_scores(y_pred, all_goals)
                        pred_scores = get_scores(precision_pred, all_goals, True)
                        if scores is not None:
                            out = get_result(scores, goal_number)
                        else:
                            out = False
                        elapsed = time.time() - start
                        res = [out, goal_number, scores, pred_scores]
                        
                        save_results_json(target_file, res, path.basename(plan_name), path.basename(model_path), perc_action, elapsed, all_goals)
                                      



if __name__ == '__main__':
    run()