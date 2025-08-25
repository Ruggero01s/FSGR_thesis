# filepath: /home/deeplearning/ruggero/FSGR_thesis/code/gr_results.py

import pandas as pd
import json
import os
from os import path
import numpy as np
import matplotlib.pyplot as plt
from os import makedirs
import datetime
import glob

# Try to import specific utilities if available, otherwise define fallbacks
try:
    from utils_torch import load_from_folder, save_file
    from utils_unibs.plots import get_line_plot, set_size
    from utils_unibs.constants import C
    from utils_unibs.plots import plt
except ImportError:
    print("Warning: Some utility modules not found, using basic implementations")
    
    # Define basic fallbacks for missing utilities
    def load_from_folder(folder_path, file_list, type=None):
        results = []
        for file in file_list:
            file_path = os.path.join(folder_path, file)
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    if type == 'TXT':
                        results.append(f.readlines())
                    else:
                        results.append(f.read())
        return results
    
    # Define a basic plotting function
    def get_line_plot(data_list, x_ticks_pos, x_ticks_labels, y_ticks_pos, y_ticks_labels,
                     title, xlabel, ylabel, styles=None, markers=None):
        fig, ax = plt.subplots(figsize=(10, 6))
        for i, data in enumerate(data_list):
            style = '-' if styles is None else styles[i]
            marker = '' if markers is None else markers[i]
            ax.plot(range(len(data)), data, style, marker=marker)
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title)
        
        if x_ticks_pos:
            ax.set_xticks(x_ticks_pos)
            if x_ticks_labels:
                ax.set_xticklabels(x_ticks_labels)
        
        if y_ticks_pos:
            ax.set_yticks(y_ticks_pos)
            if y_ticks_labels:
                ax.set_yticklabels(y_ticks_labels)
        
        return fig, ax
    
    # Define a placeholder for missing constants
    class C:
        TXT = 'TXT'

    # Set size function placeholder
    def set_size(name, size):
        pass

# Create timestamp for output files
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

###########################################
# Cell: General methods
###########################################
def load_results(target_folder: str): 
    files = np.sort([f for f in os.listdir(target_folder) if f.endswith('.txt')])
    results = load_from_folder(target_folder, files, type=C.TXT)
    return files, results

###########################################
# Cell: Planner methods
###########################################
def create_df_dict_planner(results_folder: str, verbose: int = 0):
    planner_folder = path.join("datasets/gr_logistics/results/results_jsons/planner")
    if not os.path.exists(planner_folder):
        print(f"Warning: Planner folder not found at {planner_folder}")
        return {}
        
    planner_files, planner_results = load_results(planner_folder)
    df_planner = dict()
    
    for i, file in enumerate(planner_files):
        df = None
        for l in planner_results[i]:
            json_line = json.loads(l)
            if df is None:
                for l1 in planner_results[i]:
                    json_line1 = json.loads(l1)
                    if 'FD_PLAN_STEPS' in json_line1:
                        columns = list(json_line1.keys())
                    break
                df = pd.DataFrame(columns=columns)

            df.loc[df.shape[0]] = json_line

        df_planner[file.split('.',1)[0]] = df
        if verbose > 0:
            print(f"Loaded planner file: {file}")
            print(df_planner[file.split('.',1)[0]].head())
            
    return df_planner

###########################################
# Cell: Network methods
###########################################
def get_sorted_keys(line: pd.Series, goals_number: int):
    scores_dict = {i : float(line[f'{i}']) for i in range(goals_number+1) if f'{i}' in line.index}
    sorted_keys = sorted(scores_dict, key=scores_dict.get, reverse=True)
    return sorted_keys

def create_df_network(results_file: str):
    df = None
    df_scores = None
    df_goals = None
    df_prec = None
    
    for l in results_file:
        try:
            json_line = json.loads(l)
            scores = json_line.get('SCORES', {})
            goals = json_line.get('GOALS', {})
            prec_scores = json_line.get('PRED', {})
            
            if 'SCORES' in json_line:
                json_line.pop('SCORES')
            if 'GOALS' in json_line:
                json_line.pop('GOALS')
            if 'PRED' in json_line:
                json_line.pop('PRED')
                
            if df is None:
                df = pd.DataFrame(json_line, index=[0])
            else:
                df.loc[df.shape[0]] = json_line
            
            if df_scores is None:
                df_scores = pd.DataFrame(scores, index=[0])
            else:
                df_scores.loc[df_scores.shape[0]] = scores
                
            if df_goals is None:
                df_goals = pd.DataFrame(goals, index=[0])
            else:
                df_goals.loc[df_goals.shape[0]] = goals
                
            if df_prec is None:
                df_prec = pd.DataFrame(prec_scores, index=[0])
            else:
                df_prec.loc[df_prec.shape[0]] = prec_scores
                
        except json.JSONDecodeError:
            print(f"Warning: Could not parse JSON from line: {l[:100]}...")
            continue

    if df is None:
        print("Warning: No valid data found in results file")
        return None, []
        
    if df_goals is not None and not df_goals.empty:
        df_goals.columns = [f'GOAL_{c}' for c in df_goals.columns]
        df = df.join(df_goals)
        
    if df_prec is not None and not df_prec.empty:
        df_prec.columns = [f'PREC_{c}' for c in df_prec.columns]
        df = df.join(df_prec)
        
    if df_scores is not None and not df_scores.empty:
        df = df.join(df_scores)
        goals_list = df_scores.columns
    else:
        goals_list = []
        
    return df, goals_list

def create_df_dict_network(model_number: int, results_folder: str, verbose: int = 0):
    model_folder = os.path.join(results_folder, f'model_{model_number}')
    if not os.path.exists(model_folder):
        print(f"Warning: Model folder not found at {model_folder}")
        return {}
        
    try:
        network_files, network_results = load_results(model_folder)
    except FileNotFoundError:
        print(f"Error: Could not load results from {model_folder}")
        return {}
        
    df_network = dict()
    
    for i, file in enumerate(network_files):
        df, goals_list = create_df_network(network_results[i])
        if df is not None:
            df_network[file.split('.',1)[0]] = df
            if verbose > 0:
                c = [f'GOAL_{c}' for c in goals_list if f'GOAL_{c}' in df_network[file.split('.',1)[0]].columns]
                c.extend(goals_list)
                c.extend(['PREDICTED'])
                print(f"Loaded network file: {file}")
                print(df_network[file.split('.',1)[0]][c].head())
                
    return df_network

###########################################
# Cell: MC system methods
###########################################
def get_planner_metric(df_planner: dict, df_network: dict, unresolved_instances: dict, uncertain_instances: dict, threshold: float, verbose: int = 0):
    tot = 0
    all_metrics = np.zeros((2,))
    
    for k in unresolved_instances:
        metrics = np.zeros((2,))
        p = k.rsplit('_', 1)[1]
        count = 0

        for domain in uncertain_instances.get(k, []):
            try:
                df_problem_planner_final = None
                df_optimal_planner_final = None
                problem_name = domain.split(',', 1)[0].rsplit('-', 1)[1]
                constraint = domain.rsplit('constraint', 1)[1]
                line = df_network[k][df_network[k]['DOMAIN'] == domain].iloc[0]
                actual_goal = df_network[k][df_network[k]['DOMAIN'] == domain]['ACTUAL'].values[0]
                goals_number = int(df_network[k].columns[-1])
                sorted_keys = get_sorted_keys(line, goals_number)
                goals_to_keep = []
                
                for i, sorted_key in enumerate(sorted_keys):
                    if float(line.get(f'{sorted_keys[0]}', 0)) - float(line.get(f'{sorted_keys[i]}', 0)) <= threshold:
                        goals_to_keep.append(sorted_key)
                        
                if f'results_{p}' not in df_planner:
                    continue
                    
                df_problem_planner = df_planner[f'results_{p}']
                df_problem_planner = df_problem_planner[df_problem_planner['INSTANCE'] == problem_name]
                df_problem_planner = df_problem_planner[df_problem_planner['DOMAIN'].str.contains(
                    f"constraint{int(float(constraint)*100) if float(constraint) <= 1 else int(constraint)}"
                )]

                df_optimal_planner = df_planner[f'results_{p}']
                df_optimal_planner = df_optimal_planner[df_optimal_planner['INSTANCE'] == problem_name]
                df_optimal_planner = df_optimal_planner[(df_optimal_planner['SYSTEM'] == "ottimo") | 
                                                     (df_optimal_planner['SYSTEM'] == "fast-downward_HMAX")]
                
                for goal_to_keep in goals_to_keep:
                    if df_problem_planner_final is None:
                        df_problem_planner_final = df_problem_planner[df_problem_planner['DOMAIN'].str.endswith(f"goal_{goal_to_keep}")]
                        df_optimal_planner_final = df_optimal_planner[df_optimal_planner['DOMAIN'].str.endswith(f"goal_{goal_to_keep}")]
                    else:
                        df_problem_planner_final = pd.concat([
                            df_problem_planner_final, 
                            df_problem_planner[df_problem_planner['DOMAIN'].str.endswith(f"goal_{goal_to_keep}")]
                        ])
                        df_optimal_planner_final = pd.concat([
                            df_optimal_planner_final,
                            df_optimal_planner[df_optimal_planner['DOMAIN'].str.endswith(f"goal_{goal_to_keep}")]
                        ])
                        
                if df_problem_planner_final is None or df_optimal_planner_final is None:
                    continue
                    
                df_problem_planner = df_problem_planner_final.sort_values(by=['DOMAIN'])
                df_optimal_planner = df_optimal_planner_final.sort_values(by=['DOMAIN'])
                time = np.sum(df_problem_planner['TOTALRUNTIME']) + np.sum(df_optimal_planner['TOTALRUNTIME'])
                
                if k == 'network_results_test_plans_p05':
                    if actual_goal > 6:
                        actual_goal -= 2
                    elif actual_goal > 2:
                        actual_goal -= 1

                try:
                    min_value_idx = df_problem_planner['FD_PLAN_STEPS'].reset_index(drop=True).sub(
                        df_optimal_planner['FD_PLAN_STEPS'].reset_index(drop=True)
                    )
                    min_value_idx.index = df_problem_planner.index
                    min_value_idx = min_value_idx[min_value_idx == 0]
                    
                    if min_value_idx.shape[0] > 0 and not np.isnan(min_value_idx.idxmin()):
                        min_value_idx = min_value_idx.index
                        pred_goals = []
                        for ind in min_value_idx:
                            pred_goal = df_problem_planner.loc[ind]['DOMAIN'].rsplit('_',1)[1]
                            pred_goals.append(int(pred_goal))
                        if int(actual_goal) in pred_goals:
                            metrics[0] += 1/len(pred_goals)
                    else:
                        goals_number = int(list(df_network[k].columns)[-1])
                        pred_goal = [np.random.randint(0, goals_number)]
                        
                    metrics[1] += time
                    count += 1
                except Exception as e:
                    print(f"Error processing domain {domain}: {e}")
                    continue
            except Exception as e:
                print(f"Error processing domain {domain}: {e}")
                continue

        for domain in unresolved_instances.get(k, []):
            try:
                problem_name = domain.split(',', 1)[0].rsplit('-', 1)[1]
                constraint = domain.rsplit('constraint', 1)[1]
                
                if f'results_{p}' not in df_planner:
                    continue
                    
                df_problem_planner = df_planner[f'results_{p}']
                df_problem_planner = df_problem_planner[df_problem_planner['INSTANCE'] == problem_name]
                df_problem_planner = df_problem_planner[df_problem_planner['DOMAIN'].str.contains(
                    f"constraint{int(float(constraint)*100) if float(constraint) <= 1 else int(constraint)}"
                )]
                df_problem_planner = df_problem_planner.sort_values(by=['DOMAIN'])
                
                df_optimal_planner = df_planner[f'results_{p}']
                df_optimal_planner = df_optimal_planner[df_optimal_planner['INSTANCE'] == problem_name]
                df_optimal_planner = df_optimal_planner[(df_optimal_planner['SYSTEM'] == "ottimo") | 
                                                     (df_optimal_planner['SYSTEM'] == "fast-downward_HMAX")]
                df_optimal_planner = df_optimal_planner.sort_values(by=['DOMAIN'])
                
                time = np.sum(df_problem_planner['TOTALRUNTIME']) + np.sum(df_optimal_planner['TOTALRUNTIME'])
                actual_goal = df_network[k][df_network[k]['DOMAIN'] == domain]['ACTUAL'].values[0]

                if k == 'network_results_test_plans_p05':
                    if actual_goal > 6:
                        actual_goal -= 2
                    elif actual_goal > 2:
                        actual_goal -= 1

                try:
                    min_value_idx = df_problem_planner['FD_PLAN_STEPS'].reset_index(drop=True).sub(
                        df_optimal_planner['FD_PLAN_STEPS'].reset_index(drop=True)
                    )
                    min_value_idx.index = df_problem_planner.index
                    min_value_idx = min_value_idx[min_value_idx == 0]
                    
                    if min_value_idx.shape[0] > 0 and not np.isnan(min_value_idx.idxmin()):
                        min_value_idx = min_value_idx.index
                        pred_goals = []
                        for ind in min_value_idx:
                            pred_goal = df_problem_planner.loc[ind]['DOMAIN'].rsplit('_',1)[1]
                            pred_goals.append(int(pred_goal))
                        if int(actual_goal) in pred_goals:
                            metrics[0] += 1/len(pred_goals)
                    else:
                        goals_number = int(list(df_network[k].columns)[-1])
                        pred_goal = [np.random.randint(0, goals_number)]
                        
                    metrics[1] += time
                    count += 1
                except Exception as e:
                    print(f"Error processing domain {domain}: {e}")
                    continue
            except Exception as e:
                print(f"Error processing domain {domain}: {e}")
                continue
                
        if verbose > 0 and count > 0:
            print(f'{k}: Precision: {metrics[0]/count*100 : .2f}, Time: {metrics[1]/count : .2f}')
            print('=========================')
            
        all_metrics += metrics
        tot += count
        
    return all_metrics, tot

def get_confidence(line: pd.Series, goals_number: int, confidence_threshold: float, sorted_keys: list):
    if len(sorted_keys) < 2:
        return False
    is_confident = float(line.get(f'{sorted_keys[0]}', 0)) - float(line.get(f'{sorted_keys[1]}', 0)) > confidence_threshold
    return is_confident

def get_risk_free(line: pd.Series, goals_number: int, uniqueness_threshold: float, sorted_keys: list):
    goals_dict = {i : line.get(f'GOAL_{i}', '').split(' ') if f'GOAL_{i}' in line.index else [] for i in range(goals_number+1)}
    uniq_dict = {}
    
    for k in goals_dict.keys():
        for el in goals_dict[k]:
            if el:  # Skip empty elements
                if el in uniq_dict:
                    uniq_dict[el] += 1
                else:
                    uniq_dict[el] = 1
                    
    if not uniq_dict:
        return False
        
    uniq_dict = {k: 1/uniq_dict[k] for k in uniq_dict.keys()}
    
    if sorted_keys and sorted_keys[0] in goals_dict:
        goal_sum = sum([uniq_dict.get(k, 0) for k in goals_dict[sorted_keys[0]] if k])
        total = sum([1 for k, g in goals_dict.items() if g])
        
        if total > 0:
            is_risk_free = goal_sum / total > uniqueness_threshold
            return is_risk_free
            
    return False

def get_precision(line: pd.Series, goals_number: int, precision_threshold: float, sorted_keys: list):
    if not sorted_keys:
        return False
        
    prec_dict = {i : float(line.get(f'PREC_{i}', 0)) for i in range(goals_number+1) if f'PREC_{i}' in line.index}
    
    if sorted_keys[0] in prec_dict:
        is_precise = prec_dict[sorted_keys[0]] >= precision_threshold
        return is_precise
        
    return False

def get_counts(metrics, predicted, time):
    out = []
    # Add the value for all the metrics in and
    metrics.append(True)
    
    for m in metrics:
        counts = np.zeros((3,), dtype=float)
        if m:
            counts[0] = 1
            counts[2] = time
            if predicted == True:
                counts[1] = 1
        else:
            metrics[-1] = False
        
        out.append(counts)
        
    return out

def get_metric(df_dictionary: dict, thresholds: list, verbose: int = 0, metrics_labels = ['Confidence', 'Risk Free', 'Precise', 'Confidence AND Risk Free AND Precise']):
    all_metrics = np.zeros((len(metrics_labels), 3))
    tot = 0
    unsolved_instances = {}
    uncertain_instances = {}
    
    for key in df_dictionary.keys():
        df_unresolved_instances = []
        df_uncertain_instances = []
        metrics = np.zeros((len(metrics_labels), 3))
        df = df_dictionary[key]
        indexes = list(range(0, df.shape[0]))
        np.random.seed(420)
        np.random.shuffle(indexes)
        
        for l in range(df.shape[0]):
            line = df.iloc[indexes[l]]
            
            try:
                # Find the highest numbered column to determine goals_number
                numeric_cols = [col for col in df.columns if col.isdigit()]
                if numeric_cols:
                    goals_number = int(max(numeric_cols))
                else:
                    goals_number = 0
                    
                sorted_keys = get_sorted_keys(line, goals_number)
                res = []
                
                for i, m in enumerate(metrics_labels):
                    if m == 'Confidence':
                        is_confident = get_confidence(line, goals_number, thresholds[i], sorted_keys)
                        res.append(is_confident)
                    if m == 'Risk Free':
                        is_risk_free = get_risk_free(line, goals_number, thresholds[i], sorted_keys)
                        res.append(is_risk_free)
                    if m == 'Precise':
                        is_precise = get_precision(line, goals_number, thresholds[i], sorted_keys)
                        res.append(is_precise)
                        
                predicted = line.get('PREDICTED', False)
                time_value = float(line.get('TOTALRUNTIME', 0))
                counts = get_counts(res, predicted, time_value)
                
                is_sample_uncertain = False
                if 'Confidence' in metrics_labels and 'Precise' in metrics_labels:
                    conf_index = metrics_labels.index('Confidence')
                    prec_index = metrics_labels.index('Precise')
                    
                    if counts[conf_index][0] == 0 and counts[prec_index][0] == 1:
                        is_sample_uncertain = True
                        df_uncertain_instances.append(line['DOMAIN'])
                        
                if counts[-1][0] == 0 and not is_sample_uncertain:
                    df_unresolved_instances.append(line['DOMAIN'])
                    
                metrics += counts
            except Exception as e:
                print(f"Error processing line in {key}: {e}")
                continue
                
        unsolved_instances[key] = df_unresolved_instances
        uncertain_instances[key] = df_uncertain_instances
            
        if verbose > 0:
            print(f'{key}:')
            for i, m in enumerate(metrics):
                if m[0] > 0:
                    print(f'{metrics_labels[i]}: Plan percentage: {m[0]/(l+1)*100 :.2f}, Precision: {m[1]/m[0]*100 :.2f}')
                else:
                    print(f'{metrics_labels[i]}: Plan percentage: 0.00, Precision: 0.00')
            print('=========================')
            
        all_metrics += metrics
        tot += l+1
        
    return all_metrics, tot, unsolved_instances, uncertain_instances

def get_metrics_in_range(df_dict: dict, target_metric: int, range_vals: list, verbose: int = 0, metrics_labels = ['Confidence', 'Risk Free', 'Precise', 'Confidence AND Risk Free AND Precise']):
    x, y_metric, y_pred_metric = range_vals, [], []
    thresholds = np.zeros((len(metrics_labels)-1,))
    
    for target_threshold in range_vals:
        thresholds[target_metric] = target_threshold
        all_metrics, tot, _, _ = get_metric(df_dict, thresholds, verbose, metrics_labels)
        
        if all_metrics[target_metric][0] > 0:
            y_metric.append(all_metrics[target_metric][0]/tot*100)
            y_pred_metric.append(all_metrics[target_metric][1]/all_metrics[target_metric][0]*100)
        else:
            y_metric.append(0)
            y_pred_metric.append(0)
            
    return x, y_metric, y_pred_metric

def get_metrics_for_models(models_list: list, thresholds: list, results_folder: str, use_planner: bool = True, 
                        verbose: int = 0, metrics_labels = ['Confidence', 'Risk Free', 'Precise', 'Confidence AND Risk Free AND Precise']):
    x = models_list
    network_plan_percentages = [[] for _ in range(len(metrics_labels))]
    planner_plan_percentages = []
    network_precisions = [[] for _ in range(len(metrics_labels))]
    only_network_precisions = []
    planner_precisions = []
    network_times = [[] for _ in range(len(metrics_labels))]
    only_network_times = []
    planner_times = []

    if use_planner:
        df_planner_dict = create_df_dict_planner(results_folder, verbose=verbose)
    else:
        df_planner_dict = {}

    for model_number in models_list:
        print(f'Processing Model {model_number}')
        
        if verbose > 0:
            print('Network statistics:')
            
        df_network_dict = create_df_dict_network(model_number, results_folder=results_folder, verbose=verbose)
        
        if not df_network_dict:
            print(f"Skipping model {model_number} - no data found")
            continue
            
        # Get metrics for network only
        network_only_metrics, network_only_instances, _, _ = get_metric(
            df_network_dict, [0], verbose, ['Confidence', 'C1']
        )
        
        if network_only_metrics[0][0] > 0:
            only_network_precisions.append(network_only_metrics[0][1]/network_only_metrics[0][0]*100)
            only_network_times.append(network_only_metrics[0][2]/network_only_metrics[0][0])
        else:
            only_network_precisions.append(np.nan)
            only_network_times.append(np.nan)
        
        # Get metrics with thresholds
        network_metrics, network_instances, unresolved_instances, uncertain_instances = get_metric(
            df_network_dict, thresholds, verbose, metrics_labels
        )
        
        for i, m in enumerate(metrics_labels):
            if network_metrics[i][0] > 0:
                plan_percentage = network_metrics[i][0]/network_instances*100
                precision = network_metrics[i][1]/network_metrics[i][0]*100
                time = network_metrics[i][2]/network_metrics[i][0]
            else:
                plan_percentage = 0
                precision = np.nan
                time = np.nan
                
            network_plan_percentages[i].append(plan_percentage)
            network_precisions[i].append(precision)
            network_times[i].append(time)
            
            if verbose > 0:
                print(f'{m}: Plan percentage: {plan_percentage:.2f}, Precision: {precision:.2f}')     
                print('=========================')
                
        if use_planner and df_planner_dict:
            try:
                conf_index = metrics_labels.index('Confidence')
                planner_metrics, planner_instances = get_planner_metric(
                    df_planner_dict, df_network_dict, unresolved_instances, 
                    uncertain_instances, thresholds[conf_index], verbose=verbose
                )
                
                if planner_instances > 0:
                    plan_percentage = 100 - network_plan_percentages[-1][-1]
                    precision = planner_metrics[0]/planner_instances*100
                    time = planner_metrics[1]/planner_instances
                else:
                    plan_percentage = 100 - network_plan_percentages[-1][-1]
                    precision = np.nan
                    time = np.nan
                    
                planner_plan_percentages.append(plan_percentage)
                planner_precisions.append(precision)
                planner_times.append(time)
            except Exception as e:
                print(f"Error in planner metrics for model {model_number}: {e}")
                planner_plan_percentages.append(100 - network_plan_percentages[-1][-1])
                planner_precisions.append(np.nan)
                planner_times.append(np.nan)
        else:
            planner_plan_percentages.append(100 - network_plan_percentages[-1][-1])
            planner_precisions.append(np.nan)
            planner_times.append(np.nan)
        
    return x, network_plan_percentages, network_precisions, network_times, planner_plan_percentages, planner_precisions, planner_times, only_network_precisions, only_network_times

def process_adaptive_model(model_dir, base_results_dir, plots_dir):
    """Process a single adaptive model directory"""
    model_name = os.path.basename(model_dir)
    print(f"\n{'='*50}")
    print(f"Processing model: {model_name}")
    print(f"{'='*50}")
    
    # Create output directories
    model_plots_dir = os.path.join(plots_dir, model_name)
    os.makedirs(model_plots_dir, exist_ok=True)
    
    # Set up paths
    results_folder = os.path.join(base_results_dir, model_name)
    
    # Define variables for this model
    target_model = find_highest_model_number(model_dir)
    print(f"Highest model number found: {target_model}")
    
    # Extract thresholds from model name instead of computing them
    # Assuming format like ada_mixp_004conf_08exp
    confidence_threshold = 0.04  # Default if not found
    precision_threshold = 0.8    # Default if not found
    
    # Parse model name to extract thresholds
    name_parts = model_name.split('_')
    for part in name_parts:
        if 'conf' in part:
            try:
                # Extract number before 'conf' and divide by 100
                conf_value = part.split('conf')[0]
                confidence_threshold = float(conf_value) / 100
                print(f"Extracted confidence threshold from name: {confidence_threshold}")
            except (ValueError, IndexError):
                print(f"Could not extract confidence threshold from {part}")
        elif 'exp' in part:
            try:
                # Extract number before 'exp' and divide by 10
                exp_value = part.split('exp')[0]
                precision_threshold = float(exp_value) / 10
                print(f"Extracted precision threshold from name: {precision_threshold}")
            except (ValueError, IndexError):
                print(f"Could not extract precision threshold from {part}")
    
    # Load network data for the latest model
    df_network = create_df_dict_network(target_model, results_folder=results_folder, verbose=0)
    if not df_network:
        print(f"No data found for model {target_model}, skipping...")
        return None
        
    # compute and plot the metrics across threshold ranges, 
    r_confidence = np.arange(0.00, 1, 0.01)
    r_precision = np.arange(0.00, 1, 0.01)
    r_risk = np.arange(0.00, 1, 0.01)
    
    x, y_confidence, y_pred_confidence = get_metrics_in_range(df_network, 0, r_confidence, 0, ['Confidence', 'Confidence2'])
    x, y_risk_free, y_pred_risk_free = get_metrics_in_range(df_network, 0, r_risk, 0, ['Risk Free', 'Risk Free2'])
    x, y_precise, y_pred_precise = get_metrics_in_range(df_network, 0, r_precision, 0, ['Precise', 'Precise2'])
    
    # Plot the metrics
    # Plot confidence
    fig, ax = get_line_plot([y_confidence, y_pred_confidence], 
                  range(0, len(x)+2, 20), [f'{i:.1f}' for i in np.arange(0, 1.1, 0.2)],
                  range(0, 101, 20), [], 
                  ['Plan Percentage', 'Prediction Accuracy'], 'Threshold Value', 'Percentage', 
                  styles=['-', '-'], markers=['', ''])
    fig.savefig(os.path.join(model_plots_dir, 'confidence_metrics.png'))
    plt.close(fig)
    
    # Plot risk free
    fig, ax = get_line_plot([y_risk_free, y_pred_risk_free], 
                  range(0, len(x)+2, 20), [f'{i:.1f}' for i in np.arange(0, 1.1, 0.2)],
                  range(0, 101, 20), [], 
                  ['Plan Percentage', 'Prediction Accuracy'], 'Threshold Value', 'Percentage', 
                  styles=['-', '-'], markers=['', ''])
    fig.savefig(os.path.join(model_plots_dir, 'risk_free_metrics.png'))
    plt.close(fig)

    # Plot experience/precision
    fig, ax = get_line_plot([y_precise, y_pred_precise], 
                  range(0, len(x)+2, 20), [f'{i:.1f}' for i in np.arange(0, 1.1, 0.2)],
                  range(0, 101, 20), [], 
                  ['Plan Percentage', 'Prediction Accuracy'], 'Threshold Value', 'Percentage', 
                  styles=['-', '-'], markers=['', ''])
    fig.savefig(os.path.join(model_plots_dir, 'precision_metrics.png'))
    plt.close(fig)
    
    # Use the extracted thresholds for model evaluation
    print(f"Using thresholds from model name - confidence: {confidence_threshold}, precision: {precision_threshold}")
    thresholds = [confidence_threshold, precision_threshold]
    models_list = list(range(target_model + 1))
    
    # Calculate metrics for all models
    try:
        x, network_plan_percentages, network_precisions, network_times, planner_plan_percentages, planner_precisions, planner_times, only_network_precisions, only_network_times = get_metrics_for_models(
            models_list, 
            thresholds, 
            results_folder=results_folder,
            verbose=0, 
            use_planner=True,
            metrics_labels=['Confidence', 'Precise', 'Confidence AND Precise']
        )
        
        # Calculate overall precision (combined network + planner)
        overall_precisions = []
        for i in range(len(planner_precisions)):
            if np.isnan(network_precisions[2][i]):
                overall_precisions.append(planner_precisions[i])
            else:
                network_contrib = network_precisions[2][i] * network_plan_percentages[2][i] / 100
                planner_contrib = planner_precisions[i] * planner_plan_percentages[i] / 100
                overall_precisions.append(network_contrib + planner_contrib)
                
        # Plot accuracy comparison (S1 vs S2 vs FSGR)
        fig, ax = get_line_plot(
            [only_network_precisions, planner_precisions, overall_precisions],  # Use actual planner values
            np.linspace(0, len(models_list)-1, 5), [f'{int(np.floor((i+1)*16*64/1000))}k' for i in np.linspace(0, len(models_list)-1, 5, dtype=int)],
            range(0, 101, 20), [], 
            None, '# of Train Plans', 'Accuracy', 
            styles=['-', '-', '-'], markers=['o', '*', '^']
        )
        ax.legend(['S1', 'S2', 'FSGR+'], loc='best')
        fig.savefig(os.path.join(model_plots_dir, 'accuracy_comparison.png'))
        plt.close(fig)
        
        # Plot system usage
        fig, ax = get_line_plot(
            [network_plan_percentages[2], planner_plan_percentages], 
            np.linspace(0, len(models_list)-1, 5), [f'{int(np.floor((i+1)*16*64/1000))}k' for i in np.linspace(0, len(models_list)-1, 5, dtype=int)],
            range(0, 101, 20), [], 
            None, '# of Train Plans', 'Use (%)', 
            styles=['-', '-'], markers=['o', '*']
        )
        ax.legend(['S1', 'S2'], loc='best')
        fig.savefig(os.path.join(model_plots_dir, 'system_usage.png'))
        plt.close(fig)
        
        # Plot time comparison
        overall_times = []
        for i in range(len(planner_times)):
            if np.isnan(network_times[2][i]):
                overall_times.append(planner_times[i])
            else:
                network_contrib = network_times[2][i] * network_plan_percentages[2][i] / 100
                planner_contrib = planner_times[i] * planner_plan_percentages[i] / 100
                overall_times.append(network_contrib + planner_contrib)
                
        fig, ax = get_line_plot(
            [only_network_times, planner_times, overall_times],  # Use actual planner values
            np.linspace(0, len(models_list)-1, 5), [f'{int(np.floor((i+1)*16*64/1000))}k' for i in np.linspace(0, len(models_list)-1, 5, dtype=int)],
            np.linspace(0, max(filter(lambda x: not np.isnan(x), planner_times + overall_times)), 5), [], 
            None, '# of Train Plans', 'Avg Time (s)',
            styles=['-', '-', '-'], markers=['o', '*', '^']
        )
        ax.legend(['S1', 'S2', 'FSGR+'], loc='best')
        fig.savefig(os.path.join(model_plots_dir, 'time_comparison.png'))
        plt.close(fig)
        
        # Save metrics to file
        results = {
            'model_name': model_name,
            'thresholds': {
                'confidence': confidence_threshold,
                'precision': precision_threshold
            },
            'metrics': {
                'only_network_precisions': only_network_precisions,
                'planner_precisions': planner_precisions,
                'overall_precisions': overall_precisions,
                'network_plan_percentages': network_plan_percentages,
                'planner_plan_percentages': planner_plan_percentages,
                'only_network_times': only_network_times,
                'planner_times': planner_times,
                'overall_times': overall_times
            }
        }
        
        with open(os.path.join(model_plots_dir, 'metrics.json'), 'w') as f:
            json.dump(results, f, indent=2)
            
        return results
        
    except Exception as e:
        print(f"Error processing models for {model_name}: {e}")
        return None

def find_highest_model_number(model_dir):
    """Find the highest model number in a given directory"""
    model_files = glob.glob(os.path.join(model_dir, "model_*.pth"))
    model_numbers = []
    
    for file in model_files:
        try:
            num = int(os.path.basename(file).replace("model_", "").replace(".pth", ""))
            model_numbers.append(num)
        except ValueError:
            continue
            
    return max(model_numbers) if model_numbers else 0

def find_ada_mixp_folders(base_dir):
    """Find all folders containing 'ada_mixp' in their name"""
    ada_mixp_folders = []
    
    for root, dirs, files in os.walk(base_dir):
        for dir_name in dirs:
            if "ada_mixp" in dir_name:
                full_path = os.path.join(root, dir_name)
                ada_mixp_folders.append(full_path)
                
    return ada_mixp_folders

def main():
    # Base directories
    base_dir = "datasets/gr_logistics/results/adaptive_incremental"
    base_results_dir = "datasets/gr_logistics/results/results_jsons"
    plots_dir = "datasets/gr_logistics/results/analysis"
    os.makedirs(plots_dir, exist_ok=True)
    
    # Find all ada_mixp folders
    ada_mixp_folders = find_ada_mixp_folders(base_dir)
    print(f"Found {len(ada_mixp_folders)} ada_mixp folders")
    
    # Process each folder
    all_results = {}
    for folder in ada_mixp_folders:
        results = process_adaptive_model(folder, base_results_dir, plots_dir)
        if results:
            model_name = os.path.basename(folder)
            all_results[model_name] = results
    
    # Save summary of all models
    with open(os.path.join(plots_dir, 'all_models_summary.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
        
    # Create summary table for optimal thresholds
    summary = {
        'model_name': [],
        'confidence_threshold': [],
        'precision_threshold': [],
        'final_network_usage': [],
        'final_network_accuracy': [],
        'final_overall_accuracy': [],
        'final_time_improvement': []
    }
    
    for model_name, results in all_results.items():
        summary['model_name'].append(model_name)
        summary['confidence_threshold'].append(results['thresholds']['confidence'])
        summary['precision_threshold'].append(results['thresholds']['precision'])
        
        # Get the final (last) values for the metrics
        metrics = results['metrics']
        last_idx = -1  # Use the last available index
        
        network_usage = metrics['network_plan_percentages'][2][last_idx] if last_idx < len(metrics['network_plan_percentages'][2]) else None
        network_acc = metrics['only_network_precisions'][last_idx] if last_idx < len(metrics['only_network_precisions']) else None
        overall_acc = metrics['overall_precisions'][last_idx] if last_idx < len(metrics['overall_precisions']) else None
        
        # Calculate time improvement compared to baseline planner
        baseline_time = metrics['planner_times'][0] if metrics['planner_times'] else None
        final_time = metrics['overall_times'][last_idx] if last_idx < len(metrics['overall_times']) else None
        
        if baseline_time and final_time and not np.isnan(baseline_time) and not np.isnan(final_time) and final_time > 0:
            time_improvement = baseline_time / final_time
        else:
            time_improvement = None
        
        summary['final_network_usage'].append(network_usage)
        summary['final_network_accuracy'].append(network_acc)
        summary['final_overall_accuracy'].append(overall_acc)
        summary['final_time_improvement'].append(time_improvement)
    
    # Create a DataFrame for the summary
    summary_df = pd.DataFrame(summary)
    
    # Format the table nicely
    summary_df['confidence_threshold'] = summary_df['confidence_threshold'].apply(lambda x: f"{x:.4f}")
    summary_df['precision_threshold'] = summary_df['precision_threshold'].apply(lambda x: f"{x:.4f}")
    summary_df['final_network_usage'] = summary_df['final_network_usage'].apply(lambda x: f"{x:.2f}%" if x is not None else "N/A")
    summary_df['final_network_accuracy'] = summary_df['final_network_accuracy'].apply(lambda x: f"{x:.2f}%" if x is not None else "N/A")
    summary_df['final_overall_accuracy'] = summary_df['final_overall_accuracy'].apply(lambda x: f"{x:.2f}%" if x is not None else "N/A")
    summary_df['final_time_improvement'] = summary_df['final_time_improvement'].apply(lambda x: f"{x:.2f}x" if x is not None else "N/A")
    
    # Save the summary table
    summary_df.to_csv(os.path.join(plots_dir, 'summary_table.csv'), index=False)
    
    # Print the summary
    print("\nSummary of models:")
    print(summary_df)

if __name__ == "__main__":
    main()