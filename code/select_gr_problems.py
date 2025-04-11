import os
import numpy as np
from create_planner_files import get_problems, get_hyp_number
import shutil

def select_problems(problem_dir: str, target_dir: str, problem_number: str, num_samples: int):
    problems = get_problems(problem_dir)
    problems = [p for p in problems if f'gr=p{problem_number:02}' in os.path.basename(p)]
    l = [int(os.path.basename(p).split('hyp=hyp-', 1)[1].split('.', 1)[0]) for p in problems]
    num_goals = max(l) + 1
    np.random.shuffle(problems)
    problems_per_hyp = int(np.floor(num_samples / num_goals))
    sel_problems = list()
    count = np.zeros((num_goals,))
    for _ in range(num_samples - (problems_per_hyp * num_goals)):
        sel_problems.append(problems.pop(0))
    for p in problems:
        if len(sel_problems) == num_samples:
            break
        hyp_number = get_hyp_number(os.path.basename(p))
        if count[hyp_number] < problems_per_hyp:
            sel_problems.append(p)
            count[hyp_number] += 1

    os.makedirs(target_dir, exist_ok=True)
    for p in sel_problems:
        shutil.copy(p, target_dir)
    print(count, problems_per_hyp, num_samples - (problems_per_hyp * problem_number))
        
if __name__ == '__main__':
    problems_dir = '/data/storage/dataset/minsky/zenotravel/tasks_simil_pereira/gr_problems/plans/'
    target_dir = '/data/users/mchiari/WMCA/datasets/zenotravel/optimal_plans/selected_gr_problems'
    num_samples = 100
    problem_number = 7
    for i in range(1, 8):
        select_problems(problems_dir, target_dir, i, num_samples)

