import os
import click
import shutil
from plan import Plan
from plan_generator import PlanGeneratorMultiPerc
import numpy as np


def get_problems(problems_dir):
    problems = [os.path.join(problems_dir, p) for p in os.listdir(problems_dir)
                if p.endswith('.pddl') and not p.startswith('domain')]
    return problems

def create_trace_file(plan: Plan, percentage: float, target_file: str):
    seed = plan.plan_name.rsplit("-p", 1)[1]
    seed = seed.split("_", 1)[0]
    try:
        int(seed)
    except ValueError:
        seed = seed.split(',', 1)[0]
    np.random.seed(int(seed))
    p = np.random.uniform(percentage, percentage)
    if p<1:
        actions = get_actions(plan.actions, p)
    else:
        actions = [f'({a.name})\n' for a in plan.actions]
    save_file(actions, target_file)

def get_actions(actions, perc):
    size = int(np.ceil(len(actions) * perc))
    if size == 0:
        size = 1
    indexes = np.ones(size, dtype=int) * -1
    i = 0
    ind_list = list(range(len(actions)))
    np.random.shuffle(ind_list)
    while i < size:
        ind = ind_list.pop(0)
        if ind not in indexes:
            indexes[i] = ind
            i += 1
    indexes = np.sort(indexes)
    return [f'{a.name}\n' for a in np.take(actions, indexes)]

def create_goals_file(problems_dir: str, num_goals: int, problem_number: int, target_file: str):
    problems = get_problems(problems_dir)
    goals = ''
    for hyp in range(num_goals):
        for p in problems:
            if f'gr=p{problem_number:02}' in p and f'hyp=hyp-{hyp}.' in p:
                goals += f'{extract_goal(p)}\n'
                break

    save_file(goals, target_file)


def create_correct_goal_file(problem_file: str, target_file: str):
    goal = extract_goal(problem_file)
    save_file([f'{goal}\n'], target_file)


def save_file(lines: list, target_file: str):
    with open(target_file, 'w') as wf:
        wf.writelines(lines)
        wf.close()
    

def extract_goal(p: str):
    with open(p, 'r') as rf:
        lines = rf.readlines()
        rf.close()

    for i, line in enumerate(lines):
        if line.strip().startswith('(:goal'):
            begin = i+2
            break
    line = lines[begin]
    goal = ''
    while not line.strip().startswith('))'):
        if goal != '':
            goal += ','
        goal += line.strip()
        begin +=1
        line = lines[begin]
    return goal[:-1]

def get_problem_number(problem_name: str):
    return int(problem_name.split('gr=p', 1)[1].split(',', 1)[0])

def get_hyp_number(problem_name: str):
    return int(problem_name.split('=hyp-')[1].rsplit('.',1)[0])

def get_num_goals(xml_dir: str, problem_number: int):
    l = os.listdir(xml_dir)
    l = [f for f in l if f'p_gr=p{problem_number:02}' in f]
    num_goals = 0
    for p in l:
        hyp = get_hyp_number(p)
        if hyp > num_goals:
            num_goals = hyp
    print(len(l), num_goals)
    return num_goals + 1


@click.command()
@click.option('--input', '-i', 'problem_file', required=True, prompt=False, help='Problem file to process')
def run(problem_file):
    xml_dir = '/data/users/mchiari/WMCA/datasets/blocksworld/optimal_plans/gr_problems/xmls'
    target_dir = '/data/users/mchiari/WMCA/blocksworld_dataset_gr/'
    percentages = [0.3, 0.5, 0.7]
    problem_name = os.path.basename(problem_file)
    problems_dir = os.path.dirname(problem_file)
    target_dir = os.path.join(target_dir, problem_name.rsplit('.', 1)[0])
    os.makedirs(target_dir, exist_ok=True)
    problem_number = get_problem_number(problem_name)
    num_goals = get_num_goals(xml_dir, problem_number)
    goals_file = os.path.join(target_dir, 'goals.txt')
    create_goals_file(problems_dir, num_goals, problem_number, goals_file)
    shutil.copy(problem_file, target_dir)
    correct_goal_file = os.path.join(target_dir, 'correct_goal.txt')
    create_correct_goal_file(problem_file, correct_goal_file)

    # solution_numbers = list(range(1,5))
    # np.random.shuffle(solution_numbers)
    # for sol_number in solution_numbers:
    #     xml_name = os.path.join(xml_dir, f'xml-LPG-{problem_name.rsplit(".", 1)[0]}_{sol_number}.SOL_1.SOL')
    #     print(xml_name)
    #     if os.path.isfile(xml_name):
    #         xml_file = xml_name
    #         break
    xml_file = os.path.join(xml_dir, f'xml-LPG-{problem_name.rsplit(".", 1)[0]}.SOL')
    plan = Plan(xml_file)
    for perc in percentages:
        trace_file = os.path.join(target_dir, f'{problem_name.rsplit(".", 1)[0]}_{int(perc * 100)}.txt')
        create_trace_file(plan, perc, trace_file)
    sol_file = os.path.join(target_dir, 'sol.txt')
    create_trace_file(plan, 1, sol_file)

if __name__ == '__main__':
    run()