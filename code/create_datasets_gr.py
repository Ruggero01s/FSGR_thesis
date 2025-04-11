import os
from utils_unibs.files import load_from_folder, save_file
import click
from goal_rec_utils.plan import Plan


def get_problem_number(problem_name: str):
    return int(problem_name.split('gr=p', 1)[1].split(',', 1)[0])

@click.command()
@click.option('--source-dir', '-s', 'source_dir', help='Directory containing the plans', required=True, type=click.Path(exists=True))
@click.option('--target-dir', '-t', 'target_dir', help='Directory where to save the datasets', required=True, type=click.Path(exists=False))
@click.option('--num-problems', '-n', 'num_problems', help='Number of problems', required=True, type=int)
def run(source_dir, target_dir, num_problems):
    # source_dir = '/home/mchiari/goal_recognition/datasets/sokoban/tasks_simil-pereira/gr_problems/sel_xmls'
    # target_dir = '/home/mchiari/goal_recognition/datasets/sokoban/tasks_simil-pereira/gr_testsets'
    # num_problems = 7
    gr_problems = list()
    for _ in range(num_problems):
        gr_problems.append(list())

    os.makedirs(target_dir, exist_ok=True)

    #plans = utils.get_all_plans(source_dir)
    [plans] = load_from_folder(source_dir, ['plans'])
    for plan in plans:
        problem_number = get_problem_number(os.path.basename(plan.plan_name))
        gr_problems[problem_number-1].append(plan)

    for i, plans in enumerate(gr_problems):
        save_file(plans, target_dir, f'test_plans_p{i+1:02}')


if __name__ == '__main__':
    run()
