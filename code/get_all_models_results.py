import os
import click
from multiprocessing import Pool
from functools import partial

def run_command(models_dir, test_plan_dir, target_dir, max_plan_dim, dict_dir, problems_dir, i):
    model_name = f'model_{i}'
    command = f'python /data/users/mchiari/WMCA/code/get_predictions_and_results.py --model-path {models_dir} --read-dict-dir {dict_dir} --read-test-plans-dir {test_plan_dir} --target-dir {os.path.join(target_dir, model_name)} --max-plan-dim {max_plan_dim} --model-number {i} -p {problems_dir}'
    os.system(command)

@click.command()
@click.option('--models-dir', '-m', 'models_dir', help='Directory containing the models', required=True, type=click.Path(exists=True))
@click.option('--read-test-plans-dir', '-s', 'test_plan_dir', help='Directory containing the test plans', required=True, type=click.Path(exists=True))
@click.option('--target-dir', '-t', 'target_dir', help='Directory where to save the results', required=True, type=click.Path(exists=False))
@click.option('--max-plan-dim', 'max_plan_dim', help='Maximum plan dimension', required=True, type=int)
@click.option('--total-models', '-n', 'total_models', help='Total number of models', required=True, type=int)
@click.option('--read-dict-dir', '-d', 'dict_dir', help='Directory containing the dictionaries', required=True, type=click.Path(exists=True))
@click.option('--problems-dir', '-p', 'problems_dir', help='Directory containing the problems', required=True, type=click.Path(exists=True))
def run(models_dir, test_plan_dir, target_dir, max_plan_dim, total_models, dict_dir, problems_dir):

    # for i in range(0, total_models):
    #     model_name = f'model_{i}'
    #     command = f'python /data/users/mchiari/WMCA/code/get_predictions_and_results.py --model-path {models_dir} --read-dict-dir {dict_dir} --read-test-plans-dir {test_plan_dir} --target-dir {os.path.join(target_dir, model_name)} --max-plan-dim {max_plan_dim} --model-number {i} -p {problems_dir}'
    #     os.system(command)
    parallel_process = 40
    target_dir = os.path.join(target_dir, os.path.basename(models_dir))
    parallel_process = min(parallel_process, total_models)
    with Pool(parallel_process) as p:
        f = partial(run_command, models_dir, test_plan_dir, target_dir, max_plan_dim, dict_dir, problems_dir)
        p.map(f, range(30, total_models))

if __name__ == '__main__':
    run()