import os

import click
from os.path import  join, isdir
from utils_unibs.files import load_from_folder
from multiprocess import Pool

def run_script(script: str):
    print(f'Running {script}')
    os.system(f'chmod +x {script}')
    os.system(script)


def clear_scripts_dir(scripts_path: str):
    if os.path.isdir(scripts_path):
        os.system(f'rm -r {scripts_path}')
    os.system(f'mkdir -p {scripts_path}')


@click.command()
@click.option('--python-path', 'python_path', type=click.STRING,
              default='/opt/anaconda/anaconda3/envs/goal_rec/bin/python', help='Path to Python bin file')
@click.option('--logger-dir', 'logger_dir', type=click.STRING,
              default='/home/mchiari/goal_recognition/logs/', help='Folder where to save logs')
@click.option('--file-path', 'file_path', type=click.STRING, help='Python file to execute')
@click.option('--nodes', type=click.INT, default=2, help='Number of nodes per qsub process')
@click.option('--source-dir', 'source_dir', type=click.STRING, help='Folder that contains the problems')
@click.option('--target-dir', 'target_dir', type=click.STRING, help='Folder where to save the plans')
@click.option('--addit-params', 'addit_params', type=click.STRING, help='Additional parameters to pass to the python file', default='')
@click.option('--memory-limit', 'memory_limit', type=click.INT, help='Memory limit in MB', default=-1)
@click.option('--test', is_flag=True, help='Flag for running only 5 instances')
def run(python_path, logger_dir, file_path, nodes, source_dir, target_dir, test, addit_params, memory_limit):
    scripts_path = '/data/users/mchiari/WMCA/scripts/'
    script_name = 'script_{0}.sh'

    clear_scripts_dir(scripts_path)
    os.makedirs(logger_dir, exist_ok=True)

    # plans = [os.path.join(source_dir, p) for p in os.listdir(source_dir) if p.lower().endswith('.pddl')
    #          and not p.startswith('domain')]
    plans = os.listdir(source_dir)
    # [not_completed_plans] = load_from_folder('/home/mchiari/state_embedding/', ['logistics_not_completed_gr.txt'])
    for i, plan in enumerate(plans):
        # if plan == 'domain.pddl' or f"{plan.rsplit('.',1)[0]}\n" not in not_completed_plans:
        #     continue
        with open(f'{scripts_path}{script_name.format(i)}', 'w') as f:
            f.write('#!/bin/bash\n')
            if memory_limit > 0:
                f.write(f'ulimit -v {memory_limit}\n')
            f.write(f'{python_path} {file_path} --input {os.path.join(source_dir,plan)} {addit_params} > {os.path.join(logger_dir, f"logs_script_{i}_plan{plan}.txt")} 2> {os.path.join(logger_dir, f"logs_script_{i}_plan{plan}.err.txt")}')
            #f.write(f'{python_path} {file_path} -p {os.path.join(source_dir,plan)} {addit_params} ')
            f.close()
        # os.system(
        #     #f'qsub -o {logger_dir}{plan}_out.log -e {logger_dir}{plan}out.err.log -q longbatch -l nodes=minsky.ing.unibs.it:ppn={nodes} {scripts_path}{script_name.format(i)}')
        #     f'chmod +x {scripts_path}{script_name.format(i)}')
        if test and i >= 4:
            break
    scripts = [os.path.join(scripts_path, s) for s in os.listdir(scripts_path) if s.startswith('script')]
    with Pool(nodes) as p:
        p.map(run_script, scripts)
    
    

if __name__ == '__main__':
    run()