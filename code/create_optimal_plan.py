import os
import click
import subprocess
import shlex


PROBLEM_PATH = 0
OUTPUT_DIR = 1
NC_FILE_DIR = 2
FD_PATH = 3
PYTHON_PATH =4
EXECUTION_TIME = 5
ALIAS = 6
NC_FILE_NAME = 7


def run_command(command: str) -> int:
    print(command)
    res = subprocess.run(shlex.split(command))
    return res.returncode

@click.group()
@click.pass_context
@click.option('--input', '-i', 'problem_path', type=click.STRING, required=True, prompt=False)
@click.option('--output-dir', 'output_dir', type=click.STRING, required=True, prompt=False)
@click.option('--nc-file-dir', 'nc_file_dir', type=click.STRING, required=True, prompt=False)
@click.option('--fast-downward-path', 'fd_path', type=click.STRING,
              default='/data/users/mchiari/WMCA/fast-downward/fast-downward.py')

@click.option('--execution-time', 'execution_time', type=click.INT, default=60)

@click.option('--nc-file-name', 'nc_file_name', type=click.STRING, default='not_completed_problems')
def run(ctx, problem_path, output_dir, nc_file_dir, fd_path, execution_time, nc_file_name):
    ctx.ensure_object(dict)
    ctx.obj[PROBLEM_PATH] = problem_path
    ctx.obj[OUTPUT_DIR] = output_dir
    ctx.obj[NC_FILE_DIR] = nc_file_dir
    ctx.obj[FD_PATH] = fd_path
    ctx.obj[EXECUTION_TIME] = execution_time
    ctx.obj[NC_FILE_NAME] = nc_file_name

@run.command('solution')
@click.option('--python-path', 'python_path', type=click.STRING,
              default='/opt/anaconda/anaconda3/envs/goal_rec/bin/python')
@click.option('--alias', type=click.STRING, default= 'seq-opt-bjolp')
@click.pass_context
def sol(ctx, python_path, alias):
    if ctx.ensure_object(dict):
        problem_path = ctx.obj[PROBLEM_PATH]
        output_dir = ctx.obj[OUTPUT_DIR]
        nc_file_dir = ctx.obj[NC_FILE_DIR]
        fd_path = ctx.obj[FD_PATH]
        execution_time = ctx.obj[EXECUTION_TIME]
        nc_file_name = ctx.obj[NC_FILE_NAME]
        print('RUNNING')
        not_completed_file = os.path.join(nc_file_dir, f'{nc_file_name}.txt')
        plans_path = problem_path.rsplit('/',1)[0]
        problem_name = problem_path.rsplit('/', 1)[1]
        domain_path = os.path.join(plans_path, 'domain.pddl')

        old_not_completed_file = '/data/users/mchiari/WMCA/zeno_not_complete_gr.txt'
        with open(old_not_completed_file, 'r') as rf:
            lines = rf.readlines()
        lines = [line.strip() for line in lines]

        os.makedirs(output_dir, exist_ok=True)
        if problem_name.endswith('.pddl') and not problem_name.startswith('domain'):
            problem_name = problem_name.rsplit('.', 1)[0]
            
            if not problem_name in lines:
                print(f'Problem {problem_name} already completed')
                return

            target_path = os.path.join(output_dir, f'{problem_name}.SOL')

            new_fd_path = fd_path.rsplit('/', 2)[0]
            new_fd_path = os.path.join(new_fd_path, problem_name)
            os.makedirs(new_fd_path, exist_ok=True)

            os.chdir(new_fd_path)
            cp_command = f'ln -s {fd_path.rsplit("/",1)[0]} {new_fd_path}'
            res = run_command(cp_command)

            full_fd_path = os.path.join(new_fd_path, fd_path.rsplit('/', 2)[1])
            os.chdir(full_fd_path)
            print(full_fd_path)

            fd_command = f'time timeout --signal=HUP {execution_time} ./{fd_path.rsplit("/", 2)[2]} --alias {alias} --sas-file {os.path.join(new_fd_path, "output.sas")} --plan-file {target_path} {domain_path} {problem_path} '
            #fd_command = f'time timeout --signal=HUP {execution_time} ./{fd_path.rsplit("/", 2)[2]} --sas-file {os.path.join(new_fd_path, "output.sas")} --plan-file {target_path} {domain_path} {problem_path} --search "astar(celmcut())"'
            res = run_command(fd_command)
            if res != 0:
                with open(not_completed_file, 'a') as wf:
                    wf.write(f'{problem_name}\n')

            rm_command = f'rm -rf {new_fd_path}'
            res = run_command(rm_command)



@run.command('xml')
@click.pass_context
@click.option('--solution-dir', 'sol_dir', type=click.STRING, required=True, prompt=False)
def xml(ctx, sol_dir):
    if ctx.ensure_object(dict):
        problem_path = ctx.obj[PROBLEM_PATH]
        output_dir = ctx.obj[OUTPUT_DIR]
        nc_file_dir = ctx.obj[NC_FILE_DIR]
        fd_path = ctx.obj[FD_PATH]
        execution_time = ctx.obj[EXECUTION_TIME]
        nc_file_name = ctx.obj[NC_FILE_NAME]

        not_completed_file = os.path.join(nc_file_dir, f'{nc_file_name}.txt')
        plans_path = problem_path.rsplit('/', 1)[0]
        problem_name = problem_path.rsplit('/', 1)[1]
        domain_path = os.path.join(plans_path, 'domain.pddl')

        os.makedirs(output_dir, exist_ok=True)

        if problem_name.endswith('.pddl') and not problem_name.startswith('domain'):
            problem_name = problem_name.rsplit('.', 1)[0]
            solutions_names = os.listdir(sol_dir)
            for solution_name in solutions_names:
                if problem_name in solution_name:
                    sol_name = solution_name
            # sol_name = f'{problem_name}.SOL'
            sol_path = os.path.join(sol_dir, sol_name)

            if os.path.isfile(sol_path):
                target_path = os.path.join(output_dir, f'xml-LPG-{sol_name}')

                new_fd_path = fd_path.rsplit('/', 2)[0]
                new_fd_path = os.path.join(new_fd_path, problem_name)
                os.makedirs(new_fd_path, exist_ok=True)

                cp_command = f'ln -s {fd_path.rsplit("/", 1)[0]} {new_fd_path}'
                res = run_command(cp_command)

                full_fd_path = os.path.join(new_fd_path, fd_path.rsplit('/', 2)[1])
                os.chdir(full_fd_path)

                fd_command = f'time timeout --signal=HUP {execution_time} ./{fd_path.rsplit("/", 2)[2]} -o {domain_path} -f {problem_path} ' \
                             f'-input_plan {sol_path} -n 1 -xml_addition_info -out {target_path} -v off'

                res = run_command(fd_command)
                if res != 0:
                    with open(not_completed_file, 'a') as wf:
                        wf.write(f'{problem_name}\n')

                if os.path.isfile(f'{target_path}_1.SOL'):
                    rm_file_command = f'rm {target_path}_1.SOL'
                    res = run_command(rm_file_command)

                rm_command = f'rm -r {new_fd_path}'
                res = run_command(rm_command)

if __name__ == '__main__':
    run()