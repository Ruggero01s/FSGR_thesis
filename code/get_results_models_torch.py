#!/usr/bin/env python3
# filepath: /home/deeplearning/ruggero/FSGR_thesis/run_ada_mixp_models.py

import os
import subprocess
import datetime
import time

def run_command_on_ada_mixp_folders():
    # Directory to search for ada_mixp folders
    search_dir = "datasets/gr_logistics/results/adaptive_incremental"
    python_env = "/opt/anaconda/anaconda3/envs/FSGR/bin/python"
    script = "code/get_predictions_and_results.py"
    base_output_dir = "datasets/gr_logistics/results/results_jsons"
    
    # Create a log file to track progress
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"ada_mixp_processing_{timestamp}.log"
    
    # Find all directories containing "ada_mixp" in their name
    ada_mixp_folders = []
    for root, dirs, files in os.walk(search_dir):
        for dir_name in dirs:
            if "ada_mixp" in dir_name:
                full_path = os.path.join(root, dir_name)
                ada_mixp_folders.append(full_path)
    
    print(f"Found {len(ada_mixp_folders)} folders containing 'ada_mixp'")
    with open(log_file, "w") as log:
        log.write(f"Starting processing at {datetime.datetime.now()}\n")
        log.write(f"Found {len(ada_mixp_folders)} folders containing 'ada_mixp'\n\n")
    
    # Process each folder
    for i, folder in enumerate(ada_mixp_folders, 1):
        folder_name = os.path.basename(folder)
        print(f"[{i}/{len(ada_mixp_folders)}] Processing {folder}")
        
        # Create output directory with same name as the model folder
        output_dir = os.path.join(base_output_dir, folder_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # Build the command
        cmd = [
            python_env,
            script,
            "--model-path", folder,
            "--read-dict-dir", "datasets/gr_logistics/pickles",
            "--read-test-plans-dir", "datasets/logistics/optimal_plans/gr_testset",
            "--target-dir", output_dir,
            "--problems-dir", "datasets/logistics/optimal_plans/gr_dataset/logistics_sel_gr_problems",
            "--max-plan-dim", "32",
            "--max-plan-perc", "1",
            "--threshold", "0.4"
        ]
        
        # Log the command
        cmd_str = " ".join(cmd)
        with open(log_file, "a") as log:
            log.write(f"[{i}/{len(ada_mixp_folders)}] Processing {folder}\n")
            log.write(f"Output directory: {output_dir}\n")
            log.write(f"Executing: {cmd_str}\n")
        
        # Execute the command
        start_time = time.time()
        try:
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT,
                universal_newlines=True
            )
            
            # Capture and log output in real-time
            for line in process.stdout:
                print(line.strip())
                with open(log_file, "a") as log:
                    log.write(line)
                    
            process.wait()
            
            # Check for errors
            if process.returncode != 0:
                print(f"Error processing {folder}, return code: {process.returncode}")
                with open(log_file, "a") as log:
                    log.write(f"Error processing {folder}, return code: {process.returncode}\n")
        except Exception as e:
            print(f"Exception while processing {folder}: {e}")
            with open(log_file, "a") as log:
                log.write(f"Exception while processing {folder}: {e}\n")
        
        elapsed = time.time() - start_time
        print(f"Completed processing {folder} in {elapsed:.1f} seconds")
        with open(log_file, "a") as log:
            log.write(f"Completed processing {folder} in {elapsed:.1f} seconds\n")
            log.write("-" * 50 + "\n")
    
    print(f"All ada_mixp folders processed. See {log_file} for details.")
    with open(log_file, "a") as log:
        log.write(f"\nAll processing completed at {datetime.datetime.now()}\n")

if __name__ == "__main__":
    run_command_on_ada_mixp_folders()