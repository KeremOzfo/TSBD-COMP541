import subprocess
import multiprocessing
import os

def run_experiment(args_list):
    """Run a single experiment with the given args."""
    cmd = ['python', 'main.py'] + args_list
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error in {' '.join(cmd)}: {result.stderr}")
    else:
        print(f"Completed: {' '.join(cmd)}")

if __name__ == '__main__':
    # Define your predefined combinations as lists of argument strings
    combinations = [
        # Example: Model and dataset combos (add more as needed)
        ['--model', 'TimesNet', '--root_path', './dataset/UWaveGestureLibrary', '--gpu_id', 'cuda:0'],
        ['--model', 'PatchTST', '--root_path', './dataset/UWaveGestureLibrary', '--gpu_id', 'cuda:0'],
        ['--model', 'TimesNet', '--root_path', './dataset/AnotherDataset', '--gpu_id', 'cuda:1'],  # If you have another dataset/GPU
        ['--model', 'PatchTST', '--root_path', './dataset/AnotherDataset', '--gpu_id', 'cuda:1'],
    ]

    # Number of parallel processes (match your CPU cores or GPUs)
    num_processes = min(len(combinations), multiprocessing.cpu_count())  # Or set to number of GPUs

    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.map(run_experiment, combinations)