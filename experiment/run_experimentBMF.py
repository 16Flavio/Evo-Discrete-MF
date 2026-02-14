import os
import subprocess
import time
import re
import statistics
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
MAIN_SCRIPT = BASE_DIR / "main.py"
RESULTS_BASE_DIR = BASE_DIR / "experiment"

DATA_DIRS_MAP = {
    "bmf": BASE_DIR / "data/bmf_matrix"
}

NUM_RUNS = 10

def get_configurations(filename, matrix_type):
    """
    Defines specific parameters (rank, time_limit) for each file.
    Returns a list of dictionaries: [{"rank": r, "time": t}, ...]
    """
    configs = []
    
    if matrix_type == "bmf":
        if filename == "binarizedCBCL.txt":
            configs.append({"rank": 10, "time": 1200})
            configs.append({"rank": 20, "time": 1200})
        else:
            ranks = [2, 5, 10]
            times = [30, 300]
            for r in ranks:
                for t in times:
                    configs.append({"rank": r, "time": t})
            
    return configs

def ensure_dir(directory):
    """Creates the directory if it does not exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def main_script_exists():
    """Checks if the main script exists."""
    return MAIN_SCRIPT.exists()

def run_solver():
    """Main function to run the experiments."""
    
    if not main_script_exists():
        print(f"Error : Unable to find {MAIN_SCRIPT}")
        return

    for matrix_type, data_dir in DATA_DIRS_MAP.items():
        if not data_dir.exists():
            print(f"Warning : Directory {data_dir} does not exist, skipping.")
            continue

        output_dir_base = RESULTS_BASE_DIR / data_dir.name
        ensure_dir(output_dir_base)

        files = [f for f in os.listdir(data_dir) if f.endswith(".txt")]
        files.sort()

        for filename in files:
            file_path = data_dir / filename
            
            file_configs = get_configurations(filename, matrix_type)
            
            if not file_configs:
                print(f"No configuration defined for {filename}, file ignored.")
                continue

            print(f"\n=== Processing {filename} ({len(file_configs)} configurations) ===")

            for config in file_configs:
                rank = config["rank"]
                time_limit = config["time"]
                
                if time_limit >= 60:
                    duration_str = f"{time_limit // 60}min"
                else:
                    duration_str = f"{time_limit}sec"
                
                print(f"  >> Config: Rank={rank}, Time={duration_str}")

                errors = []

                for run_i in range(1, NUM_RUNS + 1):

                    dataset_name = Path(filename).stem
                    report_filename = f"report_experiment_BMF/report_{dataset_name}_r{config['rank']}_{config['time']}seconds.txt"
                    report_path = RESULTS_BASE_DIR / report_filename

                    if os.path.exists(report_path):
                        break

                    output_filename = f"{filename.replace('.txt', '')}_r{rank}_{duration_str}.txt"
                    output_path = output_dir_base / output_filename

                    print(f"     Launching Run {run_i}/{NUM_RUNS}...")

                    current_seed = 41 + run_i

                    command = [
                        "python", str(MAIN_SCRIPT),
                        "-i", str(file_path),
                        "-o", str(output_path),
                        "--time", str(time_limit),
                        "--rank", str(rank),
                        "-n", "50",
                        "--factorization-mode", matrix_type.upper(),
                        "--seed", str(current_seed)
                    ]

                    try:
                        result = subprocess.run(command, check=True, text=True, capture_output=True)
                        output = result.stdout
            
                        error_pattern = re.compile(r"Fitness Solver\s*:\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)
                        match = error_pattern.search(output)
                        if match:
                            err_value = float(match.group(1))
                            print(f"Error detected = {err_value:.6f}")
                            errors.append(err_value)

                    except subprocess.CalledProcessError as e:
                        print(f"     ERROR Run {run_i} on {filename}: {e}")
                    except KeyboardInterrupt:
                        print("\nManual stop detected. End of script.")
                        return
                if len(errors) > 0:
                    mean_val = sum(errors) / len(errors)
                    std_dev = statistics.stdev(errors) if len(errors) > 1 else 0.0
                    min_val = min(errors)
                    max_val = max(errors)

                    print(f"mean : {mean_val}")
                    print(f"std deviation : {std_dev}")
                    print(f"min : {min_val}")
                    print(f"max : {max_val}")
                    
                    try:
                        with open(report_path, "w") as f_report:
                            f_report.write(f"Dataset: {dataset_name}\n")
                            f_report.write(f"Rank: {config['rank']}\n")
                            f_report.write(f"Time Limit: {config['time']} seconds\n")
                            f_report.write(f"Total Runs: {len(errors)}/{NUM_RUNS}\n")
                            f_report.write("-" * 30 + "\n")
                            f_report.write(f"Mean: {mean_val}\n")
                            f_report.write(f"Std Deviation: {std_dev}\n")
                            f_report.write(f"Min: {min_val}\n")
                            f_report.write(f"Max: {max_val}\n")
                            f_report.write("-" * 30 + "\n")
                            f_report.write("Raw Values:\n")
                            f_report.write(str(errors))
                        
                        print(f"Report saved to: {report_path}")
                    except Exception as e:
                        print(f"Error saving report: {e}")

    print("\n--- All experiments completed ! ---")

if __name__ == "__main__":
    run_solver()