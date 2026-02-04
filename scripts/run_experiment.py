import os
import subprocess
import time
import re
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
MAIN_SCRIPT = BASE_DIR / "main.py"
RESULTS_BASE_DIR = BASE_DIR / "results"

DATA_DIRS_MAP = {
    "bmf": BASE_DIR / "data/bmf_matrix",
    "imf": BASE_DIR / "data/imf_matrix"
}

NUM_RUNS = 30

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
    
    elif matrix_type == "imf":
        time_imf = 300 
        
        if "instance1" in filename:
            configs.append({"rank": 2, "time": time_imf})
        elif "instance2" in filename:
            configs.append({"rank": 5, "time": time_imf})
        elif "instance3" in filename:
            configs.append({"rank": 5, "time": time_imf})
        elif "instance4" in filename:
            configs.append({"rank": 10, "time": time_imf})
        elif "instance5" in filename:
            configs.append({"rank": 10, "time": 2400})
        elif "Houdain" in filename:
            configs.append({"rank": 10, "time": time_imf})
        else:
            print(f"[INFO] Generic configuration applied for {filename} (r=10, 5min)")
            configs.append({"rank": 10, "time": 300})
            
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

                for run_i in range(1, NUM_RUNS + 1):
                    output_filename = f"{filename.replace('.txt', '')}_r{rank}_{duration_str}.txt"
                    output_path = output_dir_base / output_filename

                    print(f"     Launching Run {run_i}/{NUM_RUNS}...")

                    command = [
                        "python", str(MAIN_SCRIPT),
                        "-i", str(file_path),
                        "-o", str(output_path),
                        "--time", str(time_limit),
                        "--rank", str(rank),
                        "-n", "60",
                        "-s", "4",
                        "-m", "0.0",
                        "--factorization-mode", matrix_type.upper(),
                    ]

                    try:
                        result = subprocess.run(command, check=True, text=True, capture_output=True)
                        output = result.stdout
            
                        error_pattern = re.compile(r"Fitness Solver\s*:\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)
                        match = error_pattern.search(output)
                        if match:
                            err_value = float(match.group(1))
                            print(f"Error detected = {err_value:.6f}")
                    except subprocess.CalledProcessError as e:
                        print(f"     ERROR Run {run_i} on {filename}: {e}")
                    except KeyboardInterrupt:
                        print("\nManual stop detected. End of script.")
                        return
                    
                    break

    print("\n--- All experiments completed ! ---")

if __name__ == "__main__":
    run_solver()