import subprocess
import re
import statistics
import sys

def run_benchmark(num_trials=30):
    """
    Executes main.py for a specified number of trials, captures the error 
    values from the console output using regular expressions, and 
    calculates the average, standard deviation, and range of the results.
    """
    script_name = "main.py"
    
    # params = [
    #     "--input", "data/imf_matrix/Houdain_547x745_m1616m1616.txt",
    #     "--output", "results/imf_matrix/Houdain_547x745_m1616m1616_r10_30sec.txt",
    #     "--rank", "10",
    #     "--time", "30.0",
    #     "-n", "30",
    #     "-s", "3",
    #     "-m", "0.383",
    #     "--factorization-mode", "IMF",
    # ]

    params = [
        "--input", "data/bmf_matrix/lymp.txt",
        "--output", "results/bmf_matrix/lymp_r10_30sec.txt",
        "--rank", "10",
        "--time", "30.0",
        "-n", "60",
        "-s", "3",
        "-m", "0.383",
        "--factorization-mode", "BMF",
    ]

    # params = [
    #     "--input", "data/bmf_matrix/binarizedCBCL.txt",
    #     "--output", "results/bmf_matrix/binarizedCBCL_r10_30sec.txt",
    #     "--rank", "10",
    #     "--time", "30.0",
    #     "-n", "60",
    #     "-s", "3",
    #     "-m", "0.383",
    #     "--factorization-mode", "BMF",
    # ]
    
    error_pattern = re.compile(r"Fitness Solver\s*:\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)
    errors = []
    
    print(f"Starting {num_trials} trials for {script_name}...")
    print(f"Parameters: {' '.join(params)}")
    print("-" * 40)

    for i in range(1, num_trials + 1):
        try:
            command = [sys.executable, script_name] + params
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            output = result.stdout
            
            match = error_pattern.search(output)
            if match:
                err_value = float(match.group(1))
                errors.append(err_value)
                print(f"Trial {i:02d}/{num_trials}: Error detected = {err_value:.6f}")
            else:
                print(f"Trial {i:02d}/{num_trials}: [WARNING] No error value found in console output.")
                
        except subprocess.CalledProcessError as e:
            print(f"Trial {i:02d}/{num_trials}: [ERROR] Script execution failed.")
            print(e.stderr)
        except Exception as e:
            print(f"Trial {i:02d}/{num_trials}: [ERROR] {str(e)}")

    print("-" * 40)
    if errors:
        avg_error = statistics.mean(errors)
        std_dev = statistics.stdev(errors) if len(errors) > 1 else 0.0
        
        print(f"Results across {len(errors)} successful trials:")
        print(f"Mean Error    : {avg_error:.8f}")
        print(f"Std Deviation : {std_dev:.8f}")
        print(f"Minimum       : {min(errors):.8f}")
        print(f"Maximum       : {max(errors):.8f}")
    else:
        print("Failed to collect any error data.")

if __name__ == "__main__":
    run_benchmark(30)