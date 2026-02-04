import os
import sys
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.append(PROJECT_ROOT)

try:
    from utils.file_utils import read_txt_file_bmf
except ImportError:
    print("ERROR: Unable to import 'utils'. Ensure you are running the script from the correct location.")
    sys.exit(1)

INPUT_MATRIX = os.path.join(PROJECT_ROOT, "data", "bmf_matrix", "binarizedCBCL.txt")

RESULT_FILE = os.path.join(PROJECT_ROOT, "results", "bmf_matrix", "binarizedCBCL_r20_20min.txt") 

def main():
    """
    Calculates the relative error between the original value and the result value.
    
    Reads the input matrix to calculate its norm, reads the absolute error (fitness)
    from the result file, and then computes the relative error. Prints the absolute
    error, relative error, and the error as a percentage.
    """
    print("=== RELATIVE ERROR CALCULATION ===")
    print(f"Source Matrix : {os.path.basename(INPUT_MATRIX)}")
    print(f"Result File   : {os.path.basename(RESULT_FILE)}")
    print("-" * 40)

    if not os.path.exists(INPUT_MATRIX):
        print(f"ERROR: Matrix file not found : {INPUT_MATRIX}")
        return

    X = read_txt_file_bmf(INPUT_MATRIX)
    
    if X is None:
        print("Error reading the matrix.")
        return

    norm_x = np.sum(X)
    print(f"Norm of X (Sum of elements) : {norm_x}")

    if not os.path.exists(RESULT_FILE):
        print(f"ERROR: Result file not found : {RESULT_FILE}")
        return

    try:
        with open(RESULT_FILE, 'r') as f:
            line = f.readline().strip()
            absolute_error = float(line)
    except Exception as e:
        print(f"ERROR: Unable to read score from result file ({e})")
        return

    print(f"Absolute Error (Fitness)       : {absolute_error}")

    if norm_x == 0:
        print("Warning: Matrix X is empty (norm 0), division impossible.")
    else:
        relative_error = absolute_error / norm_x
        print("-" * 40)
        print(f"RELATIVE ERROR                 : {relative_error:.6f}")
        print(f"As percentage                  : {relative_error * 100:.4f} %")

if __name__ == "__main__":
    main()