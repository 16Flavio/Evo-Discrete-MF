import numpy as np
import os

def read_txt_file_bmf(filename: str):
    """
    Reads a matrix instance file in the project's format.

    Args:
        filename (str): The absolute path to the .txt file.

    Returns:
        - X (np.ndarray): The X matrix (m x n) as integers.
    """
    print(f"  -> Reading instance : {filename}")
    if not os.path.exists(filename):
        print(f"  -> ERROR: File not found.")
        return None

    with open(filename, 'r') as f:
        try:
            premiere_ligne = f.readline().split()
            m, n = map(int, premiere_ligne)
        except Exception as e:
            print(f"  -> ERROR: Unable to read first line. {e}")
            return None
            
        print(f"     Params: m={m}, n={n}")

        try:
            X = np.loadtxt(f, dtype=int, max_rows=m)
        except Exception as e:
            print(f"  -> ERROR: Unable to read matrix X. {e}")
            return None

        if X.shape != (m, n):
            print(f"  -> ERROR: Read matrix size {X.shape}, expected ({m}, {n})")
            return None
            
        print(f"     Matrix X of size {X.shape} read successfully.")
        return X

def read_txt_file(filename: str):
    """
    Reads a matrix instance file in the project's format.

    Args:
        filename (str): The absolute path to the .txt file.

    Returns:
        tuple: (X, LW, UW, LH, UH)
            - X (np.ndarray): The X matrix (m x n) as integers.
            - LW, UW (int): Bounds for W.
            - LH, UH (int): Bounds for H.
    """
    print(f"  -> Reading instance : {filename}")
    if not os.path.exists(filename):
        print(f"  -> ERROR: File not found.")
        return None, 0, 0, 0, 0

    with open(filename, 'r') as f:
        try:
            premiere_ligne = f.readline().split()
            m, n, LW, UW, LH, UH = map(int, premiere_ligne)
        except Exception as e:
            print(f"  -> ERROR: Unable to read first line. {e}")
            return None, 0, 0, 0, 0
            
        print(f"     Params: m={m}, n={n}, LW={LW}, UW={UW}, LH={LH}, UH={UH}")

        try:
            X = np.loadtxt(f, dtype=int, max_rows=m)
        except Exception as e:
            print(f"  -> ERROR: Unable to read matrix X. {e}")
            return None, 0, 0, 0, 0

        if X.shape != (m, n):
            print(f"  -> ERROR: Read matrix size {X.shape}, expected ({m}, {n})")
            return None, 0, 0, 0, 0
            
        print(f"     Matrix X of size {X.shape} read successfully.")
        return X, LW, UW, LH, UH
    
def write_txt_file(filename: str, f_obj: int, W: np.ndarray, H: np.ndarray):
    """
    Writes the solution (objective, W, H) to a file in the required format,
    ONLY if the new solution is BETTER (lower score) than the existing one.
    
    Format:
    Line 1: Objective value (rounded to integer)
    Following lines: Matrix W (m rows, r integers per line)
    Following lines: Matrix H (r rows, n integers per line)
    
    Args:
        filename (str): Full path of the file to create.
        f_obj (float or int): Final value of the objective function.
        W (np.ndarray): Matrix W (m x r).
        H (np.ndarray): Matrix H (r x n).
    """
    current_obj = int(np.round(f_obj))

    if os.path.exists(filename):
        try:
            with open(filename, 'r') as f:
                first_line = f.readline().strip()
                if first_line:
                    old_obj = int(float(first_line))
                    
                    if current_obj >= old_obj:
                        print(f"  -> Solution not saved : New score {current_obj} >= Old score {old_obj}")
                        return 
                    else:
                        print(f"  -> Improvement found ! ({old_obj} -> {current_obj})")
        except Exception as e:
            print(f"  -> Warning : Unable to read existing file for comparison ({e}). Overwriting as precaution.")

    print(f"  -> Writing solution to : {filename}")
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
            
        with open(filename, 'w') as f:
            f.write(f"{current_obj}\n")
            
            for i in range(W.shape[0]):
                row_str = " ".join(map(str, W[i]))
                f.write(row_str + "\n")
                
            for i in range(H.shape[0]):
                row_str = " ".join(map(str, H[i]))
                f.write(row_str + "\n")
                
    except Exception as e:
        print(f"  -> ERROR: Unable to write output file. {e}")