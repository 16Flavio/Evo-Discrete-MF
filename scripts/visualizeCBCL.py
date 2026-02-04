import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import argparse
import math

def read_solution_file(filepath, m_expected):
    """
    Reads the output file to extract W and H.
    Handles the case where W and H do not have the same number of columns.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found : {filepath}")

    with open(filepath, 'r') as f:
        try:
            line = f.readline().strip()
            if not line: return 0, None, None
            score = float(line)
        except ValueError:
            print("Error: Unable to read score on the first line.")
            return 0, None, None
        
        raw_data = []
        for line in f:
            parts = list(map(float, line.split()))
            if len(parts) > 0:
                raw_data.append(parts)
        
    if m_expected is None:
        print("CRITICAL ERROR: Unable to separate W and H without knowing 'm'.")
        print("Please provide the original input file with --input.")
        sys.exit(1)

    if len(raw_data) <= m_expected:
        print(f"Error: The file seems incomplete. Read lines={len(raw_data)}, expected (W)={m_expected}")
        sys.exit(1)

    list_W = raw_data[:m_expected]
    list_H = raw_data[m_expected:]

    try:
        W = np.array(list_W)
        H = np.array(list_H)
    except Exception as e:
        print(f"Error during NumPy conversion : {e}")
        print("Check if the result file is not corrupt.")
        sys.exit(1)

    return score, W, H

def get_matrix_dims(input_file):
    """Reads the input file header to get m."""
    if not os.path.exists(input_file):
        print(f"Error: Input file not found : {input_file}")
        sys.exit(1)
    with open(input_file, 'r') as f:
        line = f.readline().split()
        m = int(line[0])
    return m

def plot_W_basis(W, score_val, img_h=None, img_w=None, title="Basis Images"):
    """
    Displays the columns of W as images.
    """
    m, r = W.shape
    
    if img_h is None or img_w is None:
        side = math.sqrt(m)
        if side.is_integer():
            img_h = int(side)
            img_w = int(side)
        else:
            print(f"ERROR: m={m} is not a perfect square (sqrt={side:.2f}).")
            print("Please force dimensions with --height and --width.")
            return

    print(f"Displaying : {r} images of size {img_h}x{img_w}")

    cols = int(math.ceil(math.sqrt(r)))
    rows = int(math.ceil(r / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2.5))
    if isinstance(axes, np.ndarray):
        axes_flat = axes.flatten()
    else:
        axes_flat = [axes]

    fig.suptitle(f"{title}\nFitness: {score_val}", fontsize=14)

    for i in range(r):
        ax = axes_flat[i]
        col_vector = W[:, i]
        
        try:
            img = col_vector.reshape((img_h, img_w))
            img = np.rot90(img, k=-1)
            ax.imshow(img, cmap='binary', interpolation='nearest')
            ax.set_title(f"Basis {i+1}")
            ax.axis('off')
        except ValueError:
            ax.text(0.5, 0.5, "Error", ha='center')

    for i in range(r, len(axes_flat)):
        axes_flat[i].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--result", required=True, help="Result file (W, H).")
    parser.add_argument("-i", "--input", required=True, help="Original matrix file (to read m).")
    parser.add_argument("--height", type=int, help="Image height.")
    parser.add_argument("--width", type=int, help="Image width.")
    
    args = parser.parse_args()

    m = get_matrix_dims(args.input)
    print(f"Detected dimension : m={m}")
    
    score_val, W, H = read_solution_file(args.result, m_expected=m)
    
    plot_W_basis(W, score_val, img_h=args.height, img_w=args.width, 
                 title=f"Columns of W ({os.path.basename(args.result)})")