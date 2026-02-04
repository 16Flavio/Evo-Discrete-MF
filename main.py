import os, sys
import argparse
import numpy as np
import random
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.append(src_dir)

from utils.verification import fobj, fobj_bmf, fobj_relu, solutionIsFeasible
from utils.file_utils import read_txt_file, write_txt_file, read_txt_file_bmf
from src.solver import metaheuristic

def main():
    """
    Main entry point for the Evo-Discrete-MF matrix optimization script.
    
    Parses command-line arguments to configure the matrix factorization process,
    including input/output paths, rank, time limits, population settings, and
    factorization mode (IMF, BMF, or RELU). Initializes the random seed if provided,
    loads the input matrix, executes the metaheuristic solver, and verifies the
    resulting solution before saving it to the output file.
    """
    parser = argparse.ArgumentParser(
        description="Main script for matrix optimization (Evo Discrete MF)."
    )
    parser.add_argument("-i", "--input", help="Path to input matrix file.", required=True)
    parser.add_argument("-o", "--output", help="Path to output file.", required=True)
    
    parser.add_argument("-r", "--rank", help="Factorization rank.", type=int, default=10)
    parser.add_argument("-t", "--time_limit", help="Time limit (s). Default: 300.0", type=float, default=300.0)
    parser.add_argument("-n", "--population_size", help="Population size. Default: 50", type=int, default=50)
    parser.add_argument("-s", "--tournament_size", help="Tournament size. Default: 3", type=int, default=3)

    parser.add_argument("--seed", help="Random seed for reproducibility. Default: None (Random)", type=int, default=None)
    
    parser.add_argument("--debug-mode", action="store_true", help="Enable debug mode for more info")
    parser.add_argument("--factorization-mode", type=str, choices=['IMF', 'BMF', 'RELU'], default='IMF',
                        help="Factorization mode: IMF (integer), BMF (binary), RELU (with ReLU)")

    args = parser.parse_args()

    if args.seed is not None:
        print("Fixing random seed to:", args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
    else:
        print("No seed provided, using random seed.")

    print(f"\n=== Launching Evo-Discrete-MF ===")
    print(f"Input: {args.input}")

    try:
        if args.factorization_mode == "BMF":
            LW = 0
            UW = 1
            LH = 0
            UH = 1
            X = read_txt_file_bmf(args.input)
        else:
            X, LW, UW, LH, UH = read_txt_file(args.input)
    except FileNotFoundError:
        print(f"Error: File {args.input} not found.")
        return

    debug = False
    if args.debug_mode:
        debug = True

    W, H, f_val = metaheuristic(
        X, args.rank, LW, UW, LH, UH, args.factorization_mode,
        TIME_LIMIT=args.time_limit, 
        N=args.population_size, 
        tournament_size=args.tournament_size,
        debug_mode=debug
    )

    print("\n=== Final Verification ===")
    if args.factorization_mode == "BMF":
        f_check = fobj_bmf(X, W, H)
    elif args.factorization_mode == "RELU":
        f_check = fobj_relu(X, W, H)
    else:
        f_check = fobj(X, W, H)
    print(f"Fitness Solver: {f_val}")
    print(f"Fitness Check : {f_check}")

    if f_val != f_check:
        print("/!\\ WARNING: Divergence between solver and verification /!\\")

    if not solutionIsFeasible(W, H, args.rank, LW, UW, LH, UH):
        print("/!\\ WARNING: Invalid solution (out of bounds) /!\\")
    else:
        print("Valid solution (constraints respected).")

    write_txt_file(args.output, f_val, W, H)
    print(f"Solution written to: {args.output}")

if __name__ == "__main__":
    main()