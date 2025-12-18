import os, sys
import argparse
import numpy as np
import random
import time

# Ajout du chemin src au path pour les imports
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.append(src_dir)

from utils.verification import fobj, fobj_bmf, fobj_relu, solutionIsFeasible
from utils.file_utils import read_txt_file, write_txt_file, read_txt_file_bmf
from src.solver import metaheuristic
from src.config import ConfigAblation

def main():
    parser = argparse.ArgumentParser(
        description="Script principal pour l'optimisation de matrices (Evo Discrete MF)."
    )
    # Arguments obligatoires
    parser.add_argument("-i", "--input", help="Chemin vers le fichier matrice en entrée.", required=True)
    parser.add_argument("-o", "--output", help="Chemin vers le fichier de sortie.", required=True)
    
    # Paramètres Algo
    parser.add_argument("-r", "--rank", help="Rang de la factorisation.", type=int, default=10)
    parser.add_argument("-t", "--time_limit", help="Temps limite (s). Défaut: 300.0", type=float, default=300.0)
    parser.add_argument("-n", "--nombre_population", help="Taille population. Défaut: 50", type=int, default=50)
    parser.add_argument("-s", "--tournament_size", help="Taille tournoi. Défaut: 3", type=int, default=3)
    parser.add_argument("-m", "--mutation_rate", help="Taux mutation. Défaut: 0.1", type=float, default=0.1)

    # --- Seed ---
    parser.add_argument("--seed", help="Graine aléatoire (Seed) pour la reproductibilité. Défaut: None (Aléatoire)", type=int, default=None)

    # --- ARGUMENTS ABLATION STUDY ---
    parser.add_argument("--no-svd", action="store_true", help="Désactiver SVD Init")
    parser.add_argument("--no-kmeans", action="store_true", help="Désactiver KMeans Init")
    parser.add_argument("--no-nmf", action="store_true", help="Désactiver NMF Init")
    parser.add_argument("--no-greedy", action="store_true", help="Désactiver Greedy Init")
    
    parser.add_argument("--crossover", type=str, choices=['UNIFORM', 'MEAN', 'BOTH'], default='UNIFORM',
                        help="Type de Crossover : UNIFORM (mélange) ou MEAN (moyenne)")
    
    parser.add_argument("--restart-mode", type=str, choices=['FULL', 'SIMPLE'], default='FULL', 
                        help="Stratégie de restart : FULL (Smart) ou SIMPLE (Random)")
    
    parser.add_argument("--no-transpose", action="store_true", help="Désactiver le changement de phase (Transpose)")
    
    parser.add_argument("--debug-mode", action="store_true", help="Activer le mode debug pour plus d'infos")
    parser.add_argument("--mutation-type", type=str, choices=['SWAP', 'GREEDY', 'NOISE', 'ALL'], default='SWAP',
                        help="Type de mutation à utiliser")
    parser.add_argument("--factorization-mode", type=str, choices=['IMF', 'BMF', 'RELU'], default='IMF',    
                        help="Mode de factorisation : IMF (entière), BMF (binaire), RELU (avec ReLU)")
    # -------------------------------

    args = parser.parse_args()

    # 1. Configuration de la seed
    if args.seed is not None:
        print("Fixing random seed to:", args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
    else:
        print("No seed provided, using random seed.")

    # 2. Configuration
    config = ConfigAblation()
    if args.no_svd: config.use_svd = False
    if args.no_kmeans: config.use_kmeans = False
    if args.no_nmf: config.use_nmf = False
    if args.no_greedy: config.use_greedy = False
    
    config.crossover_type = args.crossover
    config.restart_mode = args.restart_mode
    if args.no_transpose: config.allow_transpose = False
    if args.debug_mode: config.debug_mode = True
    config.mutation_type = args.mutation_type
    config.factorization_mode = args.factorization_mode
    
    print(f"\n=== Lancement Evo-Discrete-MF ===")
    print(f"Input: {args.input}")
    print(f"Config Active: {config}\n")

    # 3. Lecture
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
        print(f"Erreur: Fichier {args.input} introuvable.")
        return

    # 4. Lancement Metaheuristique
    W, H, f_val = metaheuristic(
        X, args.rank, LW, UW, LH, UH, 
        TIME_LIMIT=args.time_limit, 
        N=args.nombre_population, 
        tournament_size=args.tournament_size, 
        mutation_rate=args.mutation_rate,
        config=config # On passe la config !
    )

    # 5. Vérification & Sauvegarde
    print("\n=== Vérification Finale ===")
    if args.factorization_mode == "BMF":
        f_check = fobj_bmf(X, W, H)
    elif args.factorization_mode == "RELU":
        f_check = fobj_relu(X, W, H)
    else:
        f_check = fobj(X, W, H)
    print(f"Fitness Solver: {f_val}")
    print(f"Fitness Check : {f_check}")
    
    if f_val != f_check:
        print("/!\\ ATTENTION: Divergence entre solver et vérification /!\\")

    if not solutionIsFeasible(W, H, args.rank, LW, UW, LH, UH):
        print("/!\\ ATTENTION: Solution invalide (hors bornes) /!\\")
    else:
        print("Solution valide (contraintes respectées).")

    write_txt_file(args.output, f_val, W, H)
    print(f"Solution écrite dans : {args.output}")

if __name__ == "__main__":
    main()