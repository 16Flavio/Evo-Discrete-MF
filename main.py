import os, sys
import argparse
import numpy as np
import time
from utils.verification import fobj, solutionIsFeasible

def main():
    
    # ============ On se place a la racine du projet, plus simple pour acceder a tous les fichiers ============
    current_file_path = os.path.abspath(__file__)

    main_dir = os.path.dirname(current_file_path)

    src_dir = os.path.dirname(main_dir)

    PROJECT_ROOT = os.path.dirname(src_dir)

    sys.path.append(PROJECT_ROOT)
    # =========================================================================================================

    # ============ On recupere le time limit + le chemin vers le fichier + on demande ou placer la sortie =====
    parser = argparse.ArgumentParser(
        description="Script principal pour l'optimisation de matrices (Projet Graph)."
    )
    parser.add_argument(
        "-i", "--input",
        help="Chemin relatif (depuis la racine du projet) du fichier matrice en entrée.",
        required=True  # Cet argument est obligatoire
    )
    
    # Argument 2: Fichier OUTPUT
    parser.add_argument(
        "-o", "--output",
        help="Chemin relatif (depuis la racine du projet) du fichier de solution en sortie.",
        required=True  # Cet argument est obligatoire
    )
    
    # Argument 3: Time Limit
    parser.add_argument(
        "-t", "--time_limit",
        help="Temps limite pour l'algorithme, en secondes. (Défaut: 300.0)",
        type=float,       # Le type est un float
        default=300.0      # Valeur par défaut si non fourni
    )

    # Argument 4: Nombre dans la population
    parser.add_argument(
        "-n", "--nombre_population",
        help="Nombre de solution pour la population de l'algorithme. (Défaut: 50)",
        type=int,       # Le type est un int
        default=50      # Valeur par défaut si non fourni
    )

    # Argument 5: Taille du tournoi
    parser.add_argument(
        "-s", "--tournament_size",
        help="Nombre de solution pour le tournoi de l'algorithme. (Défaut: 3)",
        type=int,       # Le type est un int
        default=3      # Valeur par défaut si non fourni
    )

    # Argument 6: Nombre dans la population
    parser.add_argument(
        "-m", "--mutation_rate",
        help="Taux de mutation pour la population de l'algorithme. (Défaut: 0.1)",
        type=float,       # Le type est un int
        default=0.1      # Valeur par défaut si non fourni
    )
    args = parser.parse_args()

    mutation_rate = max(0.0, min(1.0,args.mutation_rate))
    tournament_size = args.tournament_size
    N = args.nombre_population
    TIME_LIMIT = args.time_limit
    INPUT_PATH = os.path.join(PROJECT_ROOT, args.input)
    OUTPUT_PATH = os.path.join(PROJECT_ROOT, args.output)
    # =========================================================================================================

    # ============ Lecture du fichier =========================================================================
    from utils.file_utils import read_txt_file
    X, r, LW, UW, LH, UH = read_txt_file(INPUT_PATH)
    # =========================================================================================================

    # =============================== Initialisation ==========================================================
    '''
    A faire
    '''
    # =========================================================================================================

    # =============================== Lancement de l'algorithme ===============================================
    '''
    A faire
    
    f_obj=41
    W = np.array([[2,2],[2,1],[2,2],[2,1],[2,2]])
    H = np.array([[1,2,2,0,2],[2,1,0,1,2]])
    '''
    from src.solver import metaheuristic
    W, H, f_obj = metaheuristic(X, r, LW, UW, LH, UH, TIME_LIMIT=TIME_LIMIT, N=N, tournament_size=tournament_size, mutation_rate=mutation_rate)
    # =========================================================================================================

    # =================================== Verification de la solution =========================================
    f_obj_verif = fobj(X, W, H)
    if f_obj != f_obj_verif:
        raise ValueError("La fonction objectif donnee par le solver ne correspond pas aux matrices associees !!!")
    if not solutionIsFeasible(W, H, r, LW, UW, LH, UH):
        print(W)
        print(H)
        raise ValueError("La solution ne respecte pas les contraintes !!!")
    # =========================================================================================================

    # ============ Ecriture de la solution dans le format txt requis ==========================================
    from utils.file_utils import write_txt_file
    write_txt_file(OUTPUT_PATH, f_obj, W, H)
    # =========================================================================================================

if __name__ == "__main__":
    main()