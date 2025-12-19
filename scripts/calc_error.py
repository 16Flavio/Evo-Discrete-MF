import os
import sys
import numpy as np

# --- 1. CONFIGURATION DES CHEMINS ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.append(PROJECT_ROOT)

try:
    from utils.file_utils import read_txt_file_bmf
except ImportError:
    print("ERREUR: Impossible d'importer 'utils'. Vérifiez que vous lancez le script depuis le bon endroit.")
    sys.exit(1)

# --- 2. PARAMÈTRES (À modifier si besoin) ---
INPUT_MATRIX = os.path.join(PROJECT_ROOT, "data", "bmf_matrix", "binarizedCBCL.txt")

# Chemin vers le fichier de résultat (W, H, Score)
RESULT_FILE = os.path.join(PROJECT_ROOT, "results", "bmf_matrix", "binarizedCBCL_r10_5min.txt") 

def main():
    print("=== CALCUL DE L'ERREUR RELATIVE ===")
    print(f"Matrice source : {os.path.basename(INPUT_MATRIX)}")
    print(f"Fichier résultat : {os.path.basename(RESULT_FILE)}")
    print("-" * 40)

    # 1. Lecture de la matrice originale X
    if not os.path.exists(INPUT_MATRIX):
        print(f"ERREUR: Fichier matrice introuvable : {INPUT_MATRIX}")
        return

    X = read_txt_file_bmf(INPUT_MATRIX)
    
    if X is None:
        print("Erreur lors de la lecture de la matrice.")
        return

    # Calcul de la norme de X (Somme des 1 pour le binaire, ou somme des valeurs)
    norm_x = np.sum(X)
    print(f"Norme de X (Total des éléments) : {norm_x}")

    # 2. Lecture du score (Erreur Absolue) dans le fichier résultat
    if not os.path.exists(RESULT_FILE):
        print(f"ERREUR: Fichier résultat introuvable : {RESULT_FILE}")
        return

    try:
        with open(RESULT_FILE, 'r') as f:
            line = f.readline().strip()
            absolute_error = float(line)
    except Exception as e:
        print(f"ERREUR: Impossible de lire le score dans le fichier résultat ({e})")
        return

    print(f"Erreur Absolue (Fitness)       : {absolute_error}")

    # 3. Calcul de l'Erreur Relative
    if norm_x == 0:
        print("Attention: La matrice X est vide (norme 0), division impossible.")
    else:
        relative_error = absolute_error / norm_x
        print("-" * 40)
        print(f"ERREUR RELATIVE                : {relative_error:.6f}")
        print(f"Soit en pourcentage            : {relative_error * 100:.4f} %")

if __name__ == "__main__":
    main()