import numpy as np
import os

def read_txt_file_bmf(filename: str):
    """
    Lit un fichier d'instance de matrice selon le format du projet.

    Args:
        filename (str): Le chemin absolu vers le fichier .txt.

    Returns:
        - X (np.ndarray): La matrice X (m x n) en entiers.
    """
    print(f"  -> Lecture de l'instance : {filename}")
    if not os.path.exists(filename):
        print(f"  -> ERREUR: Fichier non trouvé.")
        return None

    with open(filename, 'r') as f:
        # 1. Lire la première ligne (m, n)
        try:
            premiere_ligne = f.readline().split()
            m, n = map(int, premiere_ligne)
        except Exception as e:
            print(f"  -> ERREUR: Impossible de lire la première ligne. {e}")
            return None
            
        print(f"     Params: m={m}, n={n}")

        # 2. Lire les 'm' lignes suivantes directement dans un array NumPy
        try:
            X = np.loadtxt(f, dtype=int, max_rows=m)
        except Exception as e:
            print(f"  -> ERREUR: Impossible de lire la matrice X. {e}")
            return None

        # 3. Vérifier la taille pour être sûr
        if X.shape != (m, n):
            print(f"  -> ERREUR: La matrice lue a une taille {X.shape}, attendue ({m}, {n})")
            return None
            
        print(f"     Matrice X de taille {X.shape} lue avec succès.")
        return X

def read_txt_file(filename: str):
    """
    Lit un fichier d'instance de matrice selon le format du projet.

    Args:
        filename (str): Le chemin absolu vers le fichier .txt.

    Returns:
        tuple: (X, LW, UW, LH, UH)
            - X (np.ndarray): La matrice X (m x n) en entiers.
            - LW, UW (int): Bornes pour W.
            - LH, UH (int): Bornes pour H.
    """
    print(f"  -> Lecture de l'instance : {filename}")
    if not os.path.exists(filename):
        print(f"  -> ERREUR: Fichier non trouvé.")
        return None, 0, 0, 0, 0

    with open(filename, 'r') as f:
        # 1. Lire la première ligne (m, n, LW, UW, LH, UH)
        try:
            premiere_ligne = f.readline().split()
            m, n, LW, UW, LH, UH = map(int, premiere_ligne)
        except Exception as e:
            print(f"  -> ERREUR: Impossible de lire la première ligne. {e}")
            return None, 0, 0, 0, 0
            
        print(f"     Params: m={m}, n={n}, LW={LW}, UW={UW}, LH={LH}, UH={UH}")

        # 2. Lire les 'm' lignes suivantes directement dans un array NumPy
        try:
            X = np.loadtxt(f, dtype=int, max_rows=m)
        except Exception as e:
            print(f"  -> ERREUR: Impossible de lire la matrice X. {e}")
            return None, 0, 0, 0, 0

        # 3. Vérifier la taille pour être sûr
        if X.shape != (m, n):
            print(f"  -> ERREUR: La matrice lue a une taille {X.shape}, attendue ({m}, {n})")
            return None, 0, 0, 0, 0
            
        print(f"     Matrice X de taille {X.shape} lue avec succès.")
        return X, LW, UW, LH, UH
    
def write_txt_file(filename: str, f_obj: int, W: np.ndarray, H: np.ndarray):
    """
    Écrit la solution (objectif, W, H) dans un fichier au format requis,
    UNIQUEMENT si la nouvelle solution est MEILLEURE (score plus petit) que celle existante.
    
    Le format est :
    Ligne 1: Valeur de l'objectif (arrondie à l'entier)
    Lignes suivantes: Matrice W (m lignes, r entiers par ligne)
    Lignes suivantes: Matrice H (r lignes, n entiers par ligne)
    
    Args:
        filename (str): Chemin complet du fichier à créer.
        f_obj (float or int): Valeur finale de la fonction objectif.
        W (np.ndarray): Matrice W (m x r).
        H (np.ndarray): Matrice H (r x n).
    """
    current_obj = int(np.round(f_obj))

    # --- VÉRIFICATION DE L'AMÉLIORATION ---
    if os.path.exists(filename):
        try:
            with open(filename, 'r') as f:
                first_line = f.readline().strip()
                if first_line:
                    old_obj = int(float(first_line)) # float puis int pour gérer d'éventuels formats "123.0"
                    
                    if current_obj >= old_obj:
                        print(f"  -> Solution non sauvegardée : Nouveau score {current_obj} >= Ancien score {old_obj}")
                        return # On arrête tout, on n'écrit pas
                    else:
                        print(f"  -> Amélioration trouvée ! ({old_obj} -> {current_obj})")
        except Exception as e:
            print(f"  -> Attention : Impossible de lire le fichier existant pour comparaison ({e}). On écrase par précaution.")

    # --- ÉCRITURE DU FICHIER ---
    print(f"  -> Écriture de la solution dans : {filename}")
    try:
        # S'assurer que le dossier de sortie existe
        os.makedirs(os.path.dirname(filename), exist_ok=True)
            
        with open(filename, 'w') as f:
            # 1. Écrire la valeur de l'objectif 
            f.write(f"{current_obj}\n")
            
            # 2. Écrire W
            for i in range(W.shape[0]):
                row_str = " ".join(map(str, W[i]))
                f.write(row_str + "\n")
                
            # 3. Écrire H
            for i in range(H.shape[0]):
                row_str = " ".join(map(str, H[i]))
                f.write(row_str + "\n")
                
    except Exception as e:
        print(f"  -> ERREUR: Impossible d'écrire le fichier de sortie. {e}")