import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import argparse
import math

def read_solution_file(filepath, m_expected):
    """
    Lit le fichier de sortie pour extraire W et H.
    Gère le cas où W et H n'ont pas le même nombre de colonnes.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Fichier introuvable : {filepath}")

    with open(filepath, 'r') as f:
        # 1. Lire le score (première ligne)
        try:
            line = f.readline().strip()
            if not line: return 0, None, None
            score = float(line)
        except ValueError:
            print("Erreur: Impossible de lire le score sur la première ligne.")
            return 0, None, None
        
        # 2. Lire le reste ligne par ligne dans des listes Python simples
        raw_data = []
        for line in f:
            parts = list(map(float, line.split()))
            if len(parts) > 0:
                raw_data.append(parts)
        
    # --- C'est ici que la correction opère ---
    # On sépare les données AVANT de créer les tableaux NumPy
    
    if m_expected is None:
        print("ERREUR CRITIQUE: Impossible de séparer W et H sans connaitre 'm'.")
        print("Veuillez fournir le fichier d'entrée original avec --input.")
        sys.exit(1)

    if len(raw_data) <= m_expected:
        print(f"Erreur: Le fichier semble incomplet. Lignes lues={len(raw_data)}, attendues (W)={m_expected}")
        sys.exit(1)

    # Découpage des listes
    list_W = raw_data[:m_expected]
    list_H = raw_data[m_expected:]

    try:
        W = np.array(list_W)
        H = np.array(list_H)
    except Exception as e:
        print(f"Erreur lors de la conversion NumPy : {e}")
        print("Vérifiez que le fichier résultat n'est pas corrompu.")
        sys.exit(1)

    return score, W, H

def get_matrix_dims(input_file):
    """Lit l'en-tête du fichier d'entrée pour avoir m."""
    if not os.path.exists(input_file):
        print(f"Erreur: Fichier input introuvable : {input_file}")
        sys.exit(1)
    with open(input_file, 'r') as f:
        line = f.readline().split()
        m = int(line[0])
    return m

def plot_W_basis(W, score_val, img_h=None, img_w=None, title="Basis Images"):
    """
    Affiche les colonnes de W sous forme d'images.
    """
    m, r = W.shape
    
    # 1. Déduction des dimensions
    if img_h is None or img_w is None:
        side = math.sqrt(m)
        if side.is_integer():
            img_h = int(side)
            img_w = int(side)
        else:
            print(f"ERREUR: m={m} n'est pas un carré parfait (sqrt={side:.2f}).")
            print("Veuillez forcer les dimensions avec --height et --width.")
            return

    print(f"Affichage : {r} images de taille {img_h}x{img_w}")

    # 2. Grille d'affichage
    cols = int(math.ceil(math.sqrt(r)))
    rows = int(math.ceil(r / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2.5))
    if isinstance(axes, np.ndarray):
        axes_flat = axes.flatten()
    else:
        axes_flat = [axes] # Cas où r=1

    fig.suptitle(f"{title}\nFitness: {score_val}", fontsize=14)

    for i in range(r):
        ax = axes_flat[i]
        col_vector = W[:, i]
        
        try:
            img = col_vector.reshape((img_h, img_w))
            img = np.rot90(img, k=-1)
            # 'gray' = noir et blanc classique
            # 'binary' = inversé (utile si 0=blanc, 1=noir)
            ax.imshow(img, cmap='binary', interpolation='nearest')
            ax.set_title(f"Basis {i+1}")
            ax.axis('off')
        except ValueError:
            ax.text(0.5, 0.5, "Error", ha='center')

    # Masquer les cases vides
    for i in range(r, len(axes_flat)):
        axes_flat[i].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--result", required=True, help="Fichier résultat (W, H).")
    parser.add_argument("-i", "--input", required=True, help="Fichier matrice originale (pour lire m).")
    parser.add_argument("--height", type=int, help="Hauteur image.")
    parser.add_argument("--width", type=int, help="Largeur image.")
    
    args = parser.parse_args()

    # 1. Récupérer m
    m = get_matrix_dims(args.input)
    print(f"Dimension détectée : m={m}")
    
    # 2. Lire W correctement
    score_val, W, H = read_solution_file(args.result, m_expected=m)
    
    # 3. Plot
    plot_W_basis(W, score_val, img_h=args.height, img_w=args.width, 
                 title=f"Columns of W ({os.path.basename(args.result)})")