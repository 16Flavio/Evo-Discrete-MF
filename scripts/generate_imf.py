import argparse
import random
import os
import sys

def generate_matrix(rows, cols, min_val, max_val, density):
    """Génère une matrice aléatoire (liste de listes)."""
    matrix = []
    for _ in range(rows):
        row_data = []
        for _ in range(cols):
            if random.random() > density:
                val = 0
            else:
                val = random.randint(min_val, max_val)
            row_data.append(val)
        matrix.append(row_data)
    return matrix

def matrix_multiply(A, B):
    """Multiplication matricielle simple pour des entiers."""
    rows_A = len(A)
    cols_A = len(A[0])
    rows_B = len(B)
    cols_B = len(B[0])
    
    if cols_A != rows_B:
        raise ValueError("Dimensions incompatibles pour multiplication")

    result = [[0 for _ in range(cols_B)] for _ in range(rows_A)]
    
    for i in range(rows_A):
        for j in range(cols_B):
            sum_val = 0
            for k in range(cols_A):
                sum_val += A[i][k] * B[k][j]
            result[i][j] = sum_val
    return result

def write_imf_file(matrix, output_file):
    rows = len(matrix)
    cols = len(matrix[0]) if rows > 0 else 0
    
    with open(output_file, 'w') as f:
        f.write(f"{rows} {cols}\n")
        for row in matrix:
            f.write(" ".join(map(str, row)) + "\n")

def main():
    parser = argparse.ArgumentParser(description="Générateur d'instances IMF (Full Rank ou Low Rank)")
    
    parser.add_argument("--rows", type=int, default=100, help="Nombre de lignes")
    parser.add_argument("--cols", type=int, default=100, help="Nombre de colonnes")
    parser.add_argument("--min", type=int, default=-2, help="Min valeur (pour X ou pour les facteurs W/H)")
    parser.add_argument("--max", type=int, default=2, help="Max valeur (pour X ou pour les facteurs W/H)")
    parser.add_argument("--density", type=float, default=0.8, help="Densité (0.0 à 1.0)")
    parser.add_argument("--force_rank", type=int, default=0, help="Si > 0, génère X = W*H avec ce rang (X sera Low-Rank). Si 0, X est purement aléatoire (Full-Rank).")
    parser.add_argument("--name", type=str, default="instance.txt", help="Nom du fichier de sortie")
    parser.add_argument("--folder", type=str, default="../data/imf_matrix", help="Dossier de destination")

    args = parser.parse_args()

    # Gestion des chemins
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, args.folder)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, args.name)

    print(f"{'='*40}")
    print(f"Génération Instance IMF")
    
    if args.force_rank > 0:
        print(f"Mode : SYNTHÉTIQUE (Low-Rank)")
        print(f"Objectif : Créer X de rang {args.force_rank} exactement.")
        print(f"Méthode : Génération de W ({args.rows}x{args.force_rank}) et H ({args.force_rank}x{args.cols}) puis X = W * H")
        print(f"Note : Les valeurs dans X peuvent dépasser [{args.min}, {args.max}] suite à la multiplication.")
        
        # On génère les facteurs W et H
        # Note: On applique la densité sur les facteurs pour créer de la sparsité structurelle
        W = generate_matrix(args.rows, args.force_rank, args.min, args.max, args.density)
        H = generate_matrix(args.force_rank, args.cols, args.min, args.max, args.density)
        
        # On calcule X
        X = matrix_multiply(W, H)
        
    else:
        print(f"Mode : ALÉATOIRE (Full-Rank)")
        print(f"Objectif : Créer X aléatoire dans [{args.min}, {args.max}]")
        print(f"Note : Il sera impossible d'obtenir une erreur 0 avec un rang faible.")
        
        X = generate_matrix(args.rows, args.cols, args.min, args.max, args.density)

    # Sauvegarde
    write_imf_file(X, output_path)
    print(f"✅ Fichier généré : {output_path}")
    print(f"{'='*40}")

if __name__ == "__main__":
    main()