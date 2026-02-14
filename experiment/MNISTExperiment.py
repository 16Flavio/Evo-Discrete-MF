from scipy.io import loadmat
import math
import numpy as np
import random
import matplotlib.pyplot as plt #
import os
from datetime import datetime

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from src.solver import metaheuristic
except ImportError:
    print("Attention: Le module 'src.solver' n'a pas été trouvé. Assurez-vous que la structure des dossiers est correcte.")
    def metaheuristic(X, r, LW, UW, LH, UH, type_algo, TIME_LIMIT, N, debug_mode):
        M, N_dim = X.shape
        return np.random.rand(M, r), np.random.rand(r, N_dim), 0

def log(msg):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)

def save_factors(W, H, r, method, output_dir):
    """
    Sauvegarde les facteurs W et H dans des fichiers .txt séparés.
    """
    filename_W = os.path.join(output_dir, f"W_r{r}_{method}.txt")
    filename_H = os.path.join(output_dir, f"H_r{r}_{method}.txt")
    
    np.savetxt(filename_W, W, fmt='%.6e')
    np.savetxt(filename_H, H, fmt='%.6e')
    
    print(f"Facteurs sauvegardés : {filename_W} et {filename_H}")

def visualize_and_save(X, W, H, r, method, output_dir, indices, image_shape=(28, 28)):
    """
    Génère et sauvegarde une image comparative : Original vs Reconstruit.
    """
    if method == 'RELU':
        X_recon = np.maximum(0, W @ H)
    else:
        X_recon = W @ H

    num_images = len(indices)
    
    fig, axes = plt.subplots(2, num_images, figsize=(num_images * 2, 5))
    plt.suptitle(f"Reconstruction MNIST - Méthode : {method}, Rang : {r}")

    for i, idx in enumerate(indices):
        ax_orig = axes[0, i]
        img_orig = X[idx].reshape(image_shape)
        ax_orig.imshow(img_orig, cmap='gray')
        ax_orig.axis('off')
        if i == 0:
            ax_orig.set_title("Original")

        ax_recon = axes[1, i]
        img_recon = X_recon[idx].reshape(image_shape)
        ax_recon.imshow(img_recon, cmap='gray')
        ax_recon.axis('off')
        if i == 0:
            ax_recon.set_title(f"Reconstruit ({method})")

    save_path = os.path.join(output_dir, f"visualisation_r{r}_{method}.png")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    log(f"Visualisation sauvegardée dans : {save_path}")

def my_method_imf(X, r, alpha=-16, beta=15, budgettemps=2.0):
    LW = int(math.ceil(alpha))
    UW = int(math.floor(beta))
    LH = LW
    UH = UW
    
    W, H, f = metaheuristic(X, r, LW, UW, LH, UH, 'IMF', TIME_LIMIT=budgettemps, N=10, debug_mode=False)
    return W, H

def my_method_relu(X, r, alpha=-16, beta=15, budgettemps=2.0):
    LW = int(math.ceil(alpha))
    UW = int(math.floor(beta))
    LH = LW
    UH = UW
    
    W, H, f = metaheuristic(X, r, LW, UW, LH, UH, 'RELU', TIME_LIMIT=budgettemps, N=10, debug_mode=False)
    return W, H


output_folder = "experiment/result_experiment_RELU"
os.makedirs(output_folder, exist_ok=True)

d = loadmat("data/MNIST_numpy.mat")
X = d["X"]

a, b = -16, 15
NUM_MINUTES = 20
np.random.seed(42)
random.seed(42)

random_indices = np.random.choice(X.shape[0], 3, replace=False)
log(f"Indices sélectionnés pour visualisation : {random_indices}")

for r in [5, 10, 15, 20, 30, 40]:
    log(f"--- Traitement pour le rang r={r} ---")
    
    W_imf, H_imf = my_method_imf(X, r, alpha=a, beta=b, budgettemps=NUM_MINUTES*60)
    rel_err_imf = np.linalg.norm(X - W_imf @ H_imf, 'fro') / (np.linalg.norm(X, 'fro') + 1e-12)
    log(f"Relative error for r = {r} and IMF : {rel_err_imf:.2%}")
    
    save_factors(W_imf, H_imf, r, "IMF", output_folder)
    visualize_and_save(X, W_imf, H_imf, r, "IMF", output_folder, random_indices)

    W_relu, H_relu = my_method_relu(X, r, alpha=a, beta=b, budgettemps=NUM_MINUTES*60)
    rel_err_relu = np.linalg.norm(X - np.maximum(0, W_relu @ H_relu), 'fro') / (np.linalg.norm(X, 'fro') + 1e-12)
    log(f"Relative error for r = {r} and ReLu : {rel_err_relu:.2%}")
    
    save_factors(W_relu, H_relu, r, "RELU", output_folder)
    visualize_and_save(X, W_relu, H_relu, r, "RELU", output_folder, random_indices)

log("Terminé. Vérifiez le dossier 'resultats_mnist'.")