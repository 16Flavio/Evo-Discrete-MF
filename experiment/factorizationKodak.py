import imageio.v3 as iio
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import math
from datetime import datetime
import random

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.solver import metaheuristic

def log(msg):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)

def my_method_imf(X, r, alpha=-16, beta=15, budgettemps=2.0):
    """
    X: np.ndarray float64, shape (M,N)
    r: int
    bounds: (lo,hi) clamp
    tmax: budget temps en secondes

    Doit retourner:
        U_int : (M,r) entier
        V_int : (N,r) entier
    """
    LW = int(math.ceil(alpha))
    UW = int(math.floor(beta))
    LH = LW
    UH = UW
    
    W, H, f = metaheuristic(X, r, LW, UW, LH, UH, 'IMF', TIME_LIMIT=budgettemps, N=20, debug_mode=False)
    
    return W, H

def method_from_quantization_aware(X, r, choixinit="rand", alpha=-16, beta=15):
    X       = np.asarray(X, dtype=np.float64)
    m, n    = X.shape

    maxiter = 1000
    eps     = 1e-16

    def project(Z):
        return np.clip(np.rint(Z), math.ceil(alpha), math.floor(beta))

    def soft_thresholding(Z, lam):
        if lam == 0: return Z
        return np.sign(Z) * np.maximum(np.abs(Z) - lam, 0.0)

    # Initialisation
    if choixinit.lower() == "svd":
        Ufull, s, Vtfull = np.linalg.svd(X, full_matrices=False)
        U_tilde = Ufull[:, :r]
        s       = s[:r]
        V_tilde = Vtfull[:r, :].T
        s_sqrt  = np.sqrt(s)

        U = U_tilde * s_sqrt[None, :]
        V = V_tilde * s_sqrt[None, :]
        
        U = project(U)
        V = project(V)

    if choixinit.lower() == "rand":
        U = np.random.randint(low=alpha,high=beta+1, size=(m,r)).astype(np.float64) #np.rng.integers(alpha, beta + 1, size=(m, r)).astype(np.float64)
        V = np.random.randint(low=alpha,high=beta+1, size=(n,r)).astype(np.float64) #np.rng.integers(alpha, beta + 1, size=(n, r)).astype(np.float64)
        
    # BCD
    for it in range(maxiter):
        # Update U par les r colonnes
        A = X @ V 
        B = V.T @ V

        U_new = U.copy()

        for k in range(r):
            idx   = [j for j in range(r) if j != k]
            term1 = A[:, k:k+1]
            term2 = U_new[:, idx] @ B[np.ix_(idx, [k])]
            num   = soft_thresholding(term1 - term2, 0)
            den   = B[k, k]
            U_new[:, k:k+1] = project((num + eps) / (den + eps))
        diffU = np.sum(abs(U-U_new))
        U = U_new

        # Update V par les r rangÃ©es
        A = X.T @ U
        B = U.T @ U

        V_new = V.copy()
        for k in range(r):
            idx   = [j for j in range(r) if j != k]
            term1 = A[:, k:k+1]
            term2 = V_new[:, idx] @ B[np.ix_(idx, [k])]
            num   = soft_thresholding(term1 - term2,0)
            den   = B[k, k]
            V_new[:, k:k+1] = project((num + eps) / (den + eps))
        diffV = np.sum(abs(V-V_new))
        V = V_new
        
        if diffU==0 and diffV==0:
            # print(it)
            break

    return U, V

def rel_error(X, Xhat, eps=1e-12):
    return np.linalg.norm(X - Xhat, "fro")/ (np.linalg.norm(X,"fro") + eps)

if __name__ == "__main__":

    input_dir = "data/kodak"
    output_dir = "experiment/result_experiment_kodak"
    os.makedirs(output_dir, exist_ok=True)

    np.random.seed(42)
    random.seed(42)

    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif')
    image_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(valid_extensions)])

    if not image_files:
        log(f"Aucune image trouvée dans {input_dir}")
        exit()

    log(f"Début du traitement de {len(image_files)} images...")

    NUM_MINUTES = 5
    r = 40
    a, b = 0, 15

    for filename in image_files:
        filepath = os.path.join(input_dir, filename)
        log(f"--- Traitement de l'image : {filename} ---")

        try:
            img = iio.imread(filepath)
        except Exception as e:
            log(f"Erreur lors de la lecture de {filename}: {e}")
            continue

        if len(img.shape) == 2:
            img = np.stack((img,)*3, axis=-1)

        R = img[:, :, 0]
        G = img[:, :, 1]
        B = img[:, :, 2]

        import time

        WR, HR = method_from_quantization_aware(R, r, choixinit="svd", alpha=a, beta=b)
        best_err = np.linalg.norm(R-WR@HR.T)/(np.linalg.norm(R)+1e-12)
        log(f"Erreur relative R pour QMF : {(best_err)*100:.2f} %")
        t0 = time.time()
        
        while time.time() - t0 < NUM_MINUTES*60:
            W_tempR, H_tempR = method_from_quantization_aware(R, r, choixinit="rand", alpha=a, beta=b)
            err = np.linalg.norm(R-W_tempR@H_tempR.T)/(np.linalg.norm(R)+1e-12)
            if err < best_err:
                WR, HR = W_tempR, H_tempR
                best_err = err
                log(f"Erreur relative R pour QMF : {(np.linalg.norm(R-WR@HR.T)/(np.linalg.norm(R)+1e-12))*100:.2f} %")
        
        WG, HG = method_from_quantization_aware(G, r, choixinit="svd", alpha=a, beta=b)
        best_err = np.linalg.norm(G-WG@HG.T)/(np.linalg.norm(G)+1e-12)
        log(f"Erreur relative G pour QMF : {(best_err)*100:.2f} %")
        t0 = time.time()

        while time.time() - t0 < NUM_MINUTES*60: 
            W_tempG, H_tempG = method_from_quantization_aware(G, r, choixinit="rand", alpha=a, beta=b)
            err = np.linalg.norm(G-W_tempG@H_tempG.T)/(np.linalg.norm(G)+1e-12)
            if err < best_err:
                WG, HG = W_tempG, H_tempG
                best_err = err
                log(f"Erreur relative G pour QMF : {(np.linalg.norm(G-WG@HG.T)/(np.linalg.norm(G)+1e-12))*100:.2f} %")

        WB, HB = method_from_quantization_aware(B, r, choixinit="svd", alpha=a, beta=b)
        best_err = np.linalg.norm(B-WB@HB.T)/(np.linalg.norm(B)+1e-12)
        log(f"Erreur relative B pour QMF : {(best_err)*100:.2f} %")
        t0 = time.time()
        
        while time.time() - t0 < NUM_MINUTES*60:
            W_tempB, H_tempB = method_from_quantization_aware(B, r, choixinit="rand", alpha=a, beta=b)
            err = np.linalg.norm(B-W_tempB@H_tempB.T)/(np.linalg.norm(B)+1e-12)
            if err < best_err:
                WB, HB = W_tempB, H_tempB
                best_err = err
                log(f"Erreur relative B pour QMF : {(np.linalg.norm(B-WB@HB.T)/(np.linalg.norm(B)+1e-12))*100:.2f} %")

        WR_EvoIMF, HR_EvoIMF = my_method_imf(R, r, alpha=a, beta=b, budgettemps=NUM_MINUTES*60)
        log(f"Erreur relative R pour EvoIMF : {(np.linalg.norm(R-WR_EvoIMF@HR_EvoIMF)/(np.linalg.norm(R)+1e-12))*100:.2f} %")
        WG_EvoIMF, HG_EvoIMF = my_method_imf(G, r, alpha=a, beta=b, budgettemps=NUM_MINUTES*60)
        log(f"Erreur relative G pour EvoIMF : {(np.linalg.norm(G-WG_EvoIMF@HG_EvoIMF)/(np.linalg.norm(G)+1e-12))*100:.2f} %")
        WB_EvoIMF, HB_EvoIMF = my_method_imf(B, r, alpha=a, beta=b, budgettemps=NUM_MINUTES*60)
        log(f"Erreur relative B pour EvoIMF : {(np.linalg.norm(B-WB_EvoIMF@HB_EvoIMF)/(np.linalg.norm(B)+1e-12))*100:.2f} %")

        errR_QMF = np.linalg.norm(R-WR@HR.T)/(np.linalg.norm(R)+1e-12)
        errG_QMF = np.linalg.norm(G-WG@HG.T)/(np.linalg.norm(G)+1e-12)
        errB_QMF = np.linalg.norm(B-WB@HB.T)/(np.linalg.norm(B)+1e-12)
        mean_error_QMF = (errR_QMF + errG_QMF + errB_QMF)/3

        errR_EvoMF = np.linalg.norm(R-WR_EvoIMF@HR_EvoIMF)/(np.linalg.norm(R)+1e-12)
        errG_EvoMF = np.linalg.norm(G-WG_EvoIMF@HG_EvoIMF)/(np.linalg.norm(G)+1e-12)
        errB_EvoMF = np.linalg.norm(B-WB_EvoIMF@HB_EvoIMF)/(np.linalg.norm(B)+1e-12)
        mean_error_EvoMF = (errR_EvoMF + errG_EvoMF + errB_EvoMF)/3

        # Reconstruction + clamp + conversion uint8
        R_rec = np.clip(WR @ HR.T, 0, 255)
        G_rec = np.clip(WG @ HG.T, 0, 255)
        B_rec = np.clip(WB @ HB.T, 0, 255)

        R_rec_EvoIMF = np.clip(WR_EvoIMF@HR_EvoIMF, 0, 255)
        G_rec_EvoIMF = np.clip(WG_EvoIMF@HG_EvoIMF, 0, 255)
        B_rec_EvoIMF = np.clip(WB_EvoIMF@HB_EvoIMF, 0, 255)

        R8 = np.rint(R_rec).astype(np.uint8)
        G8 = np.rint(G_rec).astype(np.uint8)
        B8 = np.rint(B_rec).astype(np.uint8)

        R8_EvoIMF = np.rint(R_rec_EvoIMF).astype(np.uint8)
        G8_EvoIMF = np.rint(G_rec_EvoIMF).astype(np.uint8)
        B8_EvoIMF = np.rint(B_rec_EvoIMF).astype(np.uint8)

        img = np.stack([R8, G8, B8], axis=2)

        img_EvoIMF = np.stack([R8_EvoIMF, G8_EvoIMF, B8_EvoIMF], axis = 2)

        base_name = os.path.splitext(filename)[0]

        qmf_filename = f"{base_name}_QMF_err_{mean_error_QMF*100:.2f}.png"
        evo_filename = f"{base_name}_EvoIMF_err_{mean_error_EvoMF*100:.2f}.png"
        
        iio.imwrite(os.path.join(output_dir, qmf_filename), img)
        iio.imwrite(os.path.join(output_dir, evo_filename), img_EvoIMF)

        log(f"Sauvegardé : {qmf_filename} et {evo_filename}")
        log("-" * 30)

    log(f"Tout les traitements sont terminés. Résultats dans {output_dir}")