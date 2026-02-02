import numpy as np
import time
from sklearn.decomposition import TruncatedSVD
from .initialisationKMEANS import integer_kmeans
from .initialisationNMF import ExtendedNMF, round_matrix

# A implementer ALS entre les bornes puis arrondi à la fin et du multi start (30%), puis bruit (30%) et puis random (20%) et (20%) de colonne

# --- 1. SVD SMART (Classique) ---
def initialize_SVD_smart(X, r, LW, UW):
    """
    Initializes W using Truncated SVD.
    It clips and scales the results to fit within the bounds [LW, UW].
    """
    svd = TruncatedSVD(n_components=r, random_state=None) 
    W_float = svd.fit_transform(X)
    lower_p, upper_p = np.percentile(W_float, [1, 99])
    W_clipped = np.clip(W_float, lower_p, upper_p)
    w_min, w_max = W_clipped.min(), W_clipped.max()
    if w_max - w_min < 1e-9: scale = 1
    else: scale = (UW - LW) / (w_max - w_min)
    W_scaled = (W_clipped - w_min) * scale + LW
    return np.clip(np.round(W_scaled), LW, UW).astype(int)

# --- 2. RESIDUAL GREEDY INIT ---
def initialize_greedy_residual(X, r, LW, UW, LH, UH):
    """
    Initializes W using a greedy residual approach.
    Iteratively selects the column of the residual matrix with the largest norm.
    """
    m, n = X.shape
    W = np.zeros((m, r))
    R = X.astype(float) 
    
    for k in range(r):
        norms = np.linalg.norm(R, axis=0)
        idx_max = np.argmax(norms)
        w_k_raw = R[:, idx_max]
        w_min, w_max = w_k_raw.min(), w_k_raw.max()
        
        if np.abs(w_max - w_min) > 1e-6:
             scale = UW / w_max if w_max > 0 else LW / w_min
             w_k = w_k_raw * scale
        else:
             w_k = w_k_raw

        w_k = np.clip(np.round(w_k), LW, UW)
        W[:, k] = w_k
        norm_w = np.sum(w_k**2)
        if norm_w > 1e-9:
            h_k = np.dot(w_k, R) / norm_w
            h_k = np.clip(np.round(h_k), LH, UH)
            R -= np.outer(w_k, h_k)
            
    return W.astype(int)

# --- 3. COLUMN SAMPLING --- Essayer d'enlever les probas pondérés 
def initialize_column_sampling(X, r, LW, UW):
    """
    Initializes W by sampling columns from X based on their norms.
    Columns with larger norms have a higher probability of being selected.
    """
    m, n = X.shape
    col_norms = np.linalg.norm(X, axis=0)
    probs = col_norms / np.sum(col_norms)
    indices = np.random.choice(n, r, replace=False, p=probs)
    
    W = X[:, indices].copy()
    W = np.clip(W, LW, UW) 
    return W.astype(int)

# --- MUTATIONS & PERTURBATIONS ---

def perturb_W(W, mutation_rate, LW, UW):
    """
    Perturbs W by changing a percentage of its entries to random values.
    """
    m, r = W.shape
    W_new = W.copy()
    
    if np.random.rand() < 0.5:
        # Reset Colonne
        col = np.random.randint(0, r)
        W_new[:, col] = LW
    else:
        # Reset Bloc
        h_block = m // 4
        if h_block < 1: h_block = 1
        start_row = np.random.randint(0, m - h_block + 1)
        W_new[start_row : start_row + h_block, :] = LW
        
    return W_new

def initialize_boolean_greedy_cover(X, r, LW=0, UW=1):
    """
    Initialisation spécifique BMF : Sélectionne itérativement les colonnes de X
    qui couvrent le plus de 1 non encore couverts.
    """
    m, n = X.shape
    W = np.zeros((m, r), dtype=int)
    X_remaining = X.copy().astype(int)
    
    # On cherche r colonnes
    for k in range(r):
        # Calculer le "score" de chaque colonne de X (combien de 1 elle couvre dans le résidu)
        # Note: on pourrait optimiser en ne recalculant pas tout, mais pour l'init c'est acceptable
        col_scores = np.sum(X_remaining, axis=0)
        
        best_col_idx = np.argmax(col_scores)
        
        # Si plus rien à couvrir, on remplit aléatoirement
        if col_scores[best_col_idx] == 0:
            W[:, k] = np.random.randint(LW, UW + 1, size=m)
        else:
            W[:, k] = X[:, best_col_idx]
            
            # Mise à jour du résidu : on retire les 1 couverts par cette colonne
            # Logique : X_rem = X_rem AND (NOT Col_k)
            # En arithmétique : X_remaining[i] passe à 0 si W[i,k] est 1
            mask = (W[:, k] == 1)
            X_remaining[mask, :] = 0 # Astuce: ici on simplifie, en vrai on devrait update col par col, 
                                     # mais pour une init "Basis Vector" c'est souvent suffisant de retirer les lignes couvertes
                                     # Ou plus précisément : X_remaining[mask] = 0 est trop agressif pour H.
                                     # Mieux : On met à 0 les positions (i, best_col_idx) couvertes ? 
                                     # Simplification pour BMF Init : On soustrait la colonne choisie du calcul des scores futurs
            
            # Vrai Greedy Cover simplifié : On annule les lignes couvertes pour forcer la diversité
            X_remaining[mask, :] = 0 

    return W.astype(int)

def perturb_W_destructive(W, LW, UW):
    """
    Destructive perturbation: Resets a significant portion of W to random values.
    """
    m, r = W.shape
    W_new = W.copy()
    
    # Reset 25% to 50% of columns
    n_reset = np.random.randint(r // 4, r // 2 + 1)
    if n_reset < 1: n_reset = 1
    
    cols = np.random.choice(r, n_reset, replace=False)
    for c in cols:
        W_new[:, c] = np.random.randint(LW, UW + 1, size=m)
        
    return W_new

def generate_antithetic_W(W, LW, UW):
    """
    Generates the 'antithetic' (inverted) matrix. Useful for Alien Restart.
    """
    return (UW + LW) - W

def optimizeH(X, WtW, WtX, H, LH, UH):
    """
    Optimizes H given fixed W using a simple least squares approach.
    Rounds and clips the results to integer bounds.
    """
    m, n = X.shape
    r = H.shape[1]
    
    for k in range(H.shape[0]):
        hk = (WtX[k, :] - (WtW[k, :] @ H) + WtW[k, k] * H[k, :]) / (WtW[k, k] + 1e-16)
        H[k, :] = np.maximum(LH, np.minimum(UH, hk))
        
    return H

def als_bound(X, r, W_init, H_init, LW, UW, LH, UH, max_iter=10):
    """
    Alternating Least Squares (ALS) with bounds for integer matrices.
    Alternates between updating W and H while respecting the bounds.
    """
    m, n = X.shape
    r = W_init.shape[1]
    W = W_init.astype(float)
    H = H_init.astype(float)
    
    norm_X_sq = np.sum(X**2)

    prev_error = float('inf')

    iter = 0
    while(iter < max_iter):
        # Update H

        WtW = W.T @ W
        WtX = W.T @ X

        H = optimizeH(X, WtW, WtX, H, LH, UH)
        
        # Update W

        HHt = H @ H.T
        HXt = H @ X.T

        W = optimizeH(X.T, HHt, HXt, W.T, LW, UW).T

        WtW = W.T @ W
        WtX = W.T @ X

        error = norm_X_sq - 2 * np.sum(WtX * H) + np.sum(WtW * HHt)

        if abs(prev_error - error) < 1e-5:
            break
        prev_error = error

        iter += 1
    
    return W, H

# --- POPULATION GENERATION ---

def generate_population_W(X, r, N, LW, UW, LH, UH, config=None, verbose=False, perturbation_sigma=0.0):
    """
    Generates the initial population of W matrices.
    Uses a mix of SVD, Greedy Residual, KMeans, NMF, and Column Sampling, 
    followed by perturbations to fill the population.
    """
    m, n = X.shape
    population_W = []
    seen_hashes = set()

    # if perturbation_sigma > 1e-5:
    #     noise = np.random.normal(0, perturbation_sigma, size=X.shape)
    #     X_work = X.astype(float) + noise
    #     if verbose: print(f"--> Génération Hybride (Noise Sigma={perturbation_sigma:.1f})...")
    # else:
    #     X_work = X
    #     if verbose: print(f"--> Génération Hybride (Clean)...")

    # # A. Graines Structurelles
    # if config is None or config.use_svd:
    #     try:
    #         W_svd = initialize_SVD_smart(X_work, r, LW, UW)
    #         if W_svd.tobytes() not in seen_hashes:
    #             population_W.append(W_svd); seen_hashes.add(W_svd.tobytes())
    #     except: pass

    # if config is None or config.use_greedy:
    #     try:
    #         W_res = initialize_greedy_residual(X_work, r, LW, UW, LH, UH)
    #         if W_res.tobytes() not in seen_hashes:
    #             population_W.append(W_res); seen_hashes.add(W_res.tobytes())
    #     except: pass

    # if config is None or config.use_kmeans:
    #     try:
    #         W_kmeans = integer_kmeans(X_work, r, LW, UW)
    #         if W_kmeans.tobytes() not in seen_hashes:
    #             population_W.append(W_kmeans); seen_hashes.add(W_kmeans.tobytes())
    #     except: pass

    # if config is None or config.use_nmf:
    #     try:
    #         W_nmf_float, _, _ = ExtendedNMF(X_work, r, max_iter=500)
    #         W_nmf = round_matrix(W_nmf_float, LW, UW)
    #         if W_nmf.tobytes() not in seen_hashes:
    #             population_W.append(W_nmf); seen_hashes.add(W_nmf.tobytes())
    #     except: pass
    
    # # B. Remplissage Diversifié
    # target_sampling = int(N * 0.35)
    # for _ in range(target_sampling):
    #     if len(population_W) >= N: break
    #     W_samp = initialize_column_sampling(X, r, LW, UW)
    #     if np.random.rand() < 0.3:
    #         c1, c2 = np.random.randint(0, r, 2)
    #         W_samp[:, [c1, c2]] = W_samp[:, [c2, c1]]
        
    #     if W_samp.tobytes() not in seen_hashes:
    #          population_W.append(W_samp); seen_hashes.add(W_samp.tobytes())
             
    # seeds = [w for w in population_W]
    # idx = 0
    # while len(population_W) < int(N * 0.85):
    #     if not seeds: break
    #     base = seeds[idx % len(seeds)]
    #     idx += 1
    #     rate = np.random.uniform(0.15, 0.40) 
    #     W_new = perturb_W(base, rate, LW, UW)
    #     if W_new.tobytes() not in seen_hashes:
    #          population_W.append(W_new); seen_hashes.add(W_new.tobytes())

    for _ in range((N*25)//100):
        W_rand = np.random.randint(LW, UW + 1, size=(m, r))
        H_rand = np.random.randint(LH, UH + 1, size=(r, n))
        W_opt, H_opt = als_bound(X, r, W_rand.astype(float), H_rand.astype(float), LW, UW, LH, UH, max_iter=100)

        W_opt = np.round(W_opt).astype(int)

        if W_opt.tobytes() not in seen_hashes:
            population_W.append(W_opt)
            seen_hashes.add(W_opt.tobytes())

    for _ in range(int(N*25)//100):
        if len(population_W) >= N: break
        W_samp = initialize_column_sampling(X, r, LW, UW)
        if np.random.rand() < 0.3:
            c1, c2 = np.random.randint(0, r, 2)
            W_samp[:, [c1, c2]] = W_samp[:, [c2, c1]]
        
        if W_samp.tobytes() not in seen_hashes:
             population_W.append(W_samp); seen_hashes.add(W_samp.tobytes())

    seeds = [w for w in population_W]
    idx = 0
    while len(population_W) < int(N*8)//10 :
        if not seeds: break
        base = seeds[idx % len(seeds)]
        idx += 1
        rate = np.random.uniform(0.15, 0.40) 
        W_new = perturb_W(base, rate, LW, UW)
        if W_new.tobytes() not in seen_hashes:
             population_W.append(W_new); seen_hashes.add(W_new.tobytes())

    while len(population_W) < N:
        W_rand = np.random.randint(LW, UW + 1, size=(m, r))
        if W_rand.tobytes() not in seen_hashes:
            population_W.append(W_rand)
            seen_hashes.add(W_rand.tobytes())

    return population_W