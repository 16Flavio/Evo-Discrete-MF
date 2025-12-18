import numpy as np
from scipy.spatial.distance import cdist
import sys, time
from utils.verification import fobj

def _kmeans_plus_plus_init(X_T, r):
    """
    Helper function for K-means++ initialization.
    Takes X_T (n_samples, n_features) and returns r initial centroids.
    """
    n_samples, n_features = X_T.shape
    
    # 1. Choisir le premier centroïde (C1) au hasard
    first_idx = np.random.choice(n_samples)
    centroids_T = [X_T[first_idx]] # Liste pour stocker les centroïdes (transposés)
    
    # 2. Choisir les centroïdes C2 à Cr
    for _ in range(r - 1):
        # 2a. Calculer la distance au carré D(x)^2 de chaque point
        #     au centroïde le plus proche DÉJÀ choisi.
        current_centroids_array = np.array(centroids_T)
        
        # dist_matrix.shape = (n_samples, n_centroids_choisis)
        dist_matrix = cdist(X_T, current_centroids_array, 'sqeuclidean')
        
        # min_dists_sq.shape = (n_samples,)
        min_dists_sq = np.min(dist_matrix, axis=1)
        
        # 2b. Calculer les probabilités
        if np.sum(min_dists_sq) == 0:
            probabilities = np.ones(n_samples) / n_samples
        else:
            probabilities = min_dists_sq / np.sum(min_dists_sq)
        
        # 2c. Choisir le nouveau centroïde basé sur ces probabilités
        next_idx = np.random.choice(n_samples, p=probabilities)
        centroids_T.append(X_T[next_idx])

    return np.array(centroids_T)

def integer_kmeans(X, r, LW, UW, max_iters=100):
    """
    Implements modified K-means to find W.
    - The "points" are the columns of X.
    - The "centroids" (columns of W) are constrained to be integers in [LW, UW].
    """
    n_features, n_samples = X.shape

    # 1. Initialiser W avec K-means++
    X_T = X.T
    
    # W_T a pour shape (r, n_features)
    W_T = _kmeans_plus_plus_init(X_T, r)
    
    # Transposer W_T (r, n_features) -> W (n_features, r)
    W = W_T.T 
    
    # Appliquer les contraintes de bornes
    W = np.clip(W, LW, UW).astype(float)

    assignments = np.zeros(n_samples, dtype=int)

    for _ in range(max_iters):
        # 2. Étape d'assignation :
        #    Assigner chaque colonne de X (x_j) au centroïde (w_k) le plus proche.
        #    cdist(A, B) attend des (n_samples, n_features).
        #    Nous comparons n_samples (X.T) à r centroïdes (W.T).
        dist_matrix = cdist(X.T, W.T, 'sqeuclidean')
        new_assignments = np.argmin(dist_matrix, axis=1)

        # Critère d'arrêt : si les assignations n'ont pas changé
        if np.array_equal(assignments, new_assignments):
            break
        assignments = new_assignments

        # 3. Étape de mise à jour (modifiée) :
        #    Recalculer chaque centroïde k.
        for k in range(r):
            # Trouver toutes les colonnes de X assignées au cluster k
            cluster_points = X[:, assignments == k]

            if cluster_points.shape[1] > 0:
                # a) Calculer le centroïde réel (moyenne)
                mean_vector = np.mean(cluster_points, axis=1)
                
                # b) Arrondir à l'entier le plus proche
                int_vector = np.round(mean_vector)
                
                # c) Borner (clip) aux contraintes [LW, UW]
                clipped_vector = np.clip(int_vector, LW, UW)
                
                W[:, k] = clipped_vector
            else:
                idx_reinit = np.random.choice(n_samples, 1)
                W[:, k] = np.clip(X[:, idx_reinit].squeeze(), LW, UW).astype(float)
                
    # Retourner W en tant qu'entiers
    return W.astype(int)

def refine_H_coordinate_descent(X, W, H_init, LW, UW, LH, UH, max_iters=100):
    """
    Refines H using coordinate descent.
    H_init is the starting point (e.g., from lstsq).
    """
    m, n = X.shape
    r = W.shape[1]
    
    # On travaille sur une copie de H
    H = H_init.copy()
    
    H[~np.isfinite(H)] = LH
    W[~np.isfinite(W)] = LW if LW != 0 else 1e-6

    # pré-calculer le carré de la norme de chaque colonne de W
    W_col_sq_norm = np.sum(W**2, axis=0)
    
    # Éviter la division par zéro si une colonne de W est nulle
    W_col_sq_norm = np.where(~np.isfinite(W_col_sq_norm) | (W_col_sq_norm == 0), 1e-6, W_col_sq_norm)

    # Optimiser chaque colonne h_j de H indépendamment
    for j in range(n):
        h_j = H[:, j]
        x_j = X[:, j]
        errors = [np.sum((x_j - W@h_j)**2)]

        if ~np.isfinite(x_j).all(): 
            continue
        
        # Itérer pour raffiner h_j (max_iters fois ou jusqu'à convergence)
        for _ in range(max_iters):
            changed = False
            
            # Optimiser chaque coefficient k de h_j
            for k in range(r):
                # Calculer le résidu SANS la contribution de h_j[k]
                # current_approximation = W @ h_j
                # residual = x_j - current_approximation + W[:, k] * h_j[k]
                
                # Version plus rapide :
                h_j[k] = 0 # Mettre temporairement à 0
                residual = x_j - W @ h_j
                
                # Nous voulons trouver h_kj qui minimise ||residual - W[:, k] * h_kj||^2
                # La solution réelle est (W[:, k] . residual) / (W[:, k] . W[:, k])
                
                if ~np.isfinite(residual).all():
                    residual[~np.isfinite(residual)] = 0

                # a) Trouver la solution réelle optimale pour h_kj
                with np.errstate(divide='ignore', invalid='ignore'):
                    h_star_kj = np.dot(W[:, k].astype(float), residual.astype(float)) / W_col_sq_norm[k]
                
                if not np.isfinite(h_star_kj):
                    h_star_kj = 0

                # b) Trouver les deux candidats entiers optimaux
                h_cand_1 = np.floor(h_star_kj)
                h_cand_2 = np.ceil(h_star_kj)
                
                # c) Appliquer les bornes
                h_cand_1 = np.clip(h_cand_1, LH, UH).astype(int)
                h_cand_2 = np.clip(h_cand_2, LH, UH).astype(int)

                # d) Évaluer l'erreur pour les deux candidats
                err_1 = np.sum((residual - W[:, k] * h_cand_1)**2)
                err_2 = np.sum((residual - W[:, k] * h_cand_2)**2)

                if err_1 <= err_2:
                    h_new_kj = h_cand_1
                else:
                    h_new_kj = h_cand_2

                # e) Mettre à jour
                if h_new_kj != H[k, j]: # Note: on compare à H[k,j] car h_j[k] est 0
                    H[k, j] = h_new_kj
                    changed = True
                else:
                    H[k, j] = h_new_kj # Rétablir la valeur
            
            errors.append(np.sum((x_j - W@h_j)**2))

            if not changed:
                # Si une passe complète n'a rien changé, h_j est à un optimum local
                break

            if np.abs(errors[-1]-errors[-2]) < 1:
                break

    return H

def refine_W_coordinate_descent(X, W_init, H, LW, UW, LH, UH, max_iters=100):
    """
    Refines W using coordinate descent, keeping H fixed.
    
    This is equivalent to solving the transposed problem:
    min ||X^T - (H^T)(W^T)||^2
    
    Where:
    - X_transposed = X.T
    - W_transposed = H.T  (our fixed "W")
    - H_transposed = W.T  (our "H" to optimize)
    - Bounds for H_transposed = [LW, UW]
    """
    
    # 1. Transposer le problème
    X_T = X.T
    H_T = H.T
    W_T_init = W_init.T
    
    # 2. Résoudre le problème transposé avec la fonction H existante
    W_T_refined = refine_H_coordinate_descent(
        X_T, H_T, W_T_init, LH, UH, LW, UW, max_iters
    )
    
    # 3. Re-transposer le résultat
    return W_T_refined.T

def calculate_H_integer_lstsq(X, W, LW, UW, LH, UH):
    """
    Calculates H by solving min ||x_j - W h_j||^2 for each column j,
    then rounding and clipping the result h_j to [LH, UH].
    """
    r = W.shape[1]
    n = X.shape[1]
    H = np.zeros((r, n), dtype=int)
    
    # lstsq fonctionne mieux avec des flottants
    W_float = W.astype(float)

    for j in range(n):
        x_j = X[:, j]
        
        # 1. Résoudre le problème des moindres carrés en réels
        #    h_star est la solution optimale réelle h_j*
        h_star, _, _, _ = np.linalg.lstsq(W_float, x_j, rcond=None)
        
        # 2. Arrondir la solution à l'entier le plus proche
        h_int = np.round(h_star)
        
        # 3. Borner (clip) la solution aux contraintes [LH, UH]
        h_j = np.clip(h_int, LH, UH)
        
        H[:, j] = h_j

    H = refine_H_coordinate_descent(X, W, H, LW, UW, LH, UH)

    return H

def initialize_WH_kmeans(X, r, LW, UW, LH, UH):
    """
    Main initialization function:
    1. Finds W with integer K-means.
    2. Finds H with integer least squares.
    """
    
    # Étape 1: Trouver W
    W = integer_kmeans(X, r, LW, UW)
    
    # Étape 2: Trouver H
    H = calculate_H_integer_lstsq(X, W, LW, UW, LH, UH)
    
    return W, H

def find_best_initialization(n_restarts, X, r, LW, UW, LH, UH):
    """
    Runs initialization n_restarts times and returns the best pair (W, H) found.
    """
    
    # Initialiser la "meilleure" solution trouvée
    best_W = None
    best_H = None
    best_error = np.inf  # Erreur initiale = infini

    for i in range(n_restarts):
        # 1. Générer une solution initiale
        current_W, current_H = initialize_WH_kmeans(X, r, LW, UW, LH, UH)
        
        # 2. Évaluer cette solution [cite: 184-187]
        current_error = fobj(X, current_W, current_H)
        
        # 3. Mettre à jour si elle est meilleure
        if current_error < best_error:
            best_error = current_error
            best_W = current_W
            best_H = current_H

    return best_W, best_H, best_error

def initialize_WH_alternating_descent(X, r, LW, UW, LH, UH, TIME_LIMIT=30.0):
    """
    Complete initialization pipeline:
    1. K-means for W.
    2. LSQ for H.
    3. Alternating Coordinate Descent (ACD) loop to refine (W, H).
    """
    # Définir la répartition du temps
    time_limit_phase1 = TIME_LIMIT * 0.70
    time_limit_phase2 = TIME_LIMIT * 0.30
    
    start_time_global = time.time()
    
    # --- PHASE 1: K-means (Exploration) ---
    start_time_phase1 = time.time()
    # Initialiser la "meilleure" solution trouvée
    best_W = None
    best_H = None
    best_error = np.inf  # Erreur initiale = infini

    while (time.time() - start_time_phase1) < time_limit_phase1:
        # Générer une solution initiale
        current_W, current_H = initialize_WH_kmeans(X, r, LW, UW, LH, UH)

        # Évaluer cette solution [cite: 184-187]
        current_error = fobj(X, current_W, current_H)
        
        # Mettre à jour si elle est meilleure
        if current_error < best_error:
            best_error = current_error
            best_W = current_W
            best_H = current_H
    
    # --- PHASE 2: Alternating Descent (Exploitation) ---
    start_time_phase2 = time.time()

    if (time.time() - start_time_global) >= TIME_LIMIT:
        return best_W, best_H, best_error

    W = best_W
    H = best_H
    current_error = best_error

    # --- Raffinage alterné ---
    while (time.time() - start_time_phase2) < time_limit_phase2:
        # Raffiner H (en gardant W fixe)
        H = refine_H_coordinate_descent(X, W, H, LW, UW, LH, UH)
        
        # Raffiner W (en gardant H fixe)
        W = refine_W_coordinate_descent(X, W, H, LW, UW, LH, UH)
        
        new_error = fobj(X, W, H)

        # Critère d'arrêt
        if new_error >= current_error:
            break
        current_error = new_error
        
        if (time.time() - start_time_global) >= TIME_LIMIT:
            break

    return W, H, current_error