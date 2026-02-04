import numpy as np
import time

def initialize_column_sampling(X, r, LW, UW):
    """
    Initializes W by sampling columns from X based on their norms.
    Columns with larger norms have a higher probability of being selected.
    """
    m, n = X.shape
    col_norms = np.linalg.norm(X, axis=0)
    probs = col_norms / np.sum(col_norms)
    indices = np.random.choice(n, r, replace=False)
    
    W = X[:, indices].copy()
    W = np.clip(W, LW, UW) 
    return W.astype(int)

def perturb_W(W, mutation_rate, LW, UW):
    """
    Perturbs W by changing a percentage of its entries to random values.
    """
    m, r = W.shape
    W_new = W.copy()
    
    if np.random.rand() < 0.5:
        col = np.random.randint(0, r)
        W_new[:, col] = LW
    else:
        h_block = m // 4
        if h_block < 1: h_block = 1
        start_row = np.random.randint(0, m - h_block + 1)
        W_new[start_row : start_row + h_block, :] = LW
        
    return W_new

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
        WtW = W.T @ W
        WtX = W.T @ X

        H = optimizeH(X, WtW, WtX, H, LH, UH)
        
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

def generate_population_W(X, r, N, LW, UW, LH, UH, config=None, verbose=False, perturbation_sigma=0.0):
    """
    Generates the initial population of W matrices.
    Uses a mix of SVD, Greedy Residual, KMeans, NMF, and Column Sampling, 
    followed by perturbations to fill the population.
    """
    m, n = X.shape
    population_W = []
    seen_hashes = set()

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