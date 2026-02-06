import numpy as np
import time

from .local_search import optimize_alternating_wrapper

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

def generate_population_W(X, r, N, LW, UW, LH, UH, mode_opti, config=None, verbose=False, perturbation_sigma=0.0):
    """
    Generates the initial population of W matrices.
    Uses a mix of SVD, Greedy Residual, KMeans, NMF, and Column Sampling, 
    followed by perturbations to fill the population.
    """
    m, n = X.shape
    population_W = []
    seen_hashes = set()
    best = 1e20

    for _ in range((N*25)//100):
        W_rand = np.random.randint(LW, UW + 1, size=(m, r))
        H_rand = np.random.randint(LH, UH + 1, size=(r, n))

        W_opt, H_opt, f = optimize_alternating_wrapper(X, W_rand, H_rand, LW, UW, LH, UH, mode_opti, max_iters=100)

        if f < best:
            best = f
            best_W = W_opt
            best_H = H_opt

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