import time
import numpy as np
import copy

from .local_search import optimize_alternating_wrapper, optimizeHforW
from .createChild import generateNewGeneration, align_parents
from .init_pop import generate_population_W

try:
    from .fast_solver import get_aligned_distance
    USE_CPP_DIST = True
except ImportError:
    print("Warning: C++ module not found. Please compile with setup.py. (get_aligned_distance)")
    USE_CPP_DIST = False

def get_distance_py(W1, W2):
    """
    Calculates the Hamming distance between two matrices W1 and W2 after aligning them.
    This is the Python implementation used when the C++ extension is not available.
    """
    W2_al = align_parents(W1, W2)
    return np.count_nonzero(W1 != W2_al)

def transpose_individual(ind):
    """
    Transposes an individual (solution).
    Swaps W and H and transposes them, keeping the fitness f unchanged.
    """
    f, (W, H) = ind
    return [f, (H.T.copy(), W.T.copy())]

def transpose_population(pop):
    """
    Transposes the entire population of individuals.
    """
    return [transpose_individual(p) for p in pop]

def select_diverse_survivors(X, population, N, min_diff_percent=0.001, LW=0, UW=1, LH=0, UH=1, mode_opti=None):
    """
    Selects the N best diverse survivors.
    If diversity is insufficient, fills the rest with mutations
    rather than keeping duplicates.
    """

    m, n = X.shape

    population.sort(key=lambda x: x[0])
    
    if not population:
        return population, 0

    survivors = []
    survivors.append(population[0]) 

    m, r = population[0][1][0].shape
    min_pixels = int(m * r * min_diff_percent)
    if min_pixels < 1: min_pixels = 1
    
    processed_count = 0
    
    for i in range(1, len(population)):
        if len(survivors) >= N: break
        
        candidate = population[i]
        W_cand = candidate[1][0]
        
        is_distinct = True
        
        check_window = max(20, int(len(survivors) * 0.5))
        
        start_check = max(0, len(survivors) - check_window)
        
        for k in range(start_check, len(survivors)):
            survivor = survivors[k]
            
            if USE_CPP_DIST:
                dist = get_aligned_distance(survivor[1][0].astype(np.int32), W_cand.astype(np.int32))
            else:
                dist = get_distance_py(survivor[1][0], W_cand)
                
            if dist < min_pixels:
                is_distinct = False
                break
        
        if is_distinct:
            survivors.append(candidate)
            
    num_natural_survivors = len(survivors)

    while len(survivors) < N:
        num_candidates = 10
        candidates = []
        for _ in range(num_candidates):
            W_cand = np.random.randint(LW, UW + 1, size=(m, r))
            candidates.append(W_cand)
        
        reference_population = [s[1][0] for s in survivors[:min(len(survivors), 10)]]
        
        best_W_far = candidates[0]
        max_min_distance = -1
        
        for W_c in candidates:
            min_dist_to_group = float('inf')
            
            for W_ref in reference_population:
                if USE_CPP_DIST:
                    d = get_aligned_distance(W_ref.astype(np.int32), W_c.astype(np.int32))
                else:
                    d = get_distance_py(W_ref, W_c)
                
                if d < min_dist_to_group:
                    min_dist_to_group = d
            
            if min_dist_to_group > max_min_distance:
                max_min_distance = min_dist_to_group
                best_W_far = W_c
        
        H_new = np.random.randint(LH, UH + 1, size=(r, n))
        
        H_final, f_final = optimizeHforW(X, best_W_far, H_new, LW, UW, LH, UH, mode_opti)
        
        survivors.append([f_final, (best_W_far, H_final)])

    return survivors, num_natural_survivors

def metaheuristic(X, r, LW, UW, LH, UH, mode_opti, TIME_LIMIT=300.0, N=100, tournament_size=4, debug_mode = False):
    """
    Main metaheuristic function for solving the matrix factorization problem.
    """

    start_time = time.time()
    m, n = X.shape
    X_f = X.astype(float)
    
    population = []
    seen_hashes = set()
    
    trace_iter = []
    trace_best = []
    trace_avg = []
    
    global_best_W = None
    global_best_H = None
    global_best_f = float('inf')

    if debug_mode:
        print("Debug Mode Activated: Detailed logs will be shown.")
        print("Initial Population Generation...")

    pop_W_list = generate_population_W(X, r, N, LW, UW, LH, UH, mode_opti)
    
    if debug_mode:
        print(f"Generated {len(pop_W_list)} initial W matrices.")

    for i, W_opt in enumerate(pop_W_list):
        if time.time() - start_time > TIME_LIMIT - 5: break
        H_rand = np.random.randint(LH, UH + 1, size=(r, n))

        W_opt, H_opt, f = optimize_alternating_wrapper(
            X_f, W_opt, H_rand, LW, UW, LH, UH, mode_opti, max_iters=10
        )

        child_hash = (W_opt.tobytes(), H_opt.tobytes())
        if child_hash not in seen_hashes:
            seen_hashes.add(child_hash)
            population.append([f, (W_opt, H_opt)])

    population.sort(key=lambda x: x[0])
    if not population: return np.zeros((m,r)), np.zeros((r,n)), float('inf')

    if debug_mode:
        print(f"Initial Population of size {len(population)} generated.")
        print(f"Best initial fitness: {population[0][0]:.6f}")
        print("Starting Main Evolutionary Loop...")

    best_f = population[0][0]
    best_W, best_H = population[0][1]
    global_best_f = best_f
    global_best_W, global_best_H = best_W.copy(), best_H.copy()
    
    trace_iter.append(0)
    trace_best.append(best_f)
    trace_avg.append(np.mean([p[0] for p in population]))
    
    iteration = 0
    
    current_phase = 'DIRECT'

    iters_in_phase = 0
    
    min_diff_percent = 0.001
    
    if debug_mode:
        print(f"[DEBUG] Starting Metaheuristic with Phase: {current_phase}, Initial Best Fitness: {best_f:.6f}")

    while time.time() - start_time < TIME_LIMIT - 5:

        if debug_mode and iteration % 50 == 0:
            print(f"[DEBUG] Iteration {iteration}, Best Fitness: {best_f:.6f}, Population Size: {len(population)}, Phase: {current_phase}")
        
        switch_triggered = False
        if iters_in_phase > 30: switch_triggered = True
            
        if switch_triggered:
            if current_phase == 'DIRECT':
                current_phase = 'TRANSPOSE'
                population = transpose_population(population)
            else:
                current_phase = 'DIRECT'
                population = transpose_population(population)
            iters_in_phase = 0
            min_diff_percent = min(0.05, min_diff_percent * 1.5)

        if current_phase == 'DIRECT':
            active_X = X
            G_L, G_U = LW, UW; P_L, P_U = LH, UH
        else:
            active_X = X.T
            G_L, G_U = LH, UH; P_L, P_U = LW, UW

        temp_hashes = set() 
        children = generateNewGeneration(
            temp_hashes, population, N//3, active_X, 
            G_L, G_U, P_L, P_U, mode_opti,
            start_time, TIME_LIMIT, int(tournament_size)
        )
        
        if children:
            for child in children:
                if len(child) < 2: continue
                f_child = child[0]
                W_child, H_child = child[1]
                population.append([f_child, (W_child, H_child)])
            
            population, _ = select_diverse_survivors(active_X, population, N, min_diff_percent, G_L, G_U, P_L, P_U, mode_opti)

            population.sort(key=lambda x: x[0])

            current_best = population[0]
            
            if current_best[0] < best_f - 1e-3:
                best_f = current_best[0]
                if current_phase == 'DIRECT':
                    best_W, best_H = current_best[1]
                else:
                    H_T, W_T = current_best[1]
                    best_W, best_H = W_T.T, H_T.T
                
                min_diff_percent = 0.001
                
                if best_f < global_best_f:
                    global_best_f = best_f
                    global_best_W, global_best_H = best_W.copy(), best_H.copy()

        iters_in_phase += 1
        iteration += 1

        if global_best_f == 0:
            break

    remaining = TIME_LIMIT - (time.time() - start_time)
    
    final_W, final_H, final_f = global_best_W, global_best_H, global_best_f

    if remaining > 1.0:
        final_W, final_H, final_f = optimize_alternating_wrapper(
            X_f, final_W, final_H, LW, UW, LH, UH, mode_opti, max_iters=2000, time_limit=remaining
        )
    
    if final_f < global_best_f:
        global_best_W, global_best_H, global_best_f = final_W, final_H, final_f

    if debug_mode:
        print(f"[DEBUG] Metaheuristic completed in {time.time() - start_time:.2f}s over {iteration} iterations.")
        print(f"[DEBUG] Global Best Fitness: {global_best_f:.6f}")

    return global_best_W, global_best_H, global_best_f