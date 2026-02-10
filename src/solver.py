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

def metaheuristic(X, r, LW, UW, LH, UH, mode_opti, TIME_LIMIT=300.0, N=100, debug_mode = False):
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

        if type(W_opt) != list:
            H_opt, f = optimizeHforW(
                X_f, W_opt, H_rand, LW, UW, LH, UH, mode_opti
            )
        else:
            f = W_opt[0]
            H_opt = W_opt[1][1]
            W_opt = W_opt[1][0]

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
    
    if debug_mode:
        print(f"[DEBUG] Starting Metaheuristic with Phase: {current_phase}, Initial Best Fitness: {best_f:.6f}")

    while time.time() - start_time < TIME_LIMIT - 5:

        if debug_mode and iteration % 50 == 0:
            print(f"[DEBUG] Iteration {iteration}, Best Fitness: {best_f:.6f}, Population Size: {len(population)}, Phase: {current_phase}")
        
        switch_triggered = False
        if iters_in_phase > 1: switch_triggered = True
            
        if switch_triggered:
            if current_phase == 'DIRECT':
                current_phase = 'TRANSPOSE'
                population = transpose_population(population)
            else:
                current_phase = 'DIRECT'
                population = transpose_population(population)
            iters_in_phase = 0

        if current_phase == 'DIRECT':
            active_X = X
            G_L, G_U = LW, UW; P_L, P_U = LH, UH
        else:
            active_X = X.T
            G_L, G_U = LH, UH; P_L, P_U = LW, UW

        best_ind = min(population, key=lambda x: x[0])
        best_W, best_H = best_ind[1]
        for _ in range((N*10)//100):
            noise_W = np.random.randint(-(G_U-G_L), (G_U-G_L) + 1, size=best_W.shape)
            noise_H = np.random.randint(-(P_U-P_L), (P_U-P_L) + 1, size=best_H.shape)
            
            W_init = np.clip(best_W + noise_W, G_L, G_U)
            H_init = np.clip(best_H + noise_H, P_L, P_U)

            W_opti, H_opti, f = optimize_alternating_wrapper(
                active_X, W_init, H_init, G_L, G_U, P_L, P_U, mode_opti, max_iters=1
            )

            population.append([f, (W_opti, H_opti)])

        for _ in range((N*10)//100):
            W_init = np.random.randint(G_L, G_U + 1, size=best_W.shape)
            H_init = np.random.randint(P_L, P_U + 1, size=best_H.shape)

            W_opti, H_opti, f = optimize_alternating_wrapper(
                active_X, W_init, H_init, G_L, G_U, P_L, P_U, mode_opti, max_iters=5
            )

            population.append([f, (W_opti, H_opti)])

        temp_hashes = set() 
        children = generateNewGeneration(
            temp_hashes, population, len(population)//2, active_X, 
            G_L, G_U, P_L, P_U, mode_opti,
            start_time, TIME_LIMIT
        )
        
        if children:
            for child in children:
                child_W, child_H, child_f, p1_idx, p2_idx, d1, d2 = child
                if d1 <= d2:
                    target_idx = p1_idx
                else:
                    target_idx = p2_idx

                if child_f < population[target_idx][0]:
                    population[target_idx] = [child_f, (child_W, child_H)]
                    if child_f < global_best_f:
                        global_best_f = child_f
                        if current_phase == 'DIRECT':
                            global_best_W, global_best_H = child_W.copy(), child_H.copy()
                        else:
                            global_best_W, global_best_H = child_H.T.copy(), child_W.T.copy()

            population.sort(key=lambda x: x[0])
            population = population[:N]

            current_best = population[0]
            
            if current_best[0] < best_f - 1e-3:
                best_f = current_best[0]
                if current_phase == 'DIRECT':
                    best_W, best_H = current_best[1]
                else:
                    H_T, W_T = current_best[1]
                    best_W, best_H = W_T.T, H_T.T
                
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