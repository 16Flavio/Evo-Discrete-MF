import time
import numpy as np

from .local_search import optimize_alternating_wrapper
from .createChild import generateNewGeneration, align_parents
from .init_pop import generate_population_W, perturb_W, generate_antithetic_W, perturb_W_destructive

try:
    from .fast_solver import get_aligned_distance
    USE_CPP_DIST = True
except ImportError:
    print("Warning: Module C++ non trouvé. Veuillez compiler avec setup.py.")
    USE_CPP_DIST = False

# --- UTILS ---

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

def select_diverse_survivors(population, N, min_diff_percent=0.001):
    """
    Selects the top N survivors from the population, ensuring diversity.
    It prioritizes the best solutions but skips those that are too similar to already selected ones
    based on a minimum difference percentage.
    """
    population.sort(key=lambda x: x[0])
    if min_diff_percent <= 0 or not population:
        return population[:N], len(population)
    
    survivors = []
    survivors.append(population[0]) 
    
    m, r = population[0][1][0].shape
    min_pixels = int(m * r * min_diff_percent)
    if min_pixels < 1: min_pixels = 1
    
    check_limit = 20
    
    for i in range(1, len(population)):
        if len(survivors) >= N: break
        candidate = population[i]
        W_cand = candidate[1][0]
        is_distinct = True
        start_check = max(0, len(survivors) - check_limit)
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

    if len(survivors) < N:
        survivor_ids = {id(p) for p in survivors}
        for p in population:
            if len(survivors) >= N: break
            if id(p) not in survivor_ids:
                survivors.append(p)
                survivor_ids.add(id(p))
    
    return survivors, len(survivors)

def ruin_and_recreate(W, G_L, G_U, ruin_percent=0.2):
    """
    Large Neighborhood Search (LNS) operator.
    Destroys 'ruin_percent' of the columns of W and resets them to random values.
    """
    m, r = W.shape
    W_new = W.copy()
    n_ruin = int(r * ruin_percent)
    if n_ruin < 1: n_ruin = 1
    
    # Randomly select columns to ruin
    cols_to_ruin = np.random.choice(r, n_ruin, replace=False)
    
    for col in cols_to_ruin:
        # Reset to random values within bounds
        W_new[:, col] = np.random.randint(G_L, G_U + 1, size=m)
        
    return W_new

def apply_smart_restart(best_W, best_H, best_f, N, X, LW, UW, LH, UH, seen_hashes, current_phase, restart_count, population, config=None):
    """
    ENHANCED RESTART STRATEGY V2
    - Level 1 (Count 1): Soft Restart (Keep Best + Diverse)
    - Level 2 (Count 2): Ruins & Recreate (LNS) on Best
    - Level 3+ (Count > 2): Alien Restart (Antithetic)
    """
    actual_restart_count = restart_count
    if config and config.restart_mode == "SIMPLE":
        actual_restart_count = 1

    # Config Phase
    if current_phase == 'DIRECT':
        X_gen = X
        G_L, G_U = LW, UW; P_L, P_U = LH, UH
        ref_W, ref_H = best_W, best_H
        m, r = best_W.shape; n = best_H.shape[1]
    else:
        X_gen = X.T
        G_L, G_U = LH, UH; P_L, P_U = LW, UW
        ref_W, ref_H = best_H.T, best_W.T
        m, r = best_H.shape[1], best_H.shape[0]; n = best_W.shape[0]

    X_f_gen = X_gen.astype(float)

    range_val = float(G_U - G_L)
    if range_val < 1.0: range_val = 1.0
    
    sigma = float(restart_count) * (range_val * 0.05)
    new_population = []

    # --- STRATÉGIE DE RESTART ---
    
    if actual_restart_count == 1:
        new_population.append([best_f, (ref_W.copy(), ref_H.copy())])
        
        # Keep 3 diverse high-quality individuals
        if len(population) > 5:
            diverse_count = 0
            for i in range(1, len(population)):
                cand = population[i]
                if USE_CPP_DIST:
                    dist = get_aligned_distance(ref_W.astype(np.int32), cand[1][0].astype(np.int32))
                else:
                    dist = get_distance_py(ref_W, cand[1][0])
                
                if dist > (m * r * 0.05): 
                    new_population.append(cand)
                    diverse_count += 1
                if diverse_count >= 3: break
        sigma = range_val * 0.15

    elif actual_restart_count == 2:
        
        # 1. Keep the Best (Elitism) - CRITICAL FIX
        new_population.append([best_f, (ref_W.copy(), ref_H.copy())])
        
        # 2. Add Ruined version
        W_ruined = ruin_and_recreate(ref_W, G_L, G_U, ruin_percent=0.15)
        H_rand = np.random.randint(P_L, P_U + 1, size=(r, n))
        W_opt, H_opt, f = optimize_alternating_wrapper(
            X_f_gen, W_ruined, H_rand, G_L, G_U, P_L, P_U, max_iters=40, effort=2
        )
        new_population.append([f, (W_opt, H_opt)])
        sigma = range_val * 0.30

    else:
        # Generate Antithetic from the SECOND best to avoid cycling
        target_W = ref_W
        if len(population) > 1: target_W = population[1][1][0]

        W_anti = generate_antithetic_W(target_W, G_L, G_U)
        H_rand = np.random.randint(P_L, P_U + 1, size=(r, n))
        W_opt, H_opt, f = optimize_alternating_wrapper(
            X_f_gen, W_anti, H_rand, G_L, G_U, P_L, P_U, max_iters=20, effort=1
        )
        new_population.append([f, (W_opt, H_opt)])
        sigma = range_val * 0.80 

    # 1. Nouvelles Graines "Hallucinées" (Noisy Init)
    fresh_W_list = generate_population_W(
        X_gen, r, int(N), G_L, G_U, P_L, P_U, config=config,
        verbose=False, perturbation_sigma=sigma
    )
    
    for W_cand in fresh_W_list:
        if len(new_population) >= N: break
        H_rand = np.random.randint(P_L, P_U + 1, size=(r, n))
        W_opt, H_opt, f = optimize_alternating_wrapper(
            X_f_gen, W_cand, H_rand, G_L, G_U, P_L, P_U, config=config, max_iters=15, effort=1
        )
        child_hash = (W_opt.tobytes(), H_opt.tobytes())
        if child_hash not in seen_hashes:
            seen_hashes.add(child_hash)
            new_population.append([f, (W_opt, H_opt)])
            
    # 2. Remplissage Mutants & Destructeurs
    seed_base = ref_W
    while len(new_population) < N:
        mutation_type = np.random.rand()
        
        if mutation_type < 0.3:
            # Mutation classique
            W_mut = perturb_W(seed_base, np.random.uniform(0.1, 0.5), G_L, G_U)
        elif mutation_type < 0.6:
            # Mutation DESTRUCTIVE
            W_mut = perturb_W_destructive(seed_base, G_L, G_U)
        elif mutation_type < 0.8:
            # RUINS Mutation (New)
            W_mut = ruin_and_recreate(seed_base, G_L, G_U, ruin_percent=0.2)
        else:
            # Pure Random
            W_mut = np.random.randint(G_L, G_U + 1, size=(m, r))

        H_rand = np.random.randint(P_L, P_U + 1, size=(r, n))
        W_opt, H_opt, f = optimize_alternating_wrapper(
            X_f_gen, W_mut, H_rand, G_L, G_U, P_L, P_U, config=config, max_iters=10, effort=1
        )
        child_hash = (W_opt.tobytes(), H_opt.tobytes())
        if child_hash not in seen_hashes:
            seen_hashes.add(child_hash)
            new_population.append([f, (W_opt, H_opt)])

    new_population.sort(key=lambda x: x[0])
    return new_population

# --- METAHEURISTIC MAIN ---

def metaheuristic(X, r, LW, UW, LH, UH, TIME_LIMIT=300.0, N=100, tournament_size=4, mutation_rate=0.1, config=None):
    """
    Main metaheuristic function for solving the matrix factorization problem.
    """

    if config is None:
        from src.config import ConfigAblation
        config = ConfigAblation()

    start_time = time.time()
    m, n = X.shape
    X_f = X.astype(float)
    
    population = []
    seen_hashes = set()
    
    # Data recording for plotting
    trace_iter = []
    trace_best = []
    trace_avg = []
    
    # Global Best Memory
    global_best_W = None
    global_best_H = None
    global_best_f = float('inf')

    # --- 1. INITIALISATION ---
    pop_W_list = generate_population_W(X, r, N, LW, UW, LH, UH, config=config, verbose=False)
    
    for i, W_cand in enumerate(pop_W_list):
        if time.time() - start_time > TIME_LIMIT - 5: break
        H_rand = np.random.randint(LH, UH + 1, size=(r, n))
        iters = 25 if i < (N // 5) else 10
        eff = 2 if i < (N // 5) else 1
        
        W_opt, H_opt, f = optimize_alternating_wrapper(
            X_f, W_cand, H_rand, LW, UW, LH, UH, config=config, max_iters=iters, effort=eff
        )
        child_hash = (W_opt.tobytes(), H_opt.tobytes())
        if child_hash not in seen_hashes:
            seen_hashes.add(child_hash)
            population.append([f, (W_opt, H_opt)])

    population.sort(key=lambda x: x[0])
    if not population: return np.zeros((m,r)), np.zeros((r,n)), float('inf')

    # Init Global Best
    best_f = population[0][0]
    best_W, best_H = population[0][1]
    global_best_f = best_f
    global_best_W, global_best_H = best_W.copy(), best_H.copy()
    
    # --- RECORD & PLOT INITIAL STATE ---
    trace_iter.append(0)
    trace_best.append(best_f)
    trace_avg.append(np.mean([p[0] for p in population]))
    
    # if DEBUG:
    #     plot_debug_snapshot(trace_iter, trace_best, trace_avg, population, "Après Initialisation")

    iteration = 0
    stagnation_counter = 0
    last_improvement_time = time.time()
    restart_count = 0 
    
    current_phase = 'DIRECT' 
    iters_in_phase = 0
    
    curr_mut = mutation_rate
    curr_tourn = tournament_size
    
    # Dynamic Diversity Parameter
    min_diff_percent = 0.001
    
    # ADAPTIVE RESTART THRESHOLD
    base_restart_threshold = max(10.0, min(60.0, TIME_LIMIT * 0.15))
    restart_threshold = base_restart_threshold
    
    # --- 2. BOUCLE PRINCIPALE ---
    while time.time() - start_time < TIME_LIMIT - 5:
        current_time = time.time()
        
        # A. Changement de Phase
        switch_triggered = False
        if stagnation_counter > 10: switch_triggered = True
        elif iters_in_phase > 30: switch_triggered = True
            
        if switch_triggered and config.allow_transpose:
            if current_phase == 'DIRECT':
                current_phase = 'TRANSPOSE'
                population = transpose_population(population)
            else:
                current_phase = 'DIRECT'
                population = transpose_population(population)
            iters_in_phase = 0
            stagnation_counter = max(0, stagnation_counter - 5)
            min_diff_percent = min(0.05, min_diff_percent * 1.5)
        elif switch_triggered and not config.allow_transpose:
            stagnation_counter = 0
            iters_in_phase = 0

        # B. Préparation
        if current_phase == 'DIRECT':
            active_X = X
            G_L, G_U = LW, UW; P_L, P_U = LH, UH
        else:
            active_X = X.T
            G_L, G_U = LH, UH; P_L, P_U = LW, UW

        # C. Evolution
        temp_hashes = set() 
        children = generateNewGeneration(
            temp_hashes, population, N//3, active_X, 
            G_L, G_U, P_L, P_U, 
            start_time, TIME_LIMIT, int(curr_tourn), float(curr_mut),
            config=config
        )
        
        # D. Réintégration (Global Competition)
        if children:
            for child in children:
                if len(child) < 2: continue
                f_child = child[0]
                W_child, H_child = child[1]
                population.append([f_child, (W_child, H_child)])
            
            # Dynamic Diversity Selection
            population, _ = select_diverse_survivors(population, N, min_diff_percent)
            
            current_best = population[0]
            
            # Mise à jour du meilleur local
            if current_best[0] < best_f - 1e-3:
                best_f = current_best[0]
                if current_phase == 'DIRECT':
                    best_W, best_H = current_best[1]
                else:
                    H_T, W_T = current_best[1]
                    best_W, best_H = W_T.T, H_T.T
                
                stagnation_counter = 0
                last_improvement_time = current_time
                restart_count = 0 
                min_diff_percent = 0.001 
                
                # Reset Restart Threshold on improvement
                restart_threshold = max(10.0, base_restart_threshold * 0.75)
                
                # Mise à jour du GLOBAL BEST
                if best_f < global_best_f:
                    global_best_f = best_f
                    global_best_W, global_best_H = best_W.copy(), best_H.copy()

                # Intensification
                W_c, H_c = current_best[1]
                W_boost, H_boost, f_boost = optimize_alternating_wrapper(
                    active_X.astype(float), W_c, H_c, G_L, G_U, P_L, P_U, config=config, max_iters=100, effort=3
                )
                if f_boost < best_f:
                     best_f = f_boost
                     population[0] = [best_f, (W_boost, H_boost)]
                     if current_phase == 'DIRECT': best_W, best_H = W_boost, H_boost
                     else: best_W, best_H = H_boost.T, W_boost.T
                     
                     if best_f < global_best_f:
                        global_best_f = best_f
                        global_best_W, global_best_H = best_W.copy(), best_H.copy()
                        
                curr_mut = max(0.05, curr_mut * 0.9)
                curr_tourn = min(8, curr_tourn + 1)
            else:
                stagnation_counter += 1
                curr_mut = min(0.7, curr_mut * 1.1)
                if stagnation_counter % 3 == 0: curr_tourn = max(2, curr_tourn - 1)

        iters_in_phase += 1
        iteration += 1

        if global_best_f == 0:
            break

        # E. Restart Logique (ADAPTIVE)
        time_since_last_improv = current_time - last_improvement_time
        
        if time_since_last_improv > restart_threshold:
            restart_count += 1
            restart_threshold = max(5.0, restart_threshold * 0.8)
            
            population = apply_smart_restart(
                best_W, best_H, best_f, N, X, LW, UW, LH, UH, 
                seen_hashes, current_phase, restart_count, population,
                config=config
            )
            best_f = population[0][0]
            if current_phase == 'DIRECT':
                best_W, best_H = population[0][1]
            else:
                H_T, W_T = population[0][1]
                best_W, best_H = W_T.T, H_T.T
            
            last_improvement_time = time.time()
            stagnation_counter = 0
            curr_mut = mutation_rate; curr_tourn = tournament_size
            min_diff_percent = 0.005 

        # --- DATA RECORDING (For Final Plot) ---
        trace_iter.append(iteration)
        trace_best.append(best_f)
        trace_avg.append(np.mean([p[0] for p in population]))

    # --- 3. FIN ---
    remaining = TIME_LIMIT - (time.time() - start_time)
    
    final_W, final_H, final_f = global_best_W, global_best_H, global_best_f

    if remaining > 1.0:
        final_W, final_H, final_f = optimize_alternating_wrapper(
            X_f, final_W, final_H, LW, UW, LH, UH, config=config, max_iters=2000, effort=3, time_limit=remaining
        )
    
    if final_f < global_best_f:
        global_best_W, global_best_H, global_best_f = final_W, final_H, final_f

    return global_best_W, global_best_H, global_best_f