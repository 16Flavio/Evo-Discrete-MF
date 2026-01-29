import time
import numpy as np
from .local_search import optimize_alternating_wrapper

try:
    from .fast_solver import align_parents_cpp, generate_children_batch
    USE_CPP_BATCH = True
except ImportError:
    print("Warning: Module C++ non trouvé. Veuillez compiler avec setup.py. (align_parents_cpp, generate_children_batch)")
    USE_CPP_BATCH = False

def align_parents(W1, W2):
    """
    Aligns the columns of W2 to match W1 as closely as possible.
    Uses the C++ implementation if available.
    """
    if USE_CPP_BATCH:
        return align_parents_cpp(W1.astype(np.int32), W2.astype(np.int32))
    else:
        return W2

def generateNewGeneration(seen_hashes, population, num_child, X, LW, UW, LH, UH, start, TIME_LIMIT, tournament_size, mutation_rate, config=None):
    """
    Generates a new generation of children using C++ batch processing.
    It performs selection, crossover, and mutation in parallel.
    """
    if USE_CPP_BATCH:
        Pop_W = [p[1][0].astype(np.int32) for p in population]
        Pop_H = [p[1][1].astype(np.int32) for p in population]
        Pop_Fitness = [float(p[0]) for p in population]
        
        crossover_mode_int = 0
        if config and config.crossover_type == "UNIFORM":
            crossover_mode_int = 1
        elif config and config.crossover_type == "MEAN":
            crossover_mode_int = 2

        mutation_mode_int = 0
        if config and config.mutation_type == "SWAP":
            mutation_mode_int = 1
        elif config and config.mutation_type == "GREEDY":
            mutation_mode_int = 2
        elif config and config.mutation_type == "NOISE":
            mutation_mode_int = 3
        elif config and config.mutation_type == "NONE":
            mutation_mode_int = 4

        mode_opti = ""
        if config and config.factorization_mode == "IMF":
            mode_opti = "IMF"
        elif config and config.factorization_mode == "BMF":
            mode_opti = "BMF"
        elif config and config.factorization_mode == "RELU":
            mode_opti = "RELU"

        current_cpp_seed = np.random.randint(0, 2**31-1)

        # Appel optimisé au C++
        raw_results = generate_children_batch(
            X.astype(float),
            Pop_W, Pop_H, Pop_Fitness,
            int(num_child),
            int(tournament_size),
            float(mutation_rate),
            int(LW), int(UW), int(LH), int(UH),
            int(crossover_mode_int),
            int(mutation_mode_int),
            str(mode_opti),
            int(current_cpp_seed)
        )
        
        children = []
        for W_res, H_res, f_res, p1_idx, p2_idx, d1, d2 in raw_results:
            child_hash = (W_res.tobytes(), H_res.tobytes())
            if child_hash not in seen_hashes:
                seen_hashes.add(child_hash)
                children.append([f_res, (W_res, H_res), p1_idx, p2_idx, d1, d2])
        
        children.sort(key=lambda x: x[0]) 
        if children:
            best_childs = children[:1]
            for i, best_child in enumerate(best_childs):
                fitness, (W, H), p1, p2, d1, d2 = best_child
                
                W_opt, H_opt, f_opt = optimize_alternating_wrapper(
                    X, 
                    W, H, 
                    LW, UW, LH, UH,
                    max_iters=100,
                    config=config
                )
                
                if f_opt < fitness:
                    children[i] = [f_opt, (W_opt, H_opt), p1, p2, d1, d2]

        return children

    return []