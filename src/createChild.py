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

def generateNewGeneration(seen_hashes, population, num_child, X, LW, UW, LH, UH, mode_opti, start, TIME_LIMIT, tournament_size):
    """
    Generates a new generation of children using C++ batch processing.
    It performs selection, crossover, and mutation in parallel.
    """
    if USE_CPP_BATCH:
        Pop_W = [p[1][0].astype(np.int32) for p in population]
        Pop_H = [p[1][1].astype(np.int32) for p in population]
        Pop_Fitness = [float(p[0]) for p in population]
        
        current_cpp_seed = np.random.randint(0, 2**31-1)

        # Appel optimisé au C++
        raw_results = generate_children_batch(
            X.astype(float),
            Pop_W, Pop_H, Pop_Fitness,
            int(num_child),
            int(tournament_size),
            int(LW), int(UW), int(LH), int(UH),
            str(mode_opti),
            int(current_cpp_seed)
        )
        
        children = []
        for W_res, H_res, f_res, p1_idx, p2_idx, d1, d2 in raw_results:
            child_hash = (W_res.tobytes(), H_res.tobytes())
            if child_hash not in seen_hashes:
                seen_hashes.add(child_hash)
                children.append([f_res, (W_res, H_res), p1_idx, p2_idx, d1, d2])

        return children

    return []