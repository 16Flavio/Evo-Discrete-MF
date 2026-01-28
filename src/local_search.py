import numpy as np

try:
    from .fast_solver import optimize_h, optimize_alternating
    USE_CPP = True
except ImportError:
    print("Warning: Module C++ non trouvé. Veuillez compiler avec setup.py. (optimize_h, optimize_alternating)")
    USE_CPP = False

def optimize_alternating_wrapper(X, W_init, H_init, LW, UW, LH, UH, config=None, max_iters=10, time_limit=3600.0):
    """
    Wrapper for the C++ alternating optimization function.
    Optimizes W and H to minimize ||X - WH||^2.
    """
    if USE_CPP:
        mode_opti = ""
        if config and config.factorization_mode == "IMF":
            mode_opti = "IMF"
        elif config and config.factorization_mode == "BMF":
            mode_opti = "BMF"
        elif config and config.factorization_mode == "RELU":
            mode_opti = "RELU"
        W_opt, H_opt, f = optimize_alternating(
            X.astype(float), 
            W_init.astype(np.int32), 
            H_init.astype(np.int32), 
            int(LW), int(UW), int(LH), int(UH), 
            int(max_iters),
            float(time_limit),
            str(mode_opti)
        )
        return W_opt, H_opt, f
    else:
        return W_init, H_init, float('inf') 

def optimizeHforW(X, W, H_init, LW, UW, LH, UH, config=None):
    """
    Optimizes H for a fixed W.
    Uses the C++ implementation if available.
    """
    if USE_CPP:
        mode_opti = ""
        if config and config.factorization_mode == "IMF":
            mode_opti = "IMF"
        elif config and config.factorization_mode == "BMF":
            mode_opti = "BMF"
        elif config and config.factorization_mode == "RELU":
            mode_opti = "RELU"
        X_d = X.astype(float)
        W_d = W.astype(float)
        H_opt, f = optimize_h(X_d, W_d, int(LW), int(UW), int(LH), int(UH), str(mode_opti))
        return H_opt, f
    else:
        return H_init, float('inf')