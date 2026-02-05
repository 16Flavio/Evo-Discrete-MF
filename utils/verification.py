import numpy as np

def fobj(X,W,H):
    """
    Calculates the Frobenius norm of the error (X - WH).
    Returns the squared norm rounded to the nearest integer.
    """
    f = np.linalg.norm(X-W@H,'fro')**2
    return round(f)

def fobj_bmf(X,W,H):
    """
    Calculates the objective function for BMF (Boolean Matrix Factorization).
    Uses min(1, WH) to simulate boolean multiplication.
    """
    f = np.linalg.norm(X-np.minimum(1,W@H),'fro')**2
    return round(f)

def fobj_relu(X,W,H):
    """
    Calculates the objective function with ReLU constraint.
    Uses max(0, WH).
    """
    f = np.linalg.norm(X-np.maximum(0,W@H),'fro')**2
    return round(f)

def solutionIsFeasible(W,H,r,LW,UW,LH,UH):
    """
    Checks if the solution (W, H) is feasible respecting the bounds and dimensions.
    """
    if W.shape[1] != r or H.shape[0] != r:
        return False
    Wi = np.issubdtype(W.dtype, np.integer) or (np.issubdtype(W.dtype, np.floating) and np.all(np.isfinite(W)) and np.all(W == np.floor(W)))
    Hi = np.issubdtype(H.dtype, np.integer) or (np.issubdtype(H.dtype, np.floating) and np.all(np.isfinite(H)) and np.all(H == np.floor(H)))
    if not (Wi and Hi):
      return False
    return np.all((W >= LW) & (W <= UW)) and np.all((H >= LH) & (H <= UH))