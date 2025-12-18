from typing import Tuple
from typing import Tuple
import numpy as np
from sklearn.decomposition import TruncatedSVD,NMF
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder


def SVD_truncated(X: np.array, r: int)->Tuple[np.array, np.array, np.array]:
    """
    Floating point factorization of matrix X into W and H such that X ≈ W @ H
    using Truncated SVD.
    Args:
        X (np.ndarray): Input matrix of size (m, n).
        r (int): Rank of the factorization.
    Returns:
        W (np.ndarray): Matrix of size (m, r).
        H (np.ndarray): Matrix of size (r, n).
        X_approx (np.ndarray): Approximation of X obtained by W @ H.
    """
    # Initialize the model
    svd = TruncatedSVD(n_components=r, random_state=42) # Uses a fixed random state for reproducibility

    # Compute the decomposition
    # The fit_transform(A) method directly returns matrix W
    W = svd.fit_transform(X)

    # The matrix H (C^T) is in svd.components_
    H = svd.components_

    return W, H, W@H


def constrained_kmedoids_factorization(X, r, WL, WU):
    """
    Factorizes X (m*n) into W (m*r) and H (r*n) using K-Medoids
    with constraints on the medoids (W).
    
    MODIFIED: If fewer than 'r' valid candidates [WL, WU] are found,
    the algorithm fills the remaining spots with invalid candidates
    that are truncated (clipped) to respect the bounds.
    
    - W (Medoids): Integer, bounded by [WL, WU].
    - H (Assignments): Integer (binary 0/1).
    
    Args:
        X (np.ndarray): Input matrix (m x n), assumed integer.
        r (int): The rank (number of medoids).
        WL (int): Lower bound for elements of W.
        WU (int): Upper bound for elements of W.
    
    Returns:
        tuple (W, H) or (None, None) if failure.
    """
    
    m, n = X.shape
    
    # --- 1. Medoid Identification and Selection (W) ---
    
    good_candidate_indices = []
    bad_candidate_indices = []
    
    for j in range(n):
        col = X[:, j]
        # Checks if ALL elements of the column are within the bounds
        if np.all(col >= WL) and np.all(col <= WU):
            good_candidate_indices.append(j)
        else:
            bad_candidate_indices.append(j)
            
    # print(f"Found {len(good_candidate_indices)} suitable candidates (respecting bounds).")

    # W is the set of medoids (m x r)
    W = np.zeros((m, r), dtype=X.dtype)
    
    if len(good_candidate_indices) >= r:
        # Case 1: Enough good candidates. Choose r randomly.
        # print(f"Selecting {r} suitable candidates randomly.")
        selected_indices = np.random.choice(good_candidate_indices, r, replace=False)
        W = X[:, selected_indices]
        
    else:
        # Case 2: Not enough good candidates.
        num_good = len(good_candidate_indices)
        num_needed = r - num_good
        
        # print(f"Not enough candidates. Taking the {num_good} suitable candidates.")
        # print(f"Adding {num_needed} unsuitable (truncated) candidates.")
        
        # Checks if there are enough "bad" candidates to fill the gap
        if len(bad_candidate_indices) < num_needed:
            # print(f"Error: Not enough total columns ({n}) to reach r={r}.")
            return None, None
            
        # 1. Add all good candidates
        if num_good > 0:
            W[:, :num_good] = X[:, good_candidate_indices]
            
        # 2. Choose, truncate, and add bad candidates
        selected_bad_indices = np.random.choice(bad_candidate_indices, num_needed, replace=False)
        
        # Retrieve original columns
        W_bad_original = X[:, selected_bad_indices]
        
        # Truncate (clip) columns to respect bounds
        # np.clip preserves dtype (int if X is int)
        W_bad_truncated = np.clip(W_bad_original, WL, WU)
        
        # Add truncated columns to W
        W[:, num_good:] = W_bad_truncated

    # --- 2. Assignment and H Creation ---
    # (This part is unchanged from the original code)
    
    H = np.zeros((r, n), dtype=int)
    
    # print("Assigning points (columns of X) to medoids (columns of W)...")
    
    # Cost (distance) matrix (n x r)
    # Distance between each point (column of X) and each medoid (column of W)
    costs = np.zeros((n, r))
    for j in range(n):
        point = X[:, j] # Point (m x 1 column)
        for k in range(r):
            medoid = W[:, k] # Medoid (m x 1 column)
            
            # Using squared L2 (Euclidean) distance
            dist_sq = np.sum((point - medoid)**2)
            costs[j, k] = dist_sq
            
    # Assignment: find the index of the closest medoid for each point
    assignments = np.argmin(costs, axis=1) # (size n)
    
    # Create H (binary)
    for j in range(n):
        k = assignments[j] # 'k' is the medoid (0 to r-1) assigned to point 'j'
        H[k, j] = 1 # Binary assignment

    return W, H


def factorize_kmeans_integer(A, r):
    """
    Factorizes a matrix A (m*n) into W (m*r) and H (n*r)
    using K-Means, with W and H being integers.
    
    A ≈ W @ H.T
    
    Args:
        A (np.ndarray): Input matrix (m x n).
        r (int): Rank of factorization (number of clusters).
        
    Returns:
        tuple:
            - W_integer (np.ndarray): Assignment matrix (m x r), integer (0/1).
            - H_integer (np.ndarray): Transposed centroids matrix (n x r), 
                                      rounded to integer.
    """
    
    # 1. Apply K-Means to the rows of A
    # n_init='auto' to suppress future warnings
    kmeans = KMeans(n_clusters=r, random_state=42, n_init='auto')
    
    # 'labels' is a vector of size 'm' indicating the cluster of each row
    labels = kmeans.fit_predict(A) 
    
    # 'centroids' is the matrix C (r x n)
    centroids = kmeans.cluster_centers_ 
    
    
    # 2. Create matrix W (m x r) - Already integer
    
    # Reshape 'labels' into a column (m x 1) for the encoder
    labels_reshaped = labels.reshape(-1, 1)
    
    # Create the one-hot encoder
    encoder = OneHotEncoder(sparse_output=False, categories='auto')
    
    # W is the one-hot assignment matrix (m x r)
    W_integer = encoder.fit_transform(labels_reshaped).astype(int)
    
    
    # 3. Create matrix H (n x r) - Rounded to integer
    
    # H = C.T (n x r)
    H_float = centroids.T
    
    # Round H to the nearest integer
    H_integer = np.round(H_float).astype(int)
    
    return W_integer, H_integer.T


def ExtendedNMF(X: np.array, r: int, max_iter: int = 5000, tol: float = 1e-4, random_state: int =42)->Tuple[np.array, np.array,np.array]:
    """
    Non-negative floating point factorization of matrix X into W and H such that X ≈ W @ H
    using NMF (Non-negative Matrix Factorization).
    Args:
        X (np.ndarray): Input matrix of size (m, n).
        r (int): Rank of factorization.
        max_iter (int): Maximum number of iterations.
        tol (float): Tolerance for convergence.
    Returns:
        W (np.ndarray): Matrix of size (m, r).
        H (np.ndarray): Matrix of size (r, n).
    """
    min_value = np.min(X)

    if min_value < 0:
        X_shifted = X - min_value
        model = NMF(n_components=r, init='nndsvdar', solver= 'mu', max_iter=max_iter,random_state=random_state, tol=tol)
        W = model.fit_transform(X_shifted)
        H = model.components_
    else:
        model = NMF(n_components=r, init='nndsvdar', solver= 'mu', max_iter=max_iter,random_state=random_state, tol=tol)
        W = model.fit_transform(X)
        H = model.components_

    return W, H, W@H

def round_matrix(X: np.array, min_val: int, max_val: int)->np.array:
    """
    Rounds elements of matrix X to the nearest integers, ensuring they stay within allowed limits.
    Args:
        X (np.ndarray): Input matrix.
        min_val (int): Minimum allowed value.
        max_val (int): Maximum allowed value.
    Returns:
        np.ndarray: Rounded matrix.
    """
    X_rounded = np.rint(X).astype(int)
    X_clipped = np.clip(X_rounded, min_val, max_val)
    return X_clipped