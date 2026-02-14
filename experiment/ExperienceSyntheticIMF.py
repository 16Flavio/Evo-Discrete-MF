import numpy as np
import math
import time
from datetime import datetime
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from src.solver import metaheuristic
except ImportError:
    def metaheuristic(X, r, LW, UW, LH, UH, method, TIME_LIMIT=2.0, N=50, debug_mode=False):
        m, n = X.shape
        return np.zeros((m, r)), np.zeros((r, n)), 0

def method_from_quantization_aware(X, r, choixinit="rand", alpha=-16, beta=15):
    X       = np.asarray(X, dtype=np.float64)
    m, n    = X.shape
    rng     = np.random.default_rng() 
    maxiter = 10000
    eps     = 1e-16

    def project(Z):
        return np.clip(np.rint(Z), math.ceil(alpha), math.floor(beta))

    def soft_thresholding(Z, lam):
        if lam == 0: return Z
        return np.sign(Z) * np.maximum(np.abs(Z) - lam, 0.0)

    # Initialisation
    if choixinit.lower() == "svd":
        Ufull, s, Vtfull = np.linalg.svd(X, full_matrices=False)
        real_r = min(r, len(s))
        U_tilde = Ufull[:, :real_r]
        s_vec   = s[:real_r]
        V_tilde = Vtfull[:real_r, :].T
        s_sqrt  = np.sqrt(s_vec)

        U = np.zeros((m, r))
        V = np.zeros((n, r))
        
        U[:, :real_r] = U_tilde * s_sqrt[None, :]
        V[:, :real_r] = V_tilde * s_sqrt[None, :]
        
        U = project(U)
        V = project(V)

    if choixinit.lower() == "rand":
        U = rng.integers(alpha, beta + 1, size=(m, r)).astype(np.float64)
        V = rng.integers(alpha, beta + 1, size=(n, r)).astype(np.float64)
        
    # BCD
    for _ in range(maxiter):
        # Update U
        A = X @ V 
        B = V.T @ V
        U_new = U.copy()

        for k in range(r):
            idx   = [j for j in range(r) if j != k]
            term1 = A[:, k:k+1]
            if len(idx) > 0:
                term2 = U_new[:, idx] @ B[np.ix_(idx, [k])]
            else:
                term2 = 0
            num   = soft_thresholding(term1 - term2, 0)
            den   = B[k, k]
            U_new[:, k:k+1] = project((num + eps) / (den + eps))
        diffU = np.sum(np.abs(U-U_new))
        U = U_new

        # Update V
        A = X.T @ U
        B = U.T @ U
        V_new = V.copy()
        for k in range(r):
            idx   = [j for j in range(r) if j != k]
            term1 = A[:, k:k+1]
            if len(idx) > 0:
                term2 = V_new[:, idx] @ B[np.ix_(idx, [k])]
            else:
                term2 = 0
            num   = soft_thresholding(term1 - term2,0)
            den   = B[k, k]
            V_new[:, k:k+1] = project((num + eps) / (den + eps))
        diffV = np.sum(np.abs(V-V_new))
        V = V_new

        if diffU == 0 and diffV == 0:
            break

    return U, V

def my_method(X, r, alpha=-16, beta=15, budgettemps=2.0):
    LW = int(math.ceil(alpha))
    UW = int(math.floor(beta))
    LH = LW
    UH = UW
    
    W, H, f = metaheuristic(X, r, LW, UW, LH, UH, 'IMF', TIME_LIMIT=budgettemps, N=50, debug_mode=False)
    return W, H 
RESULT_FILE = "results_experiment_multirun.csv"

def log(msg):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)

def load_completed_runs(filename):
    completed = {} 
    if not os.path.exists(filename):
        with open(filename, "w") as f:
            f.write("m,n,r,a,b,rep,method,success,runs,time_taken\n")
        return completed
    
    with open(filename, "r") as f:
        lines = f.readlines()[1:]
        for line in lines:
            if not line.strip(): continue
            parts = line.split(",")
            config = tuple(int(x) for x in parts[:5])
            rep = int(parts[5])
            method = parts[6]
            
            succ = int(parts[7])
            runs = int(parts[8])
            
            completed[(config, rep, method)] = (succ, runs)
            
    return completed

def save_single_run(filename, config, rep, method_name, stats, time_taken):
    m, n, r, a, b = config
    succ, runs = stats
    
    with open(filename, "a") as f:
        f.write(f"{m},{n},{r},{a},{b},{rep},{method_name},{succ},{runs},{time_taken:.2f}\n")

T_tot = 600.0
T_run = 60.0
NUM_REPS = 5
tol = 1e-12

settings = [
    (10, 10, 3,  -3,  3),
    (10, 10, 3,  -5,  5),
    (20, 20, 5,  -5,  5),
    (20, 20, 5, -16, 15),
    (50, 50, 5, -16, 15),
]

if __name__ == "__main__":

    done_runs = load_completed_runs(RESULT_FILE)
    log(f"Expérience démarrée (Multi-run mode). {len(done_runs)} entrées déjà trouvées.")

    for config in settings:
        (m, n, r, a, b) = config
        log(f"=== Config: (m={m}, n={n}, r={r}, range=[{a},{b}]) ===")
        
        stats_summary = {
            "QMF_RAND": {"succ": [], "runs": []},
            "QMF_SVD":  {"succ": [], "runs": []},
            "OURS":     {"succ": [], "runs": []}
        }

        for rep in range(NUM_REPS):
            rng = np.random.default_rng(2026 + rep)
            W_star = rng.integers(a, b + 1, size=(m, r))
            H_star = rng.integers(a, b + 1, size=(r, n))
            X = W_star @ H_star

            log(f"  -- Repetition {rep+1}/{NUM_REPS} --")

            # ---------------------------------------------------------
            # 1. QMF (Random Init) - Boucle de 10 minutes
            # ---------------------------------------------------------
            method_key = "QMF_RAND"
            if (config, rep, method_key) in done_runs:
                prev_succ, prev_runs = done_runs[(config, rep, method_key)]
                stats_summary[method_key]["succ"].append(prev_succ)
                stats_summary[method_key]["runs"].append(prev_runs)
                log(f"    > {method_key} déjà fait. Chargé (Succ: {prev_succ}/{prev_runs}).")
            else:
                start_t = time.time()
                qmf_succ, qmf_runs = 0, 0
                t0 = time.time()
                while time.time() - t0 < T_tot:
                    U, V = method_from_quantization_aware(X, r, choixinit="rand", alpha=a, beta=b)
                    qmf_runs += 1
                    if np.linalg.norm(X - U @ V.T, "fro") ** 2 <= tol:
                        qmf_succ += 1
                    
                    if qmf_runs % 500 == 0:
                        print(f"      [{method_key}] Runs: {qmf_runs}, Succ: {qmf_succ}...", end="\r")

                duration = time.time() - start_t
                save_single_run(RESULT_FILE, config, rep, method_key, (qmf_succ, qmf_runs), duration)
                log(f"    > {method_key} Saved. ({qmf_succ}/{qmf_runs} en {duration:.1f}s)")
                
                stats_summary[method_key]["succ"].append(qmf_succ)
                stats_summary[method_key]["runs"].append(qmf_runs)

                done_runs[(config, rep, method_key)] = (qmf_succ, qmf_runs)

            # ---------------------------------------------------------
            # 2. QMF (SVD Init) - UN SEUL RUN (One shot)
            # ---------------------------------------------------------
            method_key = "QMF_SVD"
            if (config, rep, method_key) in done_runs:
                prev_succ, prev_runs = done_runs[(config, rep, method_key)]
                stats_summary[method_key]["succ"].append(prev_succ)
                stats_summary[method_key]["runs"].append(prev_runs)
                log(f"    > {method_key} déjà fait. Chargé (Succ: {prev_succ}).")
            else:
                start_t = time.time()
                U, V = method_from_quantization_aware(X, r, choixinit="svd", alpha=a, beta=b)
                
                is_success = 1 if np.linalg.norm(X - U @ V.T, "fro") ** 2 <= tol else 0
                svd_runs = 1
                
                duration = time.time() - start_t
                save_single_run(RESULT_FILE, config, rep, method_key, (is_success, svd_runs), duration)
                log(f"    > {method_key} Saved. (Success: {is_success})")

                stats_summary[method_key]["succ"].append(is_success)
                stats_summary[method_key]["runs"].append(svd_runs)
                done_runs[(config, rep, method_key)] = (is_success, svd_runs)

            # ---------------------------------------------------------
            # 3. OURS (Metaheuristic) - Boucle de 10 minutes
            # ---------------------------------------------------------
            method_key = "OURS"
            if (config, rep, method_key) in done_runs:
                prev_succ, prev_runs = done_runs[(config, rep, method_key)]
                stats_summary[method_key]["succ"].append(prev_succ)
                stats_summary[method_key]["runs"].append(prev_runs)
                log(f"    > {method_key} déjà fait. Chargé (Succ: {prev_succ}/{prev_runs}).")
            else:
                start_t = time.time()
                ours_succ, ours_runs = 0, 0
                t0 = time.time()
                while time.time() - t0 < (T_tot):
                    W, H = my_method(X, r, alpha=a, beta=b, budgettemps=min(T_run, T_tot-(time.time() - t0)))
                    ours_runs += 1
                    if np.linalg.norm(X - W @ H, "fro") ** 2 <= tol:
                        ours_succ += 1
                    
                    if ours_runs % 500 == 0:
                        print(f"      [{method_key}] Runs: {ours_runs}, Succ: {ours_succ}...", end="\r")

                duration = time.time() - start_t
                save_single_run(RESULT_FILE, config, rep, method_key, (ours_succ, ours_runs), duration)
                log(f"    > {method_key} Saved. ({ours_succ}/{ours_runs} en {duration:.1f}s)")

                stats_summary[method_key]["succ"].append(ours_succ)
                stats_summary[method_key]["runs"].append(ours_runs)
                done_runs[(config, rep, method_key)] = (ours_succ, ours_runs)

        log(f"\n--- RÉSULTATS MOYENS (sur {NUM_REPS} reps de 10min) pour config {config} ---")
        for m_key in ["QMF_RAND", "OURS", "QMF_SVD"]:
            s_list = stats_summary[m_key]["succ"]
            r_list = stats_summary[m_key]["runs"]
            
            if len(r_list) > 0:
                avg_runs = np.mean(r_list)
                
                rates = [s/r if r > 0 else 0 for s, r in zip(s_list, r_list)]
                print(rates)
                avg_rate = np.mean(rates) * 100.0
                
                print(f"  METHOD {m_key:<10} | Avg Runs: {avg_runs:8.1f} | Avg Success Rate: {avg_rate:6.2f}%")
            else:
                print(f"  METHOD {m_key:<10} | (Données non disponibles dans cette session)")
        print("-" * 60)

    log("Toutes les expériences sont terminées.")