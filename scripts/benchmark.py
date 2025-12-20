import os
import subprocess
import statistics
import sys
import csv
import glob

# ================= CONFIGURATION DU BENCHMARK =================

# Détection automatique des chemins relatifs
CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_SCRIPT_DIR)

# Chemins
SCRIPT_PATH = os.path.join(PROJECT_ROOT, "main.py")
IMF_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "imf_matrix")
OUTPUT_BASE_DIR = os.path.join(PROJECT_ROOT, "results")
CSV_FILENAME = "benchmark_summary_IMF.csv"

PYTHON_EXEC = sys.executable 
NB_RUNS = 30          # Réduit pour tester rapidement (mettre 30 pour prod)
TIME_LIMIT = 120.0    # Temps limite PAR FICHIER
RANK = 10            # Rang de factorisation fixe

# LISTE DES SEEDS (Common Random Numbers)
SEEDS = [i + 1000 for i in range(NB_RUNS)]

# Configurations à tester (Celles de votre fichier original)
CONFIGURATIONS = {
    # --- 1. INITIALISATION COMBINATIONS (2^4 - 1 = 15 cases) ---
    # 4 Actives (Full)
    # "Init: All (Ref)": [],

    # # 3 Actives (1 Disabled)
    # "Init: No SVD": ["--no-svd"],
    "Init: No KMeans": ["--no-kmeans"],
    # "Init: No NMF": ["--no-nmf"],
    "Init: No Greedy": ["--no-greedy"],

    # 2 Actives (2 Disabled)
    "Init: SVD + KMeans": ["--no-nmf", "--no-greedy"],
    # "Init: SVD + NMF": ["--no-kmeans", "--no-greedy"],
    # "Init: SVD + Greedy": ["--no-kmeans", "--no-nmf"],
    # "Init: KMeans + NMF": ["--no-svd", "--no-greedy"],
    # "Init: KMeans + Greedy": ["--no-svd", "--no-nmf"],
    # "Init: NMF + Greedy": ["--no-svd", "--no-kmeans"],

    # 1 Active (3 Disabled)
    # "Init: Only SVD": ["--no-kmeans", "--no-nmf", "--no-greedy"],
    # "Init: Only KMeans": ["--no-svd", "--no-nmf", "--no-greedy"],
    # "Init: Only NMF": ["--no-svd", "--no-kmeans", "--no-greedy"],
    # "Init: Only Greedy": ["--no-svd", "--no-kmeans", "--no-nmf"],

    # # --- 2. OTHER ABLATIONS & PARAMETERS ---
    # "Param: No Transpose": ["--no-transpose"],
    
    # # Restart Strategies
    # "Restart: Simple": ["--restart-mode", "SIMPLE"],
    
    # # Crossover Strategies
    # "Cross: Mean": ["--crossover", "MEAN"],
    # "Cross: Uniform": ["--crossover", "UNIFORM"],
    
    # # Mutation Strategies
    # "Mut: Swap": ["--mutation-type", "SWAP"],
    # "Mut: Greedy": ["--mutation-type", "GREEDY"],
    # "Mut: Noise": ["--mutation-type", "NOISE"]
}

# ==============================================================

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_score_from_file(filepath):
    """Lit le score (erreur) depuis le fichier de sortie du solver."""
    try:
        with open(filepath, 'r') as f:
            line = f.readline().strip()
            if not line: return float('inf')
            return float(line)
    except Exception:
        return float('inf')

def get_imf_datasets():
    """Récupère tous les fichiers .txt dans data/imf_matrix"""
    pattern = os.path.join(IMF_DATA_DIR, "*.txt")
    files = glob.glob(pattern)
    files.sort() # Tri pour ordre constant
    return files

def run_global_benchmark():
    datasets = get_imf_datasets()
    
    if not datasets:
        print(f"ERREUR: Aucun fichier .txt trouvé dans {IMF_DATA_DIR}")
        return

    print(f"\n{'='*60}")
    print(f"BENCHMARK GLOBAL IMF")
    print(f"Datasets trouvés ({len(datasets)}) :")
    for d in datasets:
        print(f"  - {os.path.basename(d)}")
    print(f"{'='*60}\n")

    result_csv_path = os.path.join(OUTPUT_BASE_DIR, CSV_FILENAME)
    temp_output_dir = os.path.join(OUTPUT_BASE_DIR, "benchmark_temp")
    ensure_dir(temp_output_dir)

    # --- Gestion reprise (Resume) ---
    processed_configs = set()
    file_exists = os.path.exists(result_csv_path)

    if file_exists:
        try:
            with open(result_csv_path, 'r') as f:
                reader = csv.reader(f, delimiter=';')
                for row in reader:
                    if row: processed_configs.add(row[0]) 
        except Exception:
            processed_configs = set()

    open_mode = 'a' if file_exists else 'w'
    
    # Header Console
    header_console = f"{'CONFIGURATION':<25} | {'SCORE TOTAL':<12} | {'MOYENNE':<10} | {'STD DEV':<10}"
    print("-" * len(header_console))
    print(header_console)
    print("-" * len(header_console))

    with open(result_csv_path, mode=open_mode, newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=';')

        if not file_exists:
            writer.writerow(["CONFIGURATION", "SCORE_TOTAL_MOYEN", "MEILLEUR_TOTAL", "PIRE_TOTAL", "STD_DEV"])

        for config_name, config_args in CONFIGURATIONS.items():
            
            if config_name in processed_configs:
                print(f"{config_name:<25} | DÉJÀ FAIT (Skip)")
                continue

            # Liste pour stocker le Score Total de chaque Seed
            # seed_total_scores[i] = Somme des scores de tous les fichiers pour la seed i
            seed_total_scores = [] 
            
            print(f"Run: {config_name:<20}", end="", flush=True)

            for i, seed in enumerate(SEEDS):
                current_seed_total_score = 0
                error_occurred = False

                # Pour cette seed, on lance le solver sur TOUS les fichiers
                for input_file in datasets:
                    temp_output = os.path.join(temp_output_dir, f"out_{i}.txt")
                    if os.path.exists(temp_output): os.remove(temp_output)

                    cmd = [
                        PYTHON_EXEC, SCRIPT_PATH,
                        "--input", input_file,
                        "--output", temp_output,
                        "--rank", str(RANK),
                        "--time_limit", str(TIME_LIMIT),
                        "-n", "61",
                        "-s", "8",
                        "-m", "0.383",
                        "--seed", str(seed),
                        "--factorization-mode", "IMF" # Forcé en IMF
                    ] + config_args

                    try:
                        # Capture output to avoid spamming console
                        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                        score = get_score_from_file(temp_output)
                        
                        if score == float('inf'):
                            error_occurred = True
                            break
                        
                        current_seed_total_score += score

                    except Exception:
                        error_occurred = True
                        break
                
                if not error_occurred:
                    seed_total_scores.append(current_seed_total_score)
                    print(".", end="", flush=True)
                else:
                    seed_total_scores.append(float('inf'))
                    print("E", end="", flush=True)

            # Analyse des résultats agrégés
            valid_totals = [s for s in seed_total_scores if s != float('inf')]
            
            if valid_totals:
                avg_total = statistics.mean(valid_totals)
                best_total = min(valid_totals)
                worst_total = max(valid_totals)
                std_dev = statistics.stdev(valid_totals) if len(valid_totals) > 1 else 0.0
                
                print(f"\r{config_name:<25} | {avg_total:<12.0f} | {avg_total:<10.0f} | {std_dev:<10.2f}")
                writer.writerow([config_name, f"{avg_total:.2f}", int(best_total), int(worst_total), f"{std_dev:.2f}"])
            else:
                print(f"\r{config_name:<25} | {'FAIL':<12} | - | -")
                writer.writerow([config_name, "FAIL", "-", "-", "-"])
            
            csv_file.flush()

if __name__ == "__main__":
    run_global_benchmark()