import os
import subprocess
import statistics
import sys
import csv

# ================= CONFIGURATION DU BENCHMARK =================

# Détection automatique des chemins relatifs
CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_SCRIPT_DIR)

# Chemins absolus reconstruits
SCRIPT_PATH = os.path.join(PROJECT_ROOT, "main.py")
INPUT_FILE = os.path.join(PROJECT_ROOT, "data", "imf_matrix", "large_matrix.txt") 
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "imf_matrix", "benchmark_temp")

PYTHON_EXEC = sys.executable 
NB_RUNS = 30
TIME_LIMIT = 60.0
RANK = 10

# LISTE DES SEEDS (Common Random Numbers)
# On fixe une liste de seeds pour que chaque config soit testée sur les mêmes bases aléatoires.
SEEDS = [i + 1000 for i in range(NB_RUNS)] # [1000, 1001, ..., 1019]

# Scénarios de benchmark
SCENARIOS = [
    {
        "name": "IMF_Benchmark",
        "input_file": os.path.join(PROJECT_ROOT, "data", "imf_matrix", "large_matrix.txt"),
        "mode": "IMF",
        "csv_name": "benchmark_init_IMF.csv"
    },
    {
        "name": "BMF_Benchmark",
        "input_file": os.path.join(PROJECT_ROOT, "data", "bmf_matrix", "binarizedCBCL.txt"),
        "mode": "BMF",
        "csv_name": "benchmark_init_BMF.csv"
    }
]

# Configurations à tester
CONFIGURATIONS = {
    # --- 1. INITIALISATION COMBINATIONS (2^4 - 1 = 15 cases) ---
    # 4 Actives (Full)
    "Init: All (Ref)": [],

    # 3 Actives (1 Disabled)
    "Init: No SVD": ["--no-svd"],
    "Init: No KMeans": ["--no-kmeans"],
    "Init: No NMF": ["--no-nmf"],
    "Init: No Greedy": ["--no-greedy"],

    # 2 Actives (2 Disabled)
    "Init: SVD + KMeans": ["--no-nmf", "--no-greedy"],
    "Init: SVD + NMF": ["--no-kmeans", "--no-greedy"],
    "Init: SVD + Greedy": ["--no-kmeans", "--no-nmf"],
    "Init: KMeans + NMF": ["--no-svd", "--no-greedy"],
    "Init: KMeans + Greedy": ["--no-svd", "--no-nmf"],
    "Init: NMF + Greedy": ["--no-svd", "--no-kmeans"],

    # 1 Active (3 Disabled)
    "Init: Only SVD": ["--no-kmeans", "--no-nmf", "--no-greedy"],
    "Init: Only KMeans": ["--no-svd", "--no-nmf", "--no-greedy"],
    "Init: Only NMF": ["--no-svd", "--no-kmeans", "--no-greedy"],
    "Init: Only Greedy": ["--no-svd", "--no-kmeans", "--no-nmf"],

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
    try:
        with open(filepath, 'r') as f:
            line = f.readline().strip()
            if not line: return float('inf')
            return float(line)
    except Exception:
        return float('inf')

def run_single_benchmark(scenario):
    scen_name = scenario["name"]
    input_file = scenario["input_file"]
    mode = scenario["mode"]
    csv_filename = scenario["csv_name"]
    
    print(f"\n{'='*60}")
    print(f"LANCEMENT DU SCÉNARIO : {scen_name}")
    print(f"Mode          : {mode}")
    print(f"Fichier Input : {os.path.basename(input_file)}")
    print(f"Output CSV    : {csv_filename}")
    print(f"{'='*60}\n")

    if not os.path.exists(input_file):
        print(f"ERREUR: Input introuvable. Skip.")
        return

    output_dir_temp = os.path.join(PROJECT_ROOT, "results", "benchmark_temp", mode)
    ensure_dir(output_dir_temp)
    result_csv_path = os.path.join(PROJECT_ROOT, "results", csv_filename)

    # --- 1. DÉTECTION DE LA REPRISE (RESUME) ---
    processed_configs = set()
    file_exists = os.path.exists(result_csv_path)

    if file_exists:
        print(f"-> Fichier existant détecté. Lecture des configs déjà faites...")
        try:
            with open(result_csv_path, 'r') as f:
                reader = csv.reader(f, delimiter=';')
                for row in reader:
                    if row: processed_configs.add(row[0]) 
        except Exception as e:
            print(f"-> Erreur lecture CSV ({e}). On repart de zéro.")
            processed_configs = set()

    # --- 2. OUVERTURE EN MODE APPEND ('a') ---
    open_mode = 'a' if file_exists else 'w'
    
    # Header Console
    header_console = f"{'CONFIGURATION':<25} | {'MOYENNE':<10} | {'MEILLEUR':<10} | {'PIRE':<10} | {'STD DEV':<10}"
    print("-" * len(header_console))
    print(header_console)
    print("-" * len(header_console))

    # On ouvre le fichier et on le garde ouvert
    with open(result_csv_path, mode=open_mode, newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=';')

        # Si nouveau fichier, on écrit l'en-tête CSV
        if not file_exists:
            writer.writerow(["CONFIGURATION", "MOYENNE", "MEILLEUR", "PIRE", "STD DEV"])

        for config_name, config_args in CONFIGURATIONS.items():
            
            if config_name in processed_configs:
                print(f"{config_name:<25} | DÉJÀ FAIT (Skip)")
                continue

            scores = []
            print(f"Test: {config_name:<19}", end="", flush=True)

            for i, seed in enumerate(SEEDS):
                temp_output = os.path.join(output_dir_temp, f"out_{i}.txt")
                if os.path.exists(temp_output): os.remove(temp_output)

                cmd = [
                    PYTHON_EXEC, SCRIPT_PATH,
                    "--input", input_file,
                    "--output", temp_output,
                    "--rank", str(RANK),
                    "--time_limit", str(TIME_LIMIT),
                    "-n", "61",
                    "-s", "4",
                    "-m", "0.383",
                    "--seed", str(seed),
                    "--factorization-mode", mode
                ] + config_args

                try:
                    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                    scores.append(get_score_from_file(temp_output))
                    print(".", end="", flush=True)
                except:
                    print("E", end="", flush=True)
                    scores.append(float('inf'))

            valid_scores = [s for s in scores if s != float('inf')]
            
            if valid_scores:
                avg = statistics.mean(valid_scores)
                best = min(valid_scores)
                worst = max(valid_scores)
                std = statistics.stdev(valid_scores) if len(valid_scores) > 1 else 0.0
                
                print(f"\r{config_name:<25} | {avg:<10.2f} | {best:<10.0f} | {worst:<10.0f} | {std:<10.2f}")
                writer.writerow([config_name, f"{avg:.2f}", int(best), int(worst), f"{std:.2f}"])
            else:
                print(f"\r{config_name:<25} | {'FAIL':<10} | {'-':<10} | {'-':<10} | {'-':<10}")
                writer.writerow([config_name, "FAIL", "-", "-", "-"])
            
            # --- SAUVEGARDE IMMÉDIATE ---
            csv_file.flush() # Force l'écriture sur le disque dur immédiatement

if __name__ == "__main__":
    if not os.path.exists(SCRIPT_PATH):
        print(f"ERREUR: main.py introuvable à {SCRIPT_PATH}")
    else:
        # Boucle sur tous les scénarios définis
        for scenario in SCENARIOS:
            run_single_benchmark(scenario)