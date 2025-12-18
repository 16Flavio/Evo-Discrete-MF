import optuna
import numpy as np
import sys
import os
import glob
import importlib

# --- GESTION ROBUSTE DES IMPORTS ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir) 
package_name = os.path.basename(current_dir) 

if parent_dir not in sys.path:
    sys.path.append(parent_dir)

try:
    module_name = f"{package_name}.solver"
    solver_module = importlib.import_module(module_name)
    metaheuristic = solver_module.metaheuristic
    print(f"[Info] Solver importé via '{package_name}'.")
except ImportError:
    try:
        if current_dir not in sys.path: sys.path.append(current_dir)
        from solver import metaheuristic
        print("[Info] Solver importé directement.")
    except ImportError:
        print("ERREUR CRITIQUE : Impossible d'importer solver.py")
        sys.exit(1)


# --- CONFIGURATION ---
DATA_FOLDER = "data\\matrices_all" 

# Nombre maximum d'instances à utiliser pour le tuning 
# (Mettez None pour toutes les utiliser, ou un chiffre ex: 5 pour aller plus vite)
MAX_INSTANCES = None

# Temps accordé par instance (en secondes)
# Attention : Temps total par essai = TIME_PER_INSTANCE * Nombre d'instances
TIME_PER_INSTANCE = 30.0 


# --- 1. CHARGEMENT DE TOUTES LES INSTANCES ---
def load_all_datasets():
    """
    Loads all valid .txt files from the data folder.
    Returns a list of dictionaries containing the data.
    """
    if not os.path.exists(DATA_FOLDER):
        print(f"Dossier '{DATA_FOLDER}' introuvable.")
        return []

    files = glob.glob(os.path.join(DATA_FOLDER, "*.txt"))
    if not files:
        print(f"Aucun .txt dans '{DATA_FOLDER}'.")
        return []

    # On mélange pour avoir un échantillon représentatif si on limite le nombre
    # (Ou on trie pour la reproductibilité)
    files.sort() 
    
    if MAX_INSTANCES is not None:
        files = files[:MAX_INSTANCES]

    datasets = []
    print(f"Chargement de {len(files)} instances pour le tuning...")

    for fpath in files:
        try:
            with open(fpath, 'r') as f:
                header = f.readline().strip().split()
                if len(header) < 7: continue
                
                params = list(map(int, header))
                m, n, r, LW, UW, LH, UH = params[:7]
                
                X = np.loadtxt(f)
                if X.shape != (m, n): 
                    m, n = X.shape # Correction auto
                
                datasets.append({
                    'name': os.path.basename(fpath),
                    'X': X, 'r': r,
                    'LW': LW, 'UW': UW, 'LH': LH, 'UH': UH,
                    'pixels': m * n # Pour la normalisation
                })
        except Exception as e:
            print(f"Skipped {fpath}: {e}")
            
    print(f"-> {len(datasets)} instances chargées avec succès.\n")
    return datasets

# Chargement unique au démarrage
DATASETS = load_all_datasets()

if not DATASETS:
    print("Erreur : Aucune donnée chargée. Fin du programme.")
    sys.exit(1)


# --- 2. FONCTION OBJECTIVE (MULTI-INSTANCE) ---
def objective(trial):
    """
    Tests a set of hyperparameters on all instances.
    The objective is to minimize the NORMALIZED MEAN ERROR.
    """
    
    # A. Hyperparamètres à tester
    N = trial.suggest_int('N', 30, 200)
    tournament_size = trial.suggest_int('tournament_size', 2, 8)
    mutation_rate = trial.suggest_float('mutation_rate', 0.05, 0.4)
    
    total_normalized_error = 0.0
    
    # B. Boucle sur toutes les instances
    for data in DATASETS:
        try:
            # On lance le solver
            _, _, final_f = metaheuristic(
                data['X'], data['r'],
                data['LW'], data['UW'], data['LH'], data['UH'],
                TIME_LIMIT=TIME_PER_INSTANCE, # Temps court
                N=N,
                tournament_size=tournament_size,
                mutation_rate=mutation_rate
            )
            
            # NORMALISATION IMPORTANTE :
            # On divise l'erreur (somme des carrés) par le nombre de pixels (m*n)
            # Sinon, les grandes matrices auraient un poids disproportionné dans le score.
            normalized_error = final_f / data['pixels']
            
            total_normalized_error += normalized_error
            
        except Exception:
            return float('inf') # Pénalité max en cas de crash
            
    # C. Score final = Moyenne des erreurs normalisées
    mean_score = total_normalized_error / len(DATASETS)
    return mean_score


# --- 3. LANCEMENT ---
if __name__ == "__main__":
    print(f"--- Tuning Multi-Instance ---")
    print(f"Temps estimé par essai : {len(DATASETS) * TIME_PER_INSTANCE:.1f} secondes")
    
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
    
    # Nombre d'essais (Trials)
    # Attention, si vous avez 10 instances de 8s chacune, 1 essai = 80s.
    # 30 essais = 40 minutes. Ajustez N_TRIALS selon votre patience.
    N_TRIALS = 30
    
    study.optimize(objective, n_trials=N_TRIALS)
    
    print("\n" + "="*50)
    print(" MEILLEURS PARAMÈTRES GLOBAUX ")
    print("="*50)
    print(f"Score Moyen (MSE/pixel) : {study.best_value:.4f}")
    for k, v in study.best_params.items():
        print(f"  - {k}: {v}")