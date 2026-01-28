#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <algorithm>
#include <omp.h>
#include <random>
#include <iostream>
#include <tuple>
#include <chrono>

namespace py = pybind11;
using namespace Eigen;
using namespace std;

// --- HELPERS ---

struct TabuState {
    vector<int> list;
    int tenure;
    TabuState(int r, int t) : list(r, 0), tenure(t) {}
    bool is_tabu(int index, int iter) { return list[index] > iter; }
    
    // Modified to accept a random generator
    template<typename RNG>
    void make_tabu(int index, int iter, RNG& gen) {
        uniform_int_distribution<> dist(0, 1);
        list[index] = iter + tenure + dist(gen);
    }
    
    void clear() { fill(list.begin(), list.end(), 0); }
};

void bnb_recursive(
    int k, 
    int r, 
    double current_partial_error, 
    const MatrixXd& R, 
    const VectorXd& z, 
    const vector<vector<int>>& candidates,
    VectorXi& current_h, 
    VectorXi& best_h, 
    double& best_score) 
{
    /**
     * Recursively explores the 2^r integer combinations using the 
     * triangular structure of the Cholesky decomposition to prune branches.
     */
    if (k < 0) {
        if (current_partial_error < best_score) {
            best_score = current_partial_error;
            best_h = current_h;
        }
        return;
    }

    for (int cand : candidates[k]) {
        current_h(k) = cand;

        double val = 0.0;
        for (int j = k; j < r; ++j) {
            val += R(k, j) * (double)current_h(j);
        }
        double diff = val - z(k);
        double new_error = current_partial_error + (diff * diff);

        if (new_error < best_score) {
            bnb_recursive(k - 1, r, new_error, R, z, candidates, current_h, best_h, best_score);
        }
    }
}

// Counts the number of differences (Hamming Distance) between two matrices
int count_diff(const MatrixXi& A, const MatrixXi& B) {
    return (int)(A - B).cwiseAbs().cast<bool>().count();
}

// --- OPTIMISATION LOCALE ---

// --- 1. SOLVE IMF ---

/**
 * Recherche récursive pour le Sphere Decoding.
 * Utilise la structure triangulaire de la décomposition de Cholesky R.
 * Minimise ||z - Rh||^2.
 */
void sphere_decoder_recursive(
    int k, 
    int r,
    double current_partial_error,
    const MatrixXd& R,
    const VectorXd& z,
    double LH, double UH,
    VectorXi& current_h,
    VectorXi& best_h,
    double& best_dist_sq,
    long long& nodes_visited,
    long long node_limit
) {
    if (nodes_visited > node_limit) return;
    nodes_visited++;

    if (k < 0) {
        if (current_partial_error < best_dist_sq) {
            best_dist_sq = current_partial_error;
            best_h = current_h;
        }
        return;
    }

    // Calcul du centre pour h_k : (z_k - sum_{j=k+1}^{r-1} R_kj * h_j) / R_kk
    double val_accum = 0.0;
    for (int j = k + 1; j < r; ++j) {
        val_accum += R(k, j) * (double)current_h(j);
    }
    double center = (z(k) - val_accum) / R(k, k);

    // Énumération de Schnorr-Euchner : on commence par l'entier le plus proche du centre
    int start_val = (int)std::round(center);
    start_val = std::max((int)LH, std::min((int)UH, start_val));

    // On explore les candidats dans un ordre alterné (plus proche -> plus loin)
    // Ex: center=3.2 -> [3, 4, 2, 5, 1...]
    int step = (center >= (double)start_val) ? 1 : -1;
    int cand = start_val;
    bool go_left = (step == -1);
    
    // Pour l'alternance
    int next_step = 1;

    while (cand >= LH && cand <= UH) {
        current_h(k) = cand;
        
        double diff = R(k, k) * (double)cand + val_accum - z(k);
        double new_error = current_partial_error + (diff * diff);

        // Élagage (Pruning) : si l'erreur partielle dépasse déjà le meilleur score,
        // on peut arrêter cette branche (et les suivantes car l'erreur est quadratique)
        if (new_error >= best_dist_sq) {
            // Dans SE, dès qu'un candidat dépasse la sphère, on n'a plus besoin 
            // de regarder de ce côté (gauche ou droite).
            break; 
        }

        sphere_decoder_recursive(k - 1, r, new_error, R, z, LH, UH, current_h, best_h, best_dist_sq, nodes_visited, node_limit);

        // Prochain candidat dans l'ordre alterné
        if (next_step % 2 != 0) {
            cand = start_val + step;
            if (step > 0) step++; else step--;
        } else {
            // On change de direction
            step = -step;
            cand = start_val + step;
            if (step > 0) step++; else step--;
        }
        next_step++;
        
        if (nodes_visited > node_limit) break;
    }
}

/**
 * solve_column_imf_sphere : Implémentation du Sphere Decoding (CVP entier)
 */
double solve_column_imf_sphere(
    const MatrixXd& WtW, 
    const VectorXd& Wtx, 
    VectorXi& current_vec, 
    double LH, double UH, 
    double col_norm_sq,
    long long node_limit = 100000 // Limite pour éviter l'explosion combinatoire sur r=100
) {
    int r = (int)WtW.rows();

    // 1. Décomposition de Cholesky WtW = R^T * R
    LLT<MatrixXd> llt(WtW);
    if (llt.info() != Eigen::Success) {
        return col_norm_sq; // Fallback
    }
    
    MatrixXd R = llt.matrixU(); 
    VectorXd z = llt.matrixL().solve(Wtx); 
    // z est tel que ||Wh - x||^2 = ||Rh - z||^2 + const

    // 2. Initialisation avec le point de Babai (arrondi simple)
    VectorXd h_real = llt.solve(Wtx);
    VectorXi best_h(r);
    for(int i=0; i<r; ++i) {
        best_h(i) = (int)std::max(LH, std::min(UH, std::round(h_real(i))));
    }
    
    double best_dist_sq = (R * best_h.cast<double>() - z).squaredNorm();
    
    // 3. Recherche par Sphere Decoding
    VectorXi temp_h = VectorXi::Zero(r);
    long long nodes_visited = 0;
    
    sphere_decoder_recursive(r - 1, r, 0.0, R, z, LH, UH, temp_h, best_h, best_dist_sq, nodes_visited, node_limit);

    current_vec = best_h;

    // Calcul erreur finale : ||x||^2 - ||z||^2 + ||Rh-z||^2
    // Note: z.squaredNorm() est l'erreur du problème réel non contraint
    return std::max(0.0, col_norm_sq - z.squaredNorm() + best_dist_sq);
}

double solve_matrix_imf(const MatrixXd& X, const MatrixXd& W, MatrixXi& H, double LH, double UH) {
    MatrixXd WtW = W.transpose() * W;
    MatrixXd WtX = W.transpose() * X;
    int m = (int)X.cols();
    double total_error = 0.0;

    // On définit une limite de noeuds pour éviter que l'algorithme ne gèle sur de très grands r
    // Si r est petit (< 20), le Sphere Decoding trouvera l'optimum exact.
    // Si r est grand, il agira comme un solveur approché de haute qualité.
    long long node_limit = 100000; 

    #pragma omp parallel for reduction(+:total_error)
    for (int j = 0; j < m; ++j) {
        VectorXi h_col = H.col(j);
        double col_err = solve_column_imf_sphere(WtW, WtX.col(j), h_col, LH, UH, X.col(j).squaredNorm(), node_limit);
        H.col(j) = h_col;
        total_error += col_err;
    }
    return total_error;
}

// --- 2. SOLVE BMF (Optimized Boolean: 1+1=1) ---
double solve_matrix_bmf(const MatrixXd& X, const MatrixXi& Fixed, MatrixXi& Target, int L, int U) {
    // Fixed = W (m x r) [Binaire], Target = H (r x n) [Binaire]
    int m = (int)X.rows();
    int n = (int)X.cols();
    int r = (int)Fixed.cols();

    // Conversion pour accès rapide (Column-Major par défaut dans Eigen, donc accès rapide aux colonnes de W)
    // On garde Fixed en MatrixXi pour la logique.
    
    double total_error = 0.0;

    #pragma omp parallel for schedule(dynamic, 4) reduction(+:total_error)
    for (int j = 0; j < n; ++j) {
        // Copie locale de la colonne H (Target)
        VectorXi h_col = Target.col(j);
        
        // 1. Initialisation du vecteur de couverture (Cover Count)
        // P[i] = nombre de facteurs k tels que W[i,k]=1 ET H[k,j]=1
        // C'est un produit matriciel arithmétique standard pour compter.
        VectorXi coverage = Fixed * h_col; 

        bool improved = true;
        int iter = 0;
        int max_iters = 15; 

        // Pré-calcul des indices des 1 dans chaque colonne de W pour aller vite
        // (Optionnel : si W est très creux, on pourrait le faire hors de la boucle j, 
        // mais ici on accède directement à la mémoire contiguë d'Eigen).

        while (improved && iter < max_iters) {
            improved = false;
            iter++;
            
            // On teste le flip de chaque bit k de H
            // Stratégie "Best Improvement" ou "First Improvement"
            // Ici : Greedy Best Improvement par colonne
            
            int best_k = -1;
            int best_gain = 0;
            int best_action = 0; // 1 (0->1) ou -1 (1->0)

            for (int k = 0; k < r; ++k) {
                int h_val = h_col(k);
                int action = (h_val == 0) ? 1 : -1; // Si 0 on veut passer à 1, si 1 on veut passer à 0
                
                int gain = 0;

                // On ne regarde QUE les lignes i où W(i, k) == 1
                // C'est là que l'accélération se fait vs l'approche naïve O(m)
                // En Eigen, Fixed.col(k) est un itérateur efficace.
                
                for (int i = 0; i < m; ++i) {
                    // Note: Si W est stocké sparse, on pourrait itérer seulement les non-zéros.
                    // Avec Eigen Dense, on teste :
                    if (Fixed(i, k) == 0) continue;

                    int x_val = (int)X(i, j); // 0 ou 1
                    int cov = coverage(i);    // 0, 1, 2...

                    // CAS 1 : FLIP 0 -> 1 (Ajout d'un facteur)
                    if (action == 1) {
                        // L'ajout n'a d'impact que si la couverture passe de 0 à 1.
                        // Si cov > 0, le pixel est déjà allumé (1+1=1), donc pas de changement d'erreur.
                        if (cov == 0) {
                            // On passe de 0 (éteint) à 1 (allumé)
                            if (x_val == 1) gain++; // C'était 1, on a allumé -> Bien (+1)
                            else gain--;            // C'était 0, on a allumé -> Mal (-1)
                        }
                    }
                    // CAS 2 : FLIP 1 -> 0 (Retrait d'un facteur)
                    else {
                        // Le retrait n'a d'impact que si la couverture passe de 1 à 0.
                        // Si cov > 1, le pixel reste allumé par un autre facteur.
                        if (cov == 1) {
                            // On passe de 1 (allumé) à 0 (éteint)
                            if (x_val == 1) gain--; // C'était 1, on a éteint -> Mal (-1)
                            else gain++;            // C'était 0, on a éteint -> Bien (+1)
                        }
                    }
                }

                if (gain > best_gain) {
                    best_gain = gain;
                    best_k = k;
                    best_action = action;
                }
            }

            if (best_gain > 0) {
                // Appliquer le flip
                h_col(best_k) += best_action;
                
                // Mettre à jour la couverture (uniquement là où W vaut 1)
                for (int i = 0; i < m; ++i) {
                    if (Fixed(i, best_k) == 1) {
                        coverage(i) += best_action;
                    }
                }
                improved = true;
            }
        }

        Target.col(j) = h_col;

        // Calcul erreur Hamming finale
        // Erreur = Somme |X - (Coverage > 0)|
        for(int i=0; i<m; ++i) {
            int pred = (coverage(i) > 0) ? 1 : 0;
            if ((int)X(i, j) != pred) total_error += 1.0;
        }
    }

    return total_error;
}

// --- 3. SOLVE RELU (Optimized with OpenMP) ---
double solve_matrix_relu(const MatrixXd& X, const MatrixXi& Fixed, MatrixXi& Target, int L, int U) {
    int m = (int)X.rows();
    int n = (int)X.cols();
    int r = (int)Fixed.cols();
    
    MatrixXd Fd = Fixed.cast<double>(); // m x r
    
    // Pour ReLU, on ne peut pas utiliser facilement la matrice de Gram (M = W'W) 
    // car la non-linéarité (max(0, ...)) dépend des lignes individuelles.
    // On garde l'approche itérative mais on la parallélise massivement.

    double total_error = 0.0;

    #pragma omp parallel for schedule(dynamic, 4) reduction(+:total_error)
    for (int j = 0; j < n; ++j) {
        // Copie locale de la colonne cible
        VectorXi current_col = Target.col(j);
        VectorXd current_col_d = current_col.cast<double>();

        // Pré-calcul de la prédiction actuelle pour cette colonne P = W * h
        VectorXd P = Fd * current_col_d;

        bool improved = true;
        int iter = 0;
        int max_iters = 15;

        while (improved && iter < max_iters) {
            improved = false;
            iter++;

            // Optimisation coordonnée par coordonnée (CD)
            for (int k = 0; k < r; ++k) {
                double numerator = 0.0;
                double denominator = 0.0;
                
                // Calcul du gradient partiel sur les lignes actives
                // On cherche delta tel que || X - ReLU(P + delta*W_k) ||^2 est min
                // Approximation : on linéarise autour de l'état actif actuel
                for (int i = 0; i < m; ++i) {
                    double fix_val = Fd(i, k);
                    if (std::abs(fix_val) < 1e-9) continue; // Sparse optimization

                    double p_val = P(i);
                    double x_val = X(i, j);

                    // Condition active du ReLU :
                    // Si le pixel est "allumé" par la reconstruction (p_val > 0)
                    // OU s'il devrait l'être car X est positif (p_val <= 0 && x_val > 0) -> heuristic
                    if (p_val > 0 || (p_val <= 0 && x_val > 0)) {
                        // Residual r = x - p
                        numerator += (x_val - p_val) * fix_val;
                        denominator += fix_val * fix_val;
                    }
                }

                if (denominator > 1e-9) {
                    double delta = numerator / denominator;
                    
                    // On teste la nouvelle valeur
                    double old_val_d = current_col_d(k);
                    double new_val_raw = old_val_d + delta;
                    
                    int new_val_int = (int)std::round(new_val_raw);
                    if (new_val_int < L) new_val_int = L;
                    if (new_val_int > U) new_val_int = U;
                    
                    if (new_val_int != current_col(k)) {
                        double diff = (double)new_val_int - old_val_d;
                        
                        // Mise à jour
                        current_col(k) = new_val_int;
                        current_col_d(k) = (double)new_val_int;

                        // Mise à jour incrémentale de P (O(m))
                        // C'est mieux que de tout recalculer
                        P += Fd.col(k) * diff;
                        improved = true;
                    }
                }
            }
        }

        Target.col(j) = current_col;

        // Calcul erreur finale (avec ReLU)
        // (X - max(0, P))^2
        total_error += (X.col(j) - P.cwiseMax(0.0)).squaredNorm();
    }

    return total_error;
}

// --- ALIGNMENT ---

void align_in_place(const MatrixXi& W1, MatrixXi& W2_to_align, MatrixXi& H2_to_align) {
    int r = (int)W1.cols();
    int m = (int)W1.rows();
    MatrixXi W2_copy = W2_to_align;
    MatrixXi H2_copy = H2_to_align;
    
    vector<bool> used(r, false);
    
    for (int k1 = 0; k1 < r; ++k1) {
        long long min_dist = -1;
        int best_k2 = -1;
        for (int k2 = 0; k2 < r; ++k2) {
            if (used[k2]) continue;
            long long dist = 0;
            for(int i=0; i<m; ++i) {
                int d = W1(i, k1) - W2_copy(i, k2);
                dist += d*d;
            }
            if (min_dist == -1 || dist < min_dist) { min_dist = dist; best_k2 = k2; }
        }
        if (best_k2 != -1) { 
            used[best_k2] = true; 
            W2_to_align.col(k1) = W2_copy.col(best_k2);
            H2_to_align.row(k1) = H2_copy.row(best_k2); 
        } else {
             for(int k=0; k<r; ++k) if(!used[k]) { 
                 W2_to_align.col(k1) = W2_copy.col(k); 
                 H2_to_align.row(k1) = H2_copy.row(k);
                 used[k]=true; 
                 break; 
             }
        }
    }
}

int get_aligned_distance(const MatrixXi& W1, MatrixXi W2) {
    MatrixXi H_dummy = MatrixXi::Zero(W2.cols(), 1); 
    align_in_place(W1, W2, H_dummy);
    return count_diff(W1, W2);
}

// --- BATCH GENERATION ---

vector<tuple<MatrixXi, MatrixXi, double, int, int, int, int>> generate_children_batch(
    const MatrixXd& X, 
    const vector<MatrixXi>& Pop_W, 
    const vector<MatrixXi>& Pop_H,
    const vector<double>& Pop_Fitness,
    int num_children,
    int tournament_size,
    double mutation_rate,
    int LW, int UW, int LH, int UH,
    int crossover_mode,
    int mutation_mode,
    string mode_opti,
    int seed
) {
    vector<tuple<MatrixXi, MatrixXi, double, int, int, int, int>> results(num_children);
    int pop_size = (int)Pop_W.size();
    int m = (int)X.rows();
    int r = (int)Pop_W[0].cols();
    
    #pragma omp parallel 
    {
        int thread_id = omp_get_thread_num();
        unsigned int thread_seed = seed + (thread_id+1) * 9781;
        mt19937 gen(thread_seed); 
        uniform_int_distribution<> dist_idx(0, pop_size - 1);
        uniform_real_distribution<> dist_prob(0.0, 1.0);
        uniform_int_distribution<> dist_val_W(LW, UW);
        uniform_int_distribution<> dist_col(0, r - 1);
        uniform_int_distribution<> dist_row(0, m - 1);

        #pragma omp for schedule(dynamic)
        for (int i = 0; i < num_children; ++i) {
            
            // 1. Tournois
            int best_p1 = -1; double best_f1 = 1e20;
            for(int t=0; t<tournament_size; ++t) {
                int idx = dist_idx(gen);
                if (Pop_Fitness[idx] < best_f1) { best_f1 = Pop_Fitness[idx]; best_p1 = idx; }
            }
            int best_p2 = -1; double best_f2 = 1e20;
            for(int t=0; t<tournament_size; ++t) {
                int idx = dist_idx(gen);
                if (Pop_Fitness[idx] < best_f2) { best_f2 = Pop_Fitness[idx]; best_p2 = idx; }
            }

            MatrixXi W1 = Pop_W[best_p1];
            MatrixXi H1 = Pop_H[best_p1];
            
            MatrixXi W2 = Pop_W[best_p2];
            MatrixXi H2 = Pop_H[best_p2];
            
            // Alignement
            align_in_place(W1, W2, H2);

            MatrixXi Child_W(m, r);
            MatrixXi Child_H(r, H1.cols());

            // 2. Crossover Strategy
            if(crossover_mode == 0){
                if (dist_prob(gen) < 0.6) {
                    for(int k=0; k<r; ++k) {
                        if(dist_prob(gen) < 0.5) {
                            Child_W.col(k) = W1.col(k);
                            Child_H.row(k) = H1.row(k);
                        } else {
                            Child_W.col(k) = W2.col(k);
                            Child_H.row(k) = H2.row(k);
                        }
                    }
                } else {
                    // MEAN CROSSOVER (Good for local refinement)
                    Child_W = ((W1.cast<double>() + W2.cast<double>()) * 0.5).array().round().cast<int>();
                    Child_H = ((H1.cast<double>() + H2.cast<double>()) * 0.5).array().round().cast<int>();
                }
            }else if(crossover_mode == 1){
                for(int k=0; k<r; ++k) {
                    if(dist_prob(gen) < 0.5) {
                        Child_W.col(k) = W1.col(k);
                        Child_H.row(k) = H1.row(k);
                    } else {
                        Child_W.col(k) = W2.col(k);
                        Child_H.row(k) = H2.row(k);
                    }
                }
            }else if(crossover_mode == 2){
                // MEAN CROSSOVER (Good for local refinement)
                Child_W = ((W1.cast<double>() + W2.cast<double>()) * 0.5).array().round().cast<int>();
                Child_H = ((H1.cast<double>() + H2.cast<double>()) * 0.5).array().round().cast<int>();
            }
            
            Child_W = Child_W.cwiseMax(LW).cwiseMin(UW);
            Child_H = Child_H.cwiseMax(LH).cwiseMin(UH);

            // 3. Mutation
            if(mutation_mode == 0){
                if (dist_prob(gen) < 0.20) { 
                    int c1 = dist_col(gen);
                    int c2 = dist_col(gen);
                    Child_W.col(c1).swap(Child_W.col(c2));
                    Child_H.row(c1).swap(Child_H.row(c2));
                }
                if (dist_prob(gen) < 0.30) { 
                    int col = dist_col(gen);
                    double type = dist_prob(gen);
                    
                    if (type < 0.3) {
                        for(int row=0; row<m; ++row) Child_W(row, col) = LW; 
                    } else if (type < 0.6) {
                        for(int row=0; row<m; ++row) Child_W(row, col) = dist_val_W(gen);
                    } else {
                        double alpha = (dist_prob(gen) < 0.5) ? 0.5 : 2.0;
                        for(int row=0; row<m; ++row) {
                            int val = (int)round(Child_W(row, col) * alpha); 
                            Child_W(row, col) = max(LW, min(UW, val));
                        }
                    }
                }
                if (dist_prob(gen) < 0.7) { 
                    for(int row=0; row<m; ++row) {
                        for(int col=0; col<r; ++col) {
                            if (dist_prob(gen) < mutation_rate) {
                                int shift = (dist_prob(gen) < 0.5) ? -1 : 1;
                                if (dist_prob(gen) < 0.1) shift *= 2;
                                int val = Child_W(row, col) + shift;
                                Child_W(row, col) = max(LW, min(UW, val));
                            }
                        }
                    }
                }
            }else if(mutation_mode == 1){
                if (dist_prob(gen) < 0.20) { 
                    int c1 = dist_col(gen);
                    int c2 = dist_col(gen);
                    Child_W.col(c1).swap(Child_W.col(c2));
                    Child_H.row(c1).swap(Child_H.row(c2));
                }
            }else if(mutation_mode == 2){
                if (dist_prob(gen) < 0.30) { 
                    int col = dist_col(gen);
                    double type = dist_prob(gen);
                    
                    if (type < 0.3) {
                        for(int row=0; row<m; ++row) Child_W(row, col) = LW; 
                    } else if (type < 0.6) {
                        for(int row=0; row<m; ++row) Child_W(row, col) = dist_val_W(gen);
                    } else {
                        double alpha = (dist_prob(gen) < 0.5) ? 0.5 : 2.0;
                        for(int row=0; row<m; ++row) {
                            int val = (int)round(Child_W(row, col) * alpha); 
                            Child_W(row, col) = max(LW, min(UW, val));
                        }
                    }
                }
            }else if(mutation_mode == 3){
                if (dist_prob(gen) < 0.7) { 
                    for(int row=0; row<m; ++row) {
                        for(int col=0; col<r; ++col) {
                            if (dist_prob(gen) < mutation_rate) {
                                int shift = (dist_prob(gen) < 0.5) ? -1 : 1;
                                if (dist_prob(gen) < 0.1) shift *= 2;
                                int val = Child_W(row, col) + shift;
                                Child_W(row, col) = max(LW, min(UW, val));
                            }
                        }
                    }
                }
            }

            // 4. Optimisation Rapide 
            double f_obj = 0.0;
            int max_iters_child = 5; 
            
            for(int iter=0; iter<max_iters_child; ++iter) {
                // EXPLICIT CASTING 
                MatrixXi W = Child_W.cast<int>(); 
                
                if(mode_opti == "IMF") {
                    MatrixXd W_float = W.cast<double>();
                    f_obj = solve_matrix_imf(X, W_float, Child_H, LH, UH);
                } else if(mode_opti == "BMF") {
                    f_obj = solve_matrix_bmf(X, W, Child_H, LH, UH);
                } else if(mode_opti == "RELU") {
                    f_obj = solve_matrix_relu(X, W, Child_H, LH, UH);
                }
                
                MatrixXd XT = X.transpose();
                MatrixXi H = Child_H.cast<int>();
                MatrixXi HT = H.transpose();
                
                MatrixXi WT = Child_W.transpose();
                double f_w = f_obj;
                
                if(mode_opti == "IMF") {
                    MatrixXd HT_float = HT.cast<double>();
                    f_w = solve_matrix_imf(XT, HT_float, WT, LW, UW);
                } else if(mode_opti == "BMF") {
                    f_w = solve_matrix_bmf(XT, HT, WT, LW, UW);
                } else if(mode_opti == "RELU") {
                    f_w = solve_matrix_relu(XT, HT, WT, LW, UW);
                }
                
                Child_W = WT.transpose();
                f_obj = f_w;
            }
            
            int dist_p1 = count_diff(Child_W, W1);
            int dist_p2 = count_diff(Child_W, W2);

            results[i] = make_tuple(Child_W, Child_H, f_obj, best_p1, best_p2, dist_p1, dist_p2);
        }
    }

    return results;
}

// --- BINDINGS ---

tuple<MatrixXi, MatrixXi, double> optimize_alternating_cpp(
    MatrixXd X, MatrixXi W_init, MatrixXi H_init, 
    int LW, int UW, int LH, int UH, 
    int max_global_iters, double time_limit_seconds, string mode_opti) {

    auto start_time = chrono::high_resolution_clock::now();

    MatrixXi W = W_init;
    MatrixXi H = H_init;
    double f_obj = 1e20;
    MatrixXd XT = X.transpose();

    for (int global_iter = 0; global_iter < max_global_iters; ++global_iter) {
        auto current_time = chrono::high_resolution_clock::now();
        chrono::duration<double> elapsed = current_time - start_time;
        if (elapsed.count() > time_limit_seconds) break;

        double prev_f = f_obj;
        
        // Step 1: Optimize H
        MatrixXd W_float = W.cast<double>();
        if(mode_opti == "IMF") {
            f_obj = solve_matrix_imf(X, W_float, H, LH, UH);
        }else if(mode_opti == "BMF") {
            f_obj = solve_matrix_bmf(X, W, H, LH, UH);
        }else if(mode_opti == "RELU") {
            f_obj = solve_matrix_relu(X, W, H, LH, UH);
        }
        
        // Step 2: Optimize W
        MatrixXd H_float = H.cast<double>();
        MatrixXd HT = H_float.transpose();
        MatrixXi HT_int = HT.cast<int>();
        MatrixXi WT = W.transpose();
        
        double f_w = f_obj;
        if(mode_opti == "IMF") {
            f_w = solve_matrix_imf(XT, HT, WT, LW, UW);
        }else if(mode_opti == "BMF") {
            f_w = solve_matrix_bmf(XT, HT_int, WT, LW, UW);
        }else if(mode_opti == "RELU") {
            f_w = solve_matrix_relu(XT, HT_int, WT, LW, UW);
        }
        W = WT.transpose();
        f_obj = f_w;

        if (abs(prev_f - f_obj) < 1e-3 && global_iter > 3) break;
    }
    return make_tuple(W, H, f_obj);
}

pair<MatrixXi, double> optimize_h_cpp(MatrixXd X, MatrixXd W, int LW, int UW, int LH, int UH, string mode_opti) {
    MatrixXi H = MatrixXi::Zero(W.cols(), X.cols());
    double f = 1e20;
    MatrixXi W_int = W.cast<int>();
    if (mode_opti == "IMF") {
        f = solve_matrix_imf(X, W, H, LH, UH);
    }else if (mode_opti == "BMF") {
        f = solve_matrix_bmf(X, W_int, H, LH, UH);
    }else if (mode_opti == "RELU") {
        f = solve_matrix_relu(X, W_int, H, LH, UH);
    }
    return {H, f};
}

MatrixXi align_parents_cpp(MatrixXi W1, MatrixXi W2) {
    MatrixXi W2_copy = W2;
    MatrixXi H_dummy = MatrixXi::Zero(W2.cols(), 1); 
    align_in_place(W1, W2_copy, H_dummy);
    return W2_copy;
}

int get_aligned_distance_binding(MatrixXi W1, MatrixXi W2) {
    return get_aligned_distance(W1, W2);
}

PYBIND11_MODULE(fast_solver, m) {
    m.doc() = "Solver C++ FactInZ v2.5 (Explicit Inline Fix)";
    m.def("optimize_h", &optimize_h_cpp, py::call_guard<py::gil_scoped_release>());
    m.def("optimize_alternating", &optimize_alternating_cpp, py::call_guard<py::gil_scoped_release>(),
          py::arg("X"), py::arg("W_init"), py::arg("H_init"), 
          py::arg("LW"), py::arg("UW"), py::arg("LH"), py::arg("UH"), 
          py::arg("max_global_iters"), py::arg("time_limit_seconds") = 3600.0,
          py::arg("mode_opti") = "IMF"
          );
    m.def("align_parents_cpp", &align_parents_cpp, py::call_guard<py::gil_scoped_release>());
    m.def("generate_children_batch", &generate_children_batch,
          py::arg("X"), py::arg("Pop_W"), py::arg("Pop_H"), py::arg("Pop_Fitness"),
          py::arg("num_children"), py::arg("tournament_size"), py::arg("mutation_rate"),
          py::arg("LW"), py::arg("UW"), py::arg("LH"), py::arg("UH"),
          py::arg("crossover_mode") = 0,
          py::arg("mutation_mode") = 0,
          py::arg("mode_opti") = "IMF",
          py::arg("seed") = 42 
          );
    m.def("get_aligned_distance", &get_aligned_distance_binding, py::call_guard<py::gil_scoped_release>());
}