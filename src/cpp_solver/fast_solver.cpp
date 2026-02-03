#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <Eigen/Dense>
#include <Eigen/QR> 
#include <vector>
#include <cmath>
#include <algorithm>
#include <omp.h>
#include <random>
#include <tuple>
#include <chrono>
#include <numeric>
#include <queue>

namespace py = pybind11;
using namespace Eigen;
using namespace std;

// --- OPTIMISATION LOCALE ---

// =========================================================
//   STRUCTURES & WORKSPACE
// =========================================================

struct AcdWorkspace {
    std::vector<double> breakpts;
    std::vector<int> p;     
    std::vector<int> indices;
    
    std::vector<double> sga;
    std::vector<double> aa;
    std::vector<double> bb;
    std::vector<double> bc_term; 
    std::vector<double> tl_term; 
    
    void resize(int m) {
        if (breakpts.size() < (size_t)m) {
            breakpts.resize(m);
            p.resize(m);
            indices.resize(m);
            sga.resize(m);
            aa.resize(m);
            bb.resize(m);
            bc_term.resize(m);
            tl_term.resize(m);
        }
    }
};

struct BeamNode {
    double cost;
    std::vector<int> h;
    
    bool operator<(const BeamNode& other) const {
        return cost < other.cost;
    }
};

int count_diff(const MatrixXi& A, const MatrixXi& B) {
    return (int)(A - B).cwiseAbs().cast<bool>().count();
}

// =========================================================
//   MOTEUR MATHÉMATIQUE
// =========================================================

double findmin_relu_exact(const VectorXd& a_vec, const VectorXd& b_vec, const VectorXd& c_vec, AcdWorkspace& ws) {
    int m = (int)a_vec.size();
    int nnz = 0;
    
    for(int i=0; i<m; ++i) {
        if(std::abs(a_vec(i)) > 1e-16) {
            ws.indices[nnz] = i;
            ws.breakpts[nnz] = -b_vec(i) / a_vec(i);
            ws.p[nnz] = nnz; 
            nnz++;
        }
    }
    
    if(nnz == 0) return 0.0;

    std::sort(ws.p.begin(), ws.p.begin() + nnz, [&](int i, int j) {
        return ws.breakpts[i] < ws.breakpts[j];
    });

    double ti = 0.0; double tl = 0.0; double tq = 0.0;

    for(int i=0; i<nnz; ++i) {
        int idx = ws.indices[ws.p[i]];
        ti += c_vec(idx) * c_vec(idx);
    }

    for(int k=0; k<nnz; ++k) {
        int original_idx = ws.p[k];
        int idx = ws.indices[original_idx];
        
        double av = a_vec(idx);
        double bv = b_vec(idx);
        double cv = c_vec(idx);
        
        double sgn = (av < 0 ? -1.0 : 1.0);
        ws.sga[k] = sgn;
        ws.aa[k] = av * av;
        ws.bc_term[k] = bv*bv - 2*bv*cv; 
        ws.tl_term[k] = 2*av*bv - 2*av*cv;
        
        if (av < -1e-16) {
            ti += ws.bc_term[k];
            tl += ws.tl_term[k];
            tq += ws.aa[k];
        }
    }

    double xmin = (std::abs(tq) > 1e-16) ? -tl / (2 * tq) : 0.0;
    double bp0 = ws.breakpts[ws.p[0]];
    double xopt = (xmin < bp0) ? xmin : bp0;
    double yopt = ti + tl * xopt + tq * xopt * xopt;

    for(int k=0; k<nnz; ++k) {
        double sg = ws.sga[k];
        ti += sg * ws.bc_term[k];
        tl += sg * ws.tl_term[k];
        tq += sg * ws.aa[k];
        
        xmin = (std::abs(tq) > 1e-16) ? -tl / (2 * tq) : 0.0;
        
        double current_bp = ws.breakpts[ws.p[k]];
        double next_bp = (k + 1 < nnz) ? ws.breakpts[ws.p[k+1]] : 1e20;
        
        double xt;
        if (xmin > current_bp && xmin < next_bp) {
            xt = xmin;
        } else {
            xt = current_bp;
        }
        
        double yval = ti + tl * xt + tq * xt * xt;
        if (yval < yopt) {
            yopt = yval;
            xopt = xt;
        }
    }
    return xopt;
}

void integer_cd_relu(const MatrixXd& W, const VectorXd& X_col, VectorXi& h_int, int L, int U, AcdWorkspace& ws, mt19937& gen) {
    int r = (int)h_int.size();
    VectorXd wh = W * h_int.cast<double>();
    
    std::vector<int> indices(r);
    std::iota(indices.begin(), indices.end(), 0);

    int max_passes = 3; 
    for(int pass=0; pass<max_passes; ++pass) {
        std::shuffle(indices.begin(), indices.end(), gen);

        bool changed = false;
        for(int k : indices) {
            double h_old = (double)h_int(k);
            VectorXd b_vec = wh - W.col(k) * h_old;
            
            double h_opt_cont = findmin_relu_exact(W.col(k), b_vec, X_col, ws);
            int h_new = std::max(L, std::min(U, (int)std::round(h_opt_cont)));
            
            if (h_new != h_int(k)) {
                double diff = (double)h_new - h_old;
                wh += W.col(k) * diff; 
                h_int(k) = h_new;
                changed = true;
            }
        }
        if(!changed) break; 
    }
}

void integer_cd_imf(const MatrixXd& W, const VectorXd& W_norms_sq, const VectorXd& X_col, VectorXi& h_int, int L, int U, mt19937& gen) {
    int r = (int)h_int.size();
    VectorXd resid = X_col - W * h_int.cast<double>(); 
    
    std::vector<int> indices(r);
    std::iota(indices.begin(), indices.end(), 0);

    int max_passes = 20;
    for(int pass=0; pass<max_passes; ++pass) {
        std::shuffle(indices.begin(), indices.end(), gen);
        
        bool changed = false;
        for(int k : indices) {
            double denom = W_norms_sq(k);
            if (denom < 1e-9) continue;
            
            double numer = W.col(k).dot(resid);
            double delta_cont = numer / denom;
            double h_old = (double)h_int(k);
            
            int h_new = std::max(L, std::min(U, (int)std::round(h_old + delta_cont)));
            
            if (h_new != h_int(k)) {
                double diff = (double)h_new - h_old;
                resid -= W.col(k) * diff; 
                h_int(k) = h_new;
                changed = true;
            }
        }
        if(!changed) break;
    }
}

// =========================================================
//   SOLVEURS
// =========================================================

double solve_matrix_relu(const MatrixXd& X, const MatrixXi& Fixed, MatrixXi& Target, int L, int U) {
    MatrixXd W = Fixed.cast<double>();
    int m = (int)X.rows();
    int n = (int)X.cols();
    int r = (int)Fixed.cols();

    double total_error = 0.0;
    int acd_iter = 10; 

    #pragma omp parallel reduction(+:total_error)
    {
        AcdWorkspace ws;
        ws.resize(m); 
        
        int thread_id = omp_get_thread_num();
        std::mt19937 gen(9781 + thread_id * 1337); 

        #pragma omp for 
        for (int j = 0; j < n; ++j) {
            VectorXd h_col = Target.col(j).cast<double>(); 
            VectorXd X_col = X.col(j);
            VectorXd wh_col = W * h_col; 

            for(int iter=0; iter<acd_iter; ++iter) {
                bool changed = false;
                for(int k=0; k<r; ++k) {
                    double h_old = h_col(k);
                    VectorXd b_vec = wh_col - W.col(k) * h_old;
                    
                    double h_new = findmin_relu_exact(W.col(k), b_vec, X_col, ws);
                    h_new = std::max((double)L, std::min((double)U, h_new));

                    if(std::abs(h_new - h_old) > 1e-6) {
                        h_col(k) = h_new;
                        wh_col += W.col(k) * (h_new - h_old);
                        changed = true;
                    }
                }
                if(!changed) break; 
            }

            VectorXi h_int = h_col.array().round().cast<int>().cwiseMax(L).cwiseMin(U);
            
            integer_cd_relu(W, X_col, h_int, L, U, ws, gen);

            Target.col(j) = h_int;
            VectorXd wh_final = (Fixed.cast<double>() * h_int.cast<double>()).cwiseMax(0.0);
            total_error += (X_col - wh_final).squaredNorm();
        }
    }
    return total_error;
}

double solve_matrix_imf(const MatrixXd& X, const MatrixXi& Fixed, MatrixXi& Target, int L, int U) {
    // 1. Préparation des données W
    MatrixXd W = Fixed.cast<double>();
    int n = (int)X.cols();
    int r = (int)Fixed.cols();
    int m = (int)X.rows();

    // -- Pré-calculs pour Coordinate Descent Continu --
    MatrixXd G = W.transpose() * W; // Gram matrix r x r
    VectorXd W_norms_sq = G.diagonal(); // Pour la descente entière rapide

    // -- Pré-calculs pour Babai Nearest Plane --
    // Décomposition QR : W = Q * R
    // Nous avons besoin de R (triangulaire supérieure) pour Babai.
    // L'objectif de Babai est de trouver h entier minimisant || R h - Q^T x ||
    // Astuce : Q^T x peut être calculé comme (R^T)^-1 * (W^T x)
    // sans avoir besoin de stocker ou calculer explicitement la grande matrice Q (m x m).
    
    HouseholderQR<MatrixXd> qr(W);
    // On extrait la partie triangulaire supérieure R.
    // Note: Si m > r, HouseholderQR produit un R de taille m x r où les lignes r..m-1 sont nulles.
    // Nous prenons juste le bloc r x r supérieur.
    MatrixXd R_full = qr.matrixQR().triangularView<Upper>();
    MatrixXd R = R_full.block(0, 0, r, r);
    
    // Inversion de R transposée pour projeter la cible dans l'espace de Babai
    // R est petite (r x r), donc l'inversion est très rapide.
    MatrixXd R_invT = MatrixXd::Identity(r, r);
    bool use_babai = true;
    
    // Sécurité numérique si R est singulière
    if(std::abs(R.determinant()) < 1e-10) {
        use_babai = false; // Fallback sur CD pur si mal conditionné
    } else {
        R_invT = R.transpose().inverse();
    }

    double total_error = 0.0;

    #pragma omp parallel reduction(+:total_error)
    {
        int thread_id = omp_get_thread_num();
        std::mt19937 gen(9781 + thread_id * 1337);
        
        #pragma omp for 
        for (int j = 0; j < n; ++j) {
            VectorXd x = X.col(j);
            VectorXd h_real = Target.col(j).cast<double>(); // Warm start
            VectorXd p = W.transpose() * x; // Projection W^T x

            // ---------------------------------------------------------
            // ÉTAPE 1 : Coordinate Descent Continu (avec contraintes)
            // ---------------------------------------------------------
            // Minimise 0.5 * h^T G h - p^T h
            // Utile pour trouver le centre optimal global dans la boîte [L, U]
            
            int cd_cont_iter = 15;
            for(int iter=0; iter<cd_cont_iter; ++iter) {
                for(int k=0; k<r; ++k) {
                    // Gradient partiel sans h_k:
                    // Grad_k = (G*h)_k - p_k
                    // Optimum sans contrainte: h_k = (p_k - sum_{j!=k} G_kj h_j) / G_kk
                    // Formule rapide: h_new = h_old - (Grad_k / G_kk)
                    
                    double grad_k = G.row(k).dot(h_real) - p(k);
                    double step = grad_k / G(k,k);
                    double h_val = h_real(k) - step;
                    
                    // Projection sur la boîte
                    h_real(k) = std::max((double)L, std::min((double)U, h_val));
                }
            }

            // ---------------------------------------------------------
            // ÉTAPE 2 : Babai Nearest Plane (Hybride)
            // ---------------------------------------------------------
            // On compare deux candidats :
            // 1. Babai Standard (Unconstrained Center): Vise le centre non-contraint.
            // 2. Babai Guidé (Constrained Center): Vise le centre contraint h_real (via R*h_real).
            
            VectorXi h_best_babai(r);
            double best_babai_cost = 1e20;
            
            if (use_babai) {
                // --- Candidat 1 : Standard Babai ---
                VectorXd y_target = R_invT * p; // Q^T x
                VectorXi h_cand1(r);
                for (int k = r - 1; k >= 0; --k) {
                    double val = y_target(k);
                    for (int i = k + 1; i < r; ++i) val -= R(k, i) * (double)h_cand1(i);
                    double unconstrained = val / R(k, k);
                    h_cand1(k) = std::max(L, std::min(U, (int)std::round(unconstrained)));
                }
                
                h_best_babai = h_cand1;
                best_babai_cost = (x - W * h_cand1.cast<double>()).squaredNorm();
                
                // --- Candidat 2 : Babai Guidé par h_real (NOUVEAU) ---
                // Minimise || R(h - h_real) || en respectant le réseau
                // C'est une forme de rounding conditionnel intelligent de h_real
                VectorXd y_real = R * h_real; 
                VectorXi h_cand2(r);
                for (int k = r - 1; k >= 0; --k) {
                    double val = y_real(k);
                    for (int i = k + 1; i < r; ++i) val -= R(k, i) * (double)h_cand2(i);
                    double unconstrained = val / R(k, k);
                    h_cand2(k) = std::max(L, std::min(U, (int)std::round(unconstrained)));
                }

                // Si le candidat guidé est différent, on regarde s'il est meilleur
                if (h_cand2 != h_cand1) {
                    double cost2 = (x - W * h_cand2.cast<double>()).squaredNorm();
                    if (cost2 < best_babai_cost) {
                        h_best_babai = h_cand2;
                        // best_babai_cost = cost2; // Pas nécessaire de maj pour la suite
                    }
                }

            } else {
                // Fallback si QR échoue : simple arrondi
                h_best_babai = h_real.array().round().cast<int>().cwiseMax(L).cwiseMin(U);
            }

            // ---------------------------------------------------------
            // ÉTAPE 3 : Integer Coordinate Descent (Refinement)
            // ---------------------------------------------------------
            // On part du meilleur candidat Babai
            
            VectorXi h_final = h_best_babai;
            integer_cd_imf(W, W_norms_sq, x, h_final, L, U, gen);

            // Sauvegarde et calcul erreur
            Target.col(j) = h_final;
            VectorXd resid = x - W * h_final.cast<double>();
            total_error += resid.squaredNorm();
        }
    }
    return total_error;
}

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
        int max_iters = 50;

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

// --- ALIGNMENT ---

void align_in_place(const MatrixXi& W1, MatrixXi& W2_to_align, MatrixXi& H2_to_align) {
    int r = (int)W1.cols();
    MatrixXi W2_copy = W2_to_align; MatrixXi H2_copy = H2_to_align;
    vector<bool> used(r, false);
    for (int k1 = 0; k1 < r; ++k1) {
        long long min_dist = -1; int best_k2 = -1;
        for (int k2 = 0; k2 < r; ++k2) {
            if (used[k2]) continue;
            long long dist = (W1.col(k1) - W2_copy.col(k2)).squaredNorm();
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
                 used[k]=true; break; 
             }
        }
    }
}

tuple<MatrixXi, MatrixXi, double> optimize_alternating_cpp(
    MatrixXd X, MatrixXi W_init, MatrixXi H_init, 
    int LW, int UW, int LH, int UH, 
    int max_global_iters, double time_limit_seconds, string mode_opti);

vector<tuple<MatrixXi, MatrixXi, double, int, int, int, int>> generate_children_batch(
    const MatrixXd& X, const vector<MatrixXi>& Pop_W, const vector<MatrixXi>& Pop_H,
    const vector<double>& Pop_Fitness, int num_children, int tournament_size,
    double mutation_rate, int LW, int UW, int LH, int UH,
    int crossover_mode, int mutation_mode, string mode_opti, int seed
) {
    vector<tuple<MatrixXi, MatrixXi, double, int, int, int, int>> results(num_children);
    int pop_size = (int)Pop_W.size();
    int m = (int)X.rows();
    int r = (int)Pop_W[0].cols();
    
    MatrixXd XT = X.transpose();
    
    #pragma omp parallel 
    {
        int thread_id = omp_get_thread_num();
        unsigned int thread_seed = seed + (thread_id+1) * 9781;
        mt19937 gen(thread_seed); 
        uniform_int_distribution<> dist_idx(0, pop_size - 1);
        uniform_real_distribution<> dist_prob(0.0, 1.0);
        uniform_int_distribution<> dist_col(0, r - 1);
        
        #pragma omp for schedule(dynamic)
        for (int i = 0; i < num_children; ++i) {

            // 1. Tournois
            int best_p1 = -1; double best_f1 = 1e20;
            for(int t=0; t<tournament_size; ++t) {
                int idx = dist_idx(gen);
                if (Pop_Fitness[idx] < best_f1) { best_f1 = Pop_Fitness[idx]; best_p1 = idx; }
            }
            
            int best_p2 = -1; double best_f2 = 1e20;     // Le Vainqueur
            int second_p2 = -1; double second_f2 = 1e20; // Le Dauphin

            for(int t=0; t<tournament_size; ++t) {
                int idx = dist_idx(gen);
                double f = Pop_Fitness[idx];

                if (f < best_f2) {
                    // Le nouveau est meilleur que le 1er :
                    // L'ancien 1er devient le 2ème (s'il n'est pas écrasé par le même index)
                    if (idx != best_p2) { 
                        second_f2 = best_f2;
                        second_p2 = best_p2;
                    }
                    best_f2 = f;
                    best_p2 = idx;
                } 
                else if (f < second_f2 && idx != best_p2) {
                    // Le nouveau est moins bon que le 1er mais meilleur que le 2ème
                    second_f2 = f;
                    second_p2 = idx;
                }
            }

            if (best_p2 == best_p1) {
                if (second_p2 != -1) {
                    // On prend le deuxième meilleur
                    best_p2 = second_p2;
                    best_f2 = second_f2;
                } else {
                    // Cas de secours (ex: tournoi avec un seul individu tiré plusieurs fois)
                    // On prend un individu aléatoire différent
                    if (pop_size > 1) {
                        do { best_p2 = dist_idx(gen); } while (best_p2 == best_p1);
                        best_f2 = Pop_Fitness[best_p2];
                    }
                }
            }

            MatrixXi W1 = Pop_W[best_p1];
            MatrixXi H1 = Pop_H[best_p1];
            
            MatrixXi W2 = Pop_W[best_p2];
            MatrixXi H2 = Pop_H[best_p2];
            
            // Alignement
            align_in_place(W1, W2, H2);

            MatrixXi Child_W(m, r);
            MatrixXi Child_H(r, H1.cols());

            // 2. Crossover Strategy (PURGED & BIASED)
            // On calcule la probabilité de prendre le gène du Parent 1
            // Basé sur l'erreur (fitness) : plus l'erreur est basse, plus la proba est haute.
            
            double f1 = Pop_Fitness[best_p1];
            double f2 = Pop_Fitness[best_p2];
            double total_f = f1 + f2;
            
            double prob_p1 = 0.5; // Par défaut si total_f est 0 (ex: solution parfaite)
            
            // if (total_f > 1e-16) {
            //     prob_p1 = f2 / total_f; 
            // }

            prob_p1 = std::max(0.1, std::min(0.9, prob_p1));

            for(int k=0; k<r; ++k) {
                // On tire un nombre aléatoire entre 0 et 1
                if(dist_prob(gen) < prob_p1) {
                    // On prend le facteur du Parent 1
                    Child_W.col(k) = W1.col(k);
                    Child_H.row(k) = H1.row(k);
                } else {
                    // On prend le facteur du Parent 2
                    Child_W.col(k) = W2.col(k);
                    Child_H.row(k) = H2.row(k);
                }
            }

            // 3. Mutation (Single Random Factor Reset)
            // On teste si cet enfant doit subir une mutation.
            if (dist_prob(gen) < mutation_rate) {
                
                // On choisit UN seul facteur aléatoire à réinitialiser
                int k = dist_col(gen); // k est entre 0 et r-1
                
                // Distributions pour les nouvelles valeurs
                std::uniform_int_distribution<> dist_val_W(LW, UW);
                std::uniform_int_distribution<> dist_val_H(LH, UH);

                // 1. Reset de la colonne k de W
                for (int i = 0; i < m; ++i) {
                    Child_W(i, k) = dist_val_W(gen);
                }

                // 2. Reset de la ligne k de H
                for (int j = 0; j < H1.cols(); ++j) {
                    Child_H(k, j) = dist_val_H(gen);
                }
            }

            // 4. Optimisation Rapide 
            double f_obj = 0.0;

            tuple<MatrixXi, MatrixXi, double> opt_result;
            opt_result = optimize_alternating_cpp(
                X, Child_W, Child_H, 
                LW, UW, LH, UH, 
                50, 3600.0, mode_opti
            );

            tie(Child_W, Child_H, f_obj) = opt_result;

            // int max_iters_child = 5; 
            // for(int iter=0; iter<max_iters_child; ++iter) {
            //     // EXPLICIT CASTING 
            //     MatrixXi W = Child_W.cast<int>(); 
                
            //     if(mode_opti == "IMF") {
            //         f_obj = solve_matrix_imf(X, W, Child_H, LH, UH);
            //     } else if(mode_opti == "BMF") {
            //         f_obj = solve_matrix_bmf(X, W, Child_H, LH, UH);
            //     } else if(mode_opti == "RELU") {
            //         f_obj = solve_matrix_relu(X, W, Child_H, LH, UH);
            //     }
                
            //     MatrixXd XT = X.transpose();
            //     MatrixXi H = Child_H.cast<int>();
            //     MatrixXi HT = H.transpose();
                
            //     MatrixXi WT = Child_W.transpose();
            //     double f_w = f_obj;
                
            //     if(mode_opti == "IMF") {
            //         f_w = solve_matrix_imf(XT, HT, WT, LW, UW);
            //     } else if(mode_opti == "BMF") {
            //         f_w = solve_matrix_bmf(XT, HT, WT, LW, UW);
            //     } else if(mode_opti == "RELU") {
            //         f_w = solve_matrix_relu(XT, HT, WT, LW, UW);
            //     }
                
            //     Child_W = WT.transpose();
            // }

            // MatrixXi W = Child_W.cast<int>(); 
            // if(mode_opti == "IMF") {
            //     f_obj = solve_matrix_imf(X, W, Child_H, LH, UH);
            // } else if(mode_opti == "BMF") {
            //     f_obj = solve_matrix_bmf(X, W, Child_H, LH, UH);
            // } else if(mode_opti == "RELU") {
            //     f_obj = solve_matrix_relu(X, W, Child_H, LH, UH);
            // }

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
        if(mode_opti == "IMF") {
            f_obj = solve_matrix_imf(X, W, H, LH, UH);
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
            f_w = solve_matrix_imf(XT, HT_int, WT, LW, UW);
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
        f = solve_matrix_imf(X, W_int, H, LH, UH);
    }else if (mode_opti == "BMF") {
        f = solve_matrix_bmf(X, W_int, H, LH, UH);
    }else if (mode_opti == "RELU") {
        f = solve_matrix_relu(X, W_int, H, LH, UH);
    }
    return {H, f};
}

MatrixXi align_parents_cpp(MatrixXi W1, MatrixXi W2) {
    MatrixXi W2_copy = W2; MatrixXi H_dummy = MatrixXi::Zero(W2.cols(), 1); 
    align_in_place(W1, W2_copy, H_dummy);
    return W2_copy;
}

int get_aligned_distance(const MatrixXi& W1, MatrixXi W2) {
    MatrixXi H_dummy = MatrixXi::Zero(W2.cols(), 1); 
    align_in_place(W1, W2, H_dummy);
    return count_diff(W1, W2);
}

PYBIND11_MODULE(fast_solver, m) {
    m.doc() = "Solver C++ FactInZ Cleaned (ACD + Rounding for RELU/IMF)";
    m.def("optimize_h", &optimize_h_cpp, py::call_guard<py::gil_scoped_release>());
    m.def("optimize_alternating", &optimize_alternating_cpp, py::call_guard<py::gil_scoped_release>(),
          py::arg("X"), py::arg("W_init"), py::arg("H_init"), 
          py::arg("LW"), py::arg("UW"), py::arg("LH"), py::arg("UH"), 
          py::arg("max_global_iters"), py::arg("time_limit_seconds") = 3600.0,
          py::arg("mode_opti") = "IMF"
          );
    m.def("align_parents_cpp", &align_parents_cpp, py::call_guard<py::gil_scoped_release>());
    m.def("get_aligned_distance", &get_aligned_distance, py::call_guard<py::gil_scoped_release>());
    m.def("generate_children_batch", &generate_children_batch, py::call_guard<py::gil_scoped_release>(),
          py::arg("X"), py::arg("Pop_W"), py::arg("Pop_H"), py::arg("Pop_Fitness"),
          py::arg("num_children"), py::arg("tournament_size"), py::arg("mutation_rate"),
          py::arg("LW"), py::arg("UW"), py::arg("LH"), py::arg("UH"),
          py::arg("crossover_mode"), py::arg("mutation_mode"),
          py::arg("mode_opti"), py::arg("seed") = 42
          );
}