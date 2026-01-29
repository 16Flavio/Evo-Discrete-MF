#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <algorithm>
#include <omp.h>
#include <random>
#include <tuple>
#include <chrono>
#include <numeric>

namespace py = pybind11;
using namespace Eigen;
using namespace std;

// --- HELPERS ---

int count_diff(const MatrixXi& A, const MatrixXi& B) {
    return (int)(A - B).cwiseAbs().cast<bool>().count();
}

// --- INITIALISATION CONTINUE CONTRAINTE ---

VectorXd solve_continuous_bounded(const MatrixXd& WtW, const VectorXd& Wtx, double LH, double UH) {
    int r = (int)WtW.rows();
    VectorXd h = VectorXd::Zero(r);
    for(int i=0; i<r; ++i) h(i) = std::max(LH, std::min(UH, h(i)));

    VectorXd v = WtW * h;
    int max_iter = 50; 
    double tol = 1e-1;
    
    for(int iter=0; iter<max_iter; ++iter) {
        double max_change = 0.0;
        for(int k=0; k<r; ++k) {
            double m_kk = WtW(k, k);
            if(m_kk < 1e-9) continue;
            
            double old_hk = h(k);
            double numerator = Wtx(k) - (v(k) - m_kk * old_hk);
            double target = numerator / m_kk;
            double new_hk = std::max(LH, std::min(UH, target));
            
            if(std::abs(new_hk - old_hk) > 1e-8) {
                h(k) = new_hk;
                v += WtW.col(k) * (new_hk - old_hk);
                max_change = std::max(max_change, std::abs(new_hk - old_hk));
            }
        }
        if(max_change < tol) break;
    }
    return h;
}

// --- SPHERE DECODING (CORE FLEXIBLE) ---

template <typename VisitorFunc>
void sphere_decoder_recursive(
    int k, int r,
    double current_partial_error,
    const MatrixXd& R, const VectorXd& z,
    double LH, double UH,
    VectorXi& current_h,
    double& best_dist_sq_for_pruning,
    long long& nodes_visited,
    long long node_limit,
    VisitorFunc leaf_visitor
) {
    if (nodes_visited > node_limit) return;
    nodes_visited++;

    if (k < 0) {
        leaf_visitor(current_h, current_partial_error);
        return;
    }

    double val_accum = 0.0;
    for (int j = k + 1; j < r; ++j) val_accum += R(k, j) * (double)current_h(j);
    double center = (z(k) - val_accum) / R(k, k);

    int start_val = (int)std::round(center);
    start_val = std::max((int)LH, std::min((int)UH, start_val));

    int step = (center >= (double)start_val) ? 1 : -1;
    int cand = start_val;
    int next_step = 1;

    while (cand >= LH && cand <= UH) {
        current_h(k) = cand;
        double res = R(k, k) * (double)cand + val_accum - z(k);
        double new_error = current_partial_error + (res * res);

        if (new_error < best_dist_sq_for_pruning) {
            sphere_decoder_recursive(k - 1, r, new_error, R, z, LH, UH, current_h, 
                                   best_dist_sq_for_pruning, nodes_visited, node_limit, leaf_visitor);
        } else {
            break; 
        }

        if (next_step % 2 != 0) {
            cand = start_val + step;
            if (step > 0) step++; else step--;
        } else {
            step = -step;
            cand = start_val + step;
            if (step > 0) step++; else step--;
        }
        next_step++;
        if (nodes_visited > node_limit) break;
    }
}

// --- SOLVEURS GLOBAUX ---

double solve_matrix_imf(const MatrixXd& X, const MatrixXd& W, MatrixXi& H, double LH, double UH) {
    MatrixXd WtW = W.transpose() * W;
    MatrixXd WtX = W.transpose() * X;
    int m = (int)X.cols();
    int r = (int)W.cols();
    double total_error = 0.0;
    long long node_limit = 50000;

    #pragma omp parallel for reduction(+:total_error)
    for (int j = 0; j < m; ++j) {
        VectorXi h_col = H.col(j);
        LLT<MatrixXd> llt(WtW);
        if (llt.info() == Eigen::Success) {
            MatrixXd R = llt.matrixU();
            VectorXd z = llt.matrixL().solve(WtX.col(j));
            
            VectorXd h_cont = solve_continuous_bounded(WtW, WtX.col(j), LH, UH);
            for(int i=0; i<r; ++i) h_col(i) = (int)std::round(h_cont(i));
            h_col = h_col.cwiseMax((int)LH).cwiseMin((int)UH);
            
            double best_dist_sq = (R * h_col.cast<double>() - z).squaredNorm();
            // VectorXi temp_h = VectorXi::Zero(r);
            // long long nodes = 0;
            
            // auto imf_visitor = [&](const VectorXi& candidate_h, double quad_error) {
            //     if (quad_error < best_dist_sq) {
            //         best_dist_sq = quad_error;
            //         h_col = candidate_h;
            //     }
            // };

            // sphere_decoder_recursive(r-1, r, 0.0, R, z, LH, UH, temp_h, best_dist_sq, nodes, node_limit, imf_visitor);
            
            H.col(j) = h_col;
            total_error += std::max(0.0, X.col(j).squaredNorm() - z.squaredNorm() + best_dist_sq);
        } else {
            total_error += (X.col(j) - W.cast<double>() * h_col.cast<double>()).squaredNorm();
        }
    }
    return total_error;
}

// BMF OPTIMISÉE : APPROCHE GLOUTONNE (GREEDY)
double solve_matrix_bmf(const MatrixXd& X, const MatrixXi& Fixed, MatrixXi& Target, int L, int U) {
    int m = (int)X.rows();
    int n = (int)X.cols();
    int r = (int)Fixed.cols();

    double total_error = 0.0;

    // On traite chaque colonne de X indépendamment
    #pragma omp parallel for reduction(+:total_error)
    for (int j = 0; j < n; ++j) {
        VectorXi x_target = X.col(j).cast<int>(); // La cible (colonne de X)
        VectorXi h_col = VectorXi::Zero(r);       // Le vecteur d'activation initial
        VectorXi current_recon = VectorXi::Zero(m); // La reconstruction courante (tout à 0 au début)

        bool improvement = true;

        // Boucle gloutonne : on ajoute des colonnes de W tant que ça réduit l'erreur
        while(improvement) {
            improvement = false;
            int best_k = -1;
            int best_gain = 0;

            // Essayer chaque colonne k de W non encore utilisée
            for(int k=0; k<r; ++k) {
                if(h_col(k) == 1) continue; 

                // Calculer le gain d'ajouter cette colonne
                // Gain = (Nouveaux '1' couverts correctement) - (Nouveaux '0' couverts par erreur)
                int gain = 0;
                
                for(int i=0; i<m; ++i) {
                    // Si W(i,k) est 0, ou si la ligne i est déjà couverte par une autre colonne choisie,
                    // l'ajout de k ne change rien pour cette ligne (propriété OR)
                    if (Fixed(i, k) == 1 && current_recon(i) == 0) {
                        if (x_target(i) == 1) {
                            gain++; // C'est un 1 qu'on voulait couvrir -> Positif
                        } else {
                            gain--; // C'est un 0 qu'on ne devait pas couvrir -> Négatif
                        }
                    }
                }

                if(gain > best_gain) {
                    best_gain = gain;
                    best_k = k;
                }
            }

            // Si le meilleur ajout apporte un gain positif, on valide
            if(best_k != -1) {
                h_col(best_k) = 1;
                // Mise à jour de la reconstruction (OR Logique)
                for(int i=0; i<m; ++i) {
                    if(Fixed(i, best_k) == 1) current_recon(i) = 1;
                }
                improvement = true;
            }
        }

        Target.col(j) = h_col;

        // Calcul final de l'erreur Hamming
        double err = 0.0;
        for(int i=0; i<m; ++i) {
            if(x_target(i) != current_recon(i)) err += 1.0;
        }
        total_error += err;
    }

    return total_error;
}

struct AcdWorkspace {
    std::vector<double> breakpts;
    std::vector<int> p;     // Permutation pour le tri
    std::vector<int> indices;
    
    // Buffers pour les valeurs triées et calculs
    std::vector<double> sga;
    std::vector<double> aa;
    std::vector<double> bb;
    std::vector<double> bc_term; // bb - 2*bc
    std::vector<double> tl_term; // 2*(ab - ac)
    
    // Resize une seule fois par thread
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

// Implémentation fidèle de findmin de ACD_NMD.cpp
// Minimise ||c - max(0, b + ax)||^2
double findmin_workspace(const VectorXd& a_vec, const VectorXd& b_vec, const VectorXd& c_vec, AcdWorkspace& ws) {
    int m = (int)a_vec.size();
    int nnz = 0;
    
    // 1. Filtrage des zéros et calcul des breakpoints
    // Correspond à la boucle initiale de findmin dans ACD_NMD.cpp
    for(int i=0; i<m; ++i) {
        if(std::abs(a_vec(i)) > 1e-16) {
            ws.indices[nnz] = i;
            ws.breakpts[nnz] = -b_vec(i) / a_vec(i);
            ws.p[nnz] = nnz; // Prépare la permutation
            nnz++;
        }
    }
    
    if(nnz == 0) return 0.0;

    // 2. Tri des breakpoints
    // ACD_NMD.cpp utilise gsl_sort_index, ici std::sort fait pareil
    std::sort(ws.p.begin(), ws.p.begin() + nnz, [&](int i, int j) {
        return ws.breakpts[i] < ws.breakpts[j];
    });

    // 3. Initialisation des coefficients quadratiques (Zone "Far Left")
    double ti = 0.0; 
    double tl = 0.0; 
    double tq = 0.0;

    // Pré-calculs et initialisation simultanée
    // Dans ACD_NMD, "ti" commence par la somme des c^2
    // Ici on le fait à la volée ou on suppose que l'erreur constante ne change pas le min x
    // Mais pour être exact sur la valeur de fonction, il faut ajouter c^2.
    // NOTE: Pour trouver xopt, le terme constant sum(c^2) n'importe pas, 
    // mais il est utilisé dans la logique "Quadratic function far left" de ACD_NMD.cpp.
    
    // On doit accumuler sum(c^2) pour être rigoureusement identique à l'algo original
    // bien que mathématiquement inutile pour la dérivée.
    for(int i=0; i<nnz; ++i) {
        int idx = ws.indices[ws.p[i]]; // Index réel via permutation triée
        double cv = c_vec(idx);
        ti += cv * cv; 
    }
    // Note: ACD_NMD ajoute aussi les c[i] des indices où a[i]==0. 
    // Ici on simplifie car cela ne change pas la position du minimum.

    for(int k=0; k<nnz; ++k) {
        int original_idx_in_arrays = ws.p[k];
        int idx = ws.indices[original_idx_in_arrays];
        
        double av = a_vec(idx);
        double bv = b_vec(idx);
        double cv = c_vec(idx);
        
        // Stockage dans l'ordre trié pour la boucle de balayage
        double sgn = (av < 0 ? -1.0 : 1.0);
        ws.sga[k] = sgn;
        ws.aa[k] = av * av;
        ws.bc_term[k] = bv*bv - 2*bv*cv; // bb - 2bc
        ws.tl_term[k] = 2*av*bv - 2*av*cv; // 2ab - 2ac
        
        // Logique "Far Left" de ACD_NMD.cpp
        // Si a[i] < 0, alors pour x -> -inf, a*x -> +inf, donc ReLU est actif.
        if (av < -1e-16) {
            ti += ws.bc_term[k];         // + bb - 2bc
            tl += ws.tl_term[k];         // + 2ab - 2ac
            tq += ws.aa[k];              // + aa
        }
    }

    // Premier candidat : minimum de la parabole de gauche ou le premier breakpoint
    double xmin = (std::abs(tq) > 1e-16) ? -tl / (2 * tq) : 0.0;
    
    // "Which is the best: the min or the first breakpoint ?"
    double bp0 = ws.breakpts[ws.p[0]]; // Premier breakpoint trié
    double xopt = (xmin < bp0) ? xmin : bp0;
    
    double yopt = ti + tl * xopt + tq * xopt * xopt;

    // 4. Balayage des intervalles (Update dynamique)
    for(int k=0; k<nnz; ++k) {
        // Mise à jour des coefficients en traversant le breakpoint k
        // ACD_NMD.cpp : ti += sga * (bb - 2bc), etc.
        double sg = ws.sga[k];
        ti += sg * ws.bc_term[k];
        tl += sg * ws.tl_term[k];
        tq += sg * ws.aa[k];
        
        // Nouveau min local
        xmin = (std::abs(tq) > 1e-16) ? -tl / (2 * tq) : 0.0; // 0.0 fallback si plat
        
        // Bornes de l'intervalle courant
        double current_bp = ws.breakpts[ws.p[k]];
        double next_bp = (k + 1 < nnz) ? ws.breakpts[ws.p[k+1]] : 1e20; // +inf
        
        // "Check if xmin is between two breakpoints"
        double xt;
        if (xmin > current_bp && xmin < next_bp) {
            xt = xmin;
        } else {
            // Sinon on teste la borne (ici current_bp, logic de ACD_NMD)
            xt = current_bp; 
            // Note: ACD_NMD teste 'breakpts[i]' comme candidat si xmin hors borne.
        }
        
        double yval = ti + tl * xt + tq * xt * xt;
        if (yval < yopt) {
            yopt = yval;
            xopt = xt;
        }
    }
    
    return xopt;
}

// Fonction de remplacement pour solve_matrix_relu
double solve_matrix_relu(const MatrixXd& X, const MatrixXi& Fixed, MatrixXi& Target, int L, int U) {
    MatrixXd W = Fixed.cast<double>();
    int m = (int)X.rows();
    int n = (int)X.cols();
    int r = (int)Fixed.cols();

    double total_error = 0.0;
    int acd_iter = 10; // Suffisant pour convergence locale

    #pragma omp parallel 
    {
        // Allocation UNIQUE par thread (Workspace)
        // Ceci évite l'allocation répétée à chaque appel de findmin
        AcdWorkspace ws;
        ws.resize(m); // Pré-alloue pour la taille m (547 dans votre cas)

        #pragma omp for reduction(+:total_error)
        for (int j = 0; j < n; ++j) {
            VectorXd h_col = Target.col(j).cast<double>();
            VectorXd X_col = X.col(j);
            VectorXd wh_col = W * h_col; // Cache

            for(int iter=0; iter<acd_iter; ++iter) {
                bool changed = false;
                for(int k=0; k<r; ++k) {
                    double h_old = h_col(k);
                    
                    // b = wh - Wk * h_old
                    // Nous calculons b_vec explicitement. C'est le coût principal mais nécessaire.
                    // Pour optimiser, on pourrait le faire en une ligne Eigen optimisée.
                    VectorXd b_vec = wh_col - W.col(k) * h_old;
                    
                    // Appel avec le workspace du thread
                    double h_new = findmin_workspace(W.col(k), b_vec, X_col, ws);
                    
                    if(std::abs(h_new - h_old) > 1e-6) {
                        h_col(k) = h_new;
                        wh_col += W.col(k) * (h_new - h_old);
                        changed = true;
                    }
                }
                if(!changed) break; 
            }

            // Arrondi et Bornage
            VectorXi h_int = h_col.array().round().cast<int>().cwiseMax(L).cwiseMin(U);
            Target.col(j) = h_int;
            
            // Erreur finale
            VectorXd wh_final = (Fixed.cast<double>() * h_int.cast<double>()).cwiseMax(0.0);
            total_error += (X_col - wh_final).squaredNorm();
        }
    }

    return total_error;
}

// --- ALIGNEMENT ---

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

MatrixXi align_parents_cpp(MatrixXi W1, MatrixXi W2) {
    MatrixXi W2_copy = W2; MatrixXi H_dummy = MatrixXi::Zero(W2.cols(), 1); 
    align_in_place(W1, W2_copy, H_dummy);
    return W2_copy;
}

// --- BATCH GENERATION ---

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
        
        #pragma omp for schedule(dynamic)
        for (int i = 0; i < num_children; ++i) {
            // Sélection par Tournoi
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
            
            // Ordonnancement P1 = Meilleur
            if (best_f2 < best_f1) { std::swap(best_p1, best_p2); std::swap(best_f1, best_f2); }
            
            MatrixXi W1 = Pop_W[best_p1], H1 = Pop_H[best_p1];
            MatrixXi W2 = Pop_W[best_p2], H2 = Pop_H[best_p2];
            align_in_place(W1, W2, H2);
            
            // Crossover biaisé
            double prob_p1 = 0.5;
            if (best_f1 + best_f2 > 1e-9) prob_p1 = best_f2 / (best_f1 + best_f2); 
            if(prob_p1 < 0.5) prob_p1 = 0.5; 

            MatrixXi Child_W(m, r);
            MatrixXi Child_H(r, H1.cols());

            // if(crossover_mode == 0){ 
            //     for(int k=0; k<r; ++k) {
            //         if(dist_prob(gen) < prob_p1) { Child_W.col(k) = W1.col(k); Child_H.row(k) = H1.row(k); } 
            //         else { Child_W.col(k) = W2.col(k); Child_H.row(k) = H2.row(k); }
            //     }
            // } else { 
            //     Child_W = ((W1.cast<double>() * prob_p1 + W2.cast<double>() * (1.0-prob_p1))).array().round().cast<int>();
            //     Child_H = ((H1.cast<double>() * prob_p1 + H2.cast<double>() * (1.0-prob_p1))).array().round().cast<int>();
            // }

            for(int k=0; k<r; ++k) {
                if(dist_prob(gen) < prob_p1) { Child_W.col(k) = W1.col(k); Child_H.row(k) = H1.row(k); } 
                else { Child_W.col(k) = W2.col(k); Child_H.row(k) = H2.row(k); }
            }
            
            Child_W = Child_W.cwiseMax(LW).cwiseMin(UW);
            Child_H = Child_H.cwiseMax(LH).cwiseMin(UH);

            double col_mutation_intensity = 0.1; 

            for (int col = 0; col < r; ++col) {
                if (dist_prob(gen) < mutation_rate) {
                    for (int row = 0; row < m; ++row) {
                        if (dist_prob(gen) < col_mutation_intensity) {
                            int current_val = Child_W(row, col);
                            
                            if (mode_opti == "BMF") {
                                // BMF 
                                Child_W(row, col) = (current_val == LW) ? UW : LW;
                            } else {
                                // IMF 
                                int delta = (dist_prob(gen) < 0.5) ? 1 : -1;
                                int new_val = current_val + delta;
                                Child_W(row, col) = std::max(LW, std::min(UW, new_val));
                            }
                        }
                    }
                }
            }

            double f_obj = 0.0;
            
            int max_refine = 1;
            for(int iter=0; iter<max_refine; ++iter) {

                if(mode_opti == "IMF") f_obj = solve_matrix_imf(X, Child_W.cast<double>(), Child_H, LH, UH);
                else if(mode_opti == "BMF") f_obj = solve_matrix_bmf(X, Child_W, Child_H, LH, UH);
                else f_obj = solve_matrix_relu(X, Child_W, Child_H, LH, UH);

                MatrixXi WT = Child_W.transpose();
                if(mode_opti == "IMF") f_obj = solve_matrix_imf(XT, Child_H.transpose().cast<double>(), WT, LW, UW);
                else if(mode_opti == "BMF") f_obj = solve_matrix_bmf(XT, Child_H.transpose(), WT, LW, UW);
                else f_obj = solve_matrix_relu(XT, Child_H.transpose(), WT, LW, UW);
                Child_W = WT.transpose();
            }
            
            // if(mode_opti == "IMF") f_obj = solve_matrix_imf(X, Child_W.cast<double>(), Child_H, LH, UH);
            // else if(mode_opti == "BMF") f_obj = solve_matrix_bmf(X, Child_W, Child_H, LH, UH);
            // else f_obj = solve_matrix_relu(X, Child_W, Child_H, LH, UH);

            // Anti-Clone (léger)
            if (count_diff(Child_W, W1) == 0 || count_diff(Child_W, W2) == 0) {
                 int bits_to_flip = std::max(1, (int)(m * r * 0.01)); 
                 for(int k=0; k<bits_to_flip; ++k) {
                     int rc = (int)(dist_prob(gen) * r * m);
                     int c = rc / m; int r_idx = rc % m;
                     if(c < r && r_idx < m) Child_W(r_idx, c) = (Child_W(r_idx, c) == LW) ? UW : LW;
                 }
                 if(mode_opti == "BMF") f_obj = solve_matrix_bmf(X, Child_W, Child_H, LH, UH);
            }
            
            results[i] = make_tuple(Child_W, Child_H, f_obj, best_p1, best_p2, count_diff(Child_W, W1), count_diff(Child_W, W2));
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
    MatrixXi W = W_init, H = H_init;
    double f_obj = 1e20;
    MatrixXd XT = X.transpose();

    for (int global_iter = 0; global_iter < max_global_iters; ++global_iter) {
        auto current_time = chrono::high_resolution_clock::now();
        if (chrono::duration<double>(current_time - start_time).count() > time_limit_seconds) break;
        double prev_f = f_obj;
        
        if(mode_opti == "IMF") f_obj = solve_matrix_imf(X, W.cast<double>(), H, LH, UH);
        else if(mode_opti == "BMF") f_obj = solve_matrix_bmf(X, W, H, LH, UH);
        else f_obj = solve_matrix_relu(X, W, H, LH, UH);
        
        MatrixXi WT = W.transpose();
        if(mode_opti == "IMF") f_obj = solve_matrix_imf(XT, H.transpose().cast<double>(), WT, LW, UW);
        else if(mode_opti == "BMF") f_obj = solve_matrix_bmf(XT, H.transpose(), WT, LW, UW);
        else f_obj = solve_matrix_relu(XT, H.transpose(), WT, LW, UW);
        W = WT.transpose();

        if (abs(prev_f - f_obj) < 1e-3 && global_iter > 3) break;
    }
    return make_tuple(W, H, f_obj);
}

pair<MatrixXi, double> optimize_h_cpp(MatrixXd X, MatrixXd W, int LW, int UW, int LH, int UH, string mode_opti) {
    MatrixXi H = MatrixXi::Zero(W.cols(), X.cols());
    double f = 1e20;
    if (mode_opti == "IMF") f = solve_matrix_imf(X, W, H, LH, UH);
    else if (mode_opti == "BMF") f = solve_matrix_bmf(X, W.cast<int>(), H, LH, UH);
    else f = solve_matrix_relu(X, W.cast<int>(), H, LH, UH);
    return {H, f};
}

int get_aligned_distance(const MatrixXi& W1, MatrixXi W2) {
    MatrixXi H_dummy = MatrixXi::Zero(W2.cols(), 1); 
    align_in_place(W1, W2, H_dummy);
    return count_diff(W1, W2);
}

PYBIND11_MODULE(fast_solver, m) {
    m.doc() = "Solver C++ FactInZ v6.2 (Mutation locale +/- 1)";
    m.def("optimize_h", &optimize_h_cpp, py::call_guard<py::gil_scoped_release>());
    m.def("optimize_alternating", &optimize_alternating_cpp, py::call_guard<py::gil_scoped_release>());
    m.def("generate_children_batch", &generate_children_batch);
    m.def("align_parents_cpp", &align_parents_cpp, py::call_guard<py::gil_scoped_release>());
    m.def("get_aligned_distance", &get_aligned_distance, py::call_guard<py::gil_scoped_release>());
}