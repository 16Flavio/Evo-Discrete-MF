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
    double tol = 1e-5;
    
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

// --- SOLVEURS SPÉCIALISÉS (RELU) ---

double solve_column_relu_sphere(
    const VectorXd& X_col, const MatrixXd& W,
    VectorXi& h_col, double LH, double UH, long long node_limit
) {
    int r = (int)W.cols();
    int m = (int)W.rows();
    
    // Init
    MatrixXd WtW = W.transpose() * W;
    VectorXd Wtx = W.transpose() * X_col;
    VectorXd h_cont = solve_continuous_bounded(WtW, Wtx, LH, UH);
    for(int i=0; i<r; ++i) h_col(i) = (int)std::round(h_cont(i));
    
    double best_relu_error = (X_col - (W * h_col.cast<double>()).cwiseMax(0.0)).squaredNorm();

    for(int iter=0; iter<2; ++iter) {
        VectorXd weights = VectorXd::Ones(m);
        VectorXd rec = W * h_col.cast<double>();
        for(int i=0; i<m; ++i) if (rec(i) <= 0 && X_col(i) <= 0) weights(i) = 1e-4;
        
        MatrixXd W_scaled = W;
        VectorXd sqrt_w = weights.cwiseSqrt();
        for(int i=0; i<m; ++i) W_scaled.row(i) *= sqrt_w(i);
        VectorXd X_scaled = X_col.cwiseProduct(sqrt_w);
        
        MatrixXd WtW_s = W_scaled.transpose() * W_scaled;
        VectorXd Wtx_s = W_scaled.transpose() * X_scaled;
        
        LLT<MatrixXd> llt(WtW_s);
        if (llt.info() == Eigen::Success) {
            MatrixXd R = llt.matrixU();
            VectorXd z = llt.matrixL().solve(Wtx_s);
            
            double best_quad_dist = (R * h_col.cast<double>() - z).squaredNorm();
            VectorXi temp_h = VectorXi::Zero(r);
            long long nodes = 0;

            auto relu_visitor = [&](const VectorXi& candidate_h, double quad_error) {
                if (quad_error < best_quad_dist) best_quad_dist = quad_error;

                double real_err = (X_col - (W * candidate_h.cast<double>()).cwiseMax(0.0)).squaredNorm();
                if (real_err < best_relu_error) {
                    best_relu_error = real_err;
                    h_col = candidate_h;
                }
            };

            sphere_decoder_recursive(r-1, r, 0.0, R, z, LH, UH, temp_h, best_quad_dist, nodes, node_limit, relu_visitor);
        }
    }
    return best_relu_error;
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
            VectorXi temp_h = VectorXi::Zero(r);
            long long nodes = 0;
            
            auto imf_visitor = [&](const VectorXi& candidate_h, double quad_error) {
                if (quad_error < best_dist_sq) {
                    best_dist_sq = quad_error;
                    h_col = candidate_h;
                }
            };

            sphere_decoder_recursive(r-1, r, 0.0, R, z, LH, UH, temp_h, best_dist_sq, nodes, node_limit, imf_visitor);
            
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

double solve_matrix_relu(const MatrixXd& X, const MatrixXi& Fixed, MatrixXi& Target, int L, int U) {
    MatrixXd W_d = Fixed.cast<double>();
    int n = (int)X.cols();
    double total_error = 0.0;
    long long node_limit = 20000;

    #pragma omp parallel for reduction(+:total_error)
    for (int j = 0; j < n; ++j) {
        VectorXi h_col = Target.col(j);
        double err = solve_column_relu_sphere(X.col(j), W_d, h_col, (double)L, (double)U, node_limit);
        Target.col(j) = h_col;
        total_error += err;
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
        uniform_int_distribution<> dist_val_W(LW, UW);
        uniform_int_distribution<> dist_col(0, r - 1);
        uniform_int_distribution<> dist_row(0, m - 1);

        #pragma omp for schedule(dynamic)
        for (int i = 0; i < num_children; ++i) {
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
            
            MatrixXi W1 = Pop_W[best_p1], H1 = Pop_H[best_p1];
            MatrixXi W2 = Pop_W[best_p2], H2 = Pop_H[best_p2];
            align_in_place(W1, W2, H2);
            
            MatrixXi Child_W(m, r);
            MatrixXi Child_H(r, H1.cols());

            if(crossover_mode == 0){ 
                if (dist_prob(gen) < 0.6) {
                    for(int k=0; k<r; ++k) {
                        if(dist_prob(gen) < 0.5) { Child_W.col(k) = W1.col(k); Child_H.row(k) = H1.row(k); }
                        else { Child_W.col(k) = W2.col(k); Child_H.row(k) = H2.row(k); }
                    }
                } else { 
                    Child_W = ((W1.cast<double>() + W2.cast<double>()) * 0.5).array().round().cast<int>();
                    Child_H = ((H1.cast<double>() + H2.cast<double>()) * 0.5).array().round().cast<int>();
                }
            } else { 
                 for(int k=0; k<r; ++k) {
                    if(dist_prob(gen) < 0.5) { Child_W.col(k) = W1.col(k); Child_H.row(k) = H1.row(k); }
                    else { Child_W.col(k) = W2.col(k); Child_H.row(k) = H2.row(k); }
                }
            }
            
            Child_W = Child_W.cwiseMax(LW).cwiseMin(UW);
            Child_H = Child_H.cwiseMax(LH).cwiseMin(UH);

            // --- MUTATION : PERTURBATION LOCALE ---
            // On parcourt chaque élément de Child_W et on le modifie de +/- 1 
            // avec une probabilité égale à mutation_rate
            for (int col = 0; col < r; ++col) {
                for (int row = 0; row < m; ++row) {
                    if (dist_prob(gen) < mutation_rate) {
                        int current_val = Child_W(row, col);
                        int delta = (dist_prob(gen) < 0.5) ? 1 : -1;
                        int new_val = current_val + delta;
                        
                        // Respect des bornes (clamping)
                        if (new_val >= LW && new_val <= UW) {
                            Child_W(row, col) = new_val;
                        }
                        // Si le mouvement nous sort des bornes, on tente l'inverse
                        else if (new_val < LW) {
                            Child_W(row, col) = current_val + 1; // Forcé vers le haut
                        } else {
                            Child_W(row, col) = current_val - 1; // Forcé vers le bas
                        }
                        // Re-clamping final par sécurité
                        Child_W(row, col) = std::max(LW, std::min(UW, Child_W(row, col)));
                    }
                }
            }

            double f_obj = 0.0;
            
            // Raffinement classique (3 itérations)
            for(int iter=0; iter<3; ++iter) {
                if(mode_opti == "IMF") f_obj = solve_matrix_imf(X, Child_W.cast<double>(), Child_H, LH, UH);
                else if(mode_opti == "BMF") f_obj = solve_matrix_bmf(X, Child_W, Child_H, LH, UH);
                else f_obj = solve_matrix_relu(X, Child_W, Child_H, LH, UH);

                MatrixXi WT = Child_W.transpose();
                if(mode_opti == "IMF") f_obj = solve_matrix_imf(XT, Child_H.transpose().cast<double>(), WT, LW, UW);
                else if(mode_opti == "BMF") f_obj = solve_matrix_bmf(XT, Child_H.transpose(), WT, LW, UW);
                else f_obj = solve_matrix_relu(XT, Child_H.transpose(), WT, LW, UW);
                Child_W = WT.transpose();
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