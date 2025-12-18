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

// --- STRUCTURES & HELPERS ---

struct TabuState {
    std::vector<int> list;
    int tenure;
    TabuState(int r, int t) : list(r, 0), tenure(t) {}
    bool is_tabu(int index, int iter) { return list[index] > iter; }
    
    // Modified to accept a random generator
    template<typename RNG>
    void make_tabu(int index, int iter, RNG& gen) {
        std::uniform_int_distribution<> dist(0, 1);
        list[index] = iter + tenure + dist(gen);
    }
    
    void clear() { std::fill(list.begin(), list.end(), 0); }
};

// Counts the number of differences (Hamming Distance) between two matrices
int count_diff(const MatrixXi& A, const MatrixXi& B) {
    return (A - B).cwiseAbs().cast<bool>().count();
}

// --- OPTIMISATION LOCALE (Gram Matrix) ---
// Solves min ||Target - Base * Result||^2 column by column using coordinate descent.
// Uses a precomputed Gram matrix (M = Base^T * Base) for efficiency.
double solve_matrix_step_gram(const MatrixXd& Target, const MatrixXd& Base, MatrixXi& Result, 
                              int L_res, int U_res, int effort) {
    
    int n_cols = Target.cols();
    int r = Base.cols();
    
    // Pré-calcul de la matrice de Gram (O(r^2))
    MatrixXd M = Base.transpose() * Base;
    M += MatrixXd::Identity(r, r) * 1e-9; // Régularisation
    MatrixXd V = Base.transpose() * Target;
    VectorXd M_diag = M.diagonal();

    // Solution continue initiale (LDLT) - Aide à guider le démarrage
    auto solver_small = M.ldlt();
    MatrixXd Float_Init = solver_small.solve(V);

    double total_final_error = 0.0;
    
    // Détection si déjà dans une région parallèle
    bool in_parallel = omp_in_parallel();

    #pragma omp parallel for schedule(dynamic, 8) reduction(+:total_final_error) if(!in_parallel)
    for (int j = 0; j < n_cols; ++j) {
        // Thread-local RNG
        std::mt19937 rng(12345 ^ (omp_get_thread_num() + 1) ^ j); 
        std::uniform_int_distribution<int> dist_r(0, r - 1);
        std::uniform_int_distribution<int> dist_4(0, 3);

        VectorXi current_vec = Result.col(j);
        
        // Initialisation si vide ou si effort élevé (pour sortir des minimums)
        if (current_vec.isZero()) {
            for (int k = 0; k < r; ++k) {
                int val = std::round(Float_Init(k, j));
                current_vec(k) = std::max(L_res, std::min(U_res, val));
            }
        }

        VectorXd gradient = (M * current_vec.cast<double>()) - V.col(j);
        VectorXi best_local_vec = current_vec;
        double current_obj_val = 0.0; // Delta score

        // Configuration Effort
        // 1=Light (Enfants), 2=Deep (Init/PR), 3=Ultra (Final)
        int max_restarts = (effort == 3) ? 4 : ((effort == 2) ? 2 : 0); // Increased restarts for effort 2
        int max_iter_per_run = (effort == 1) ? 35 : ((effort == 2) ? 200 : 400); // Increased iter for effort 2
        bool use_tabu = (effort > 0);

        if (use_tabu) {
            int tabu_tenure = std::max(3, r / 3);
            TabuState tabu(r, tabu_tenure);

            for (int restart = 0; restart <= max_restarts; ++restart) {
                int stagnation = 0;

                // --- SMART KICK (Restart Stratégique) ---
                if (restart > 0) {
                    current_vec = best_local_vec;
                    gradient = (M * current_vec.cast<double>()) - V.col(j);
                    
                    // On perturbe les indices à fort gradient
                    int n_perturb = std::max(1, r / 6);
                    for(int p=0; p<n_perturb; ++p) {
                        int k = -1;
                        double max_g = -1.0;
                        // Tournoi pour trouver une variable tendue
                        for(int t=0; t<3; ++t) {
                            int cand = dist_r(rng);
                            if (std::abs(gradient(cand)) > max_g) { max_g = std::abs(gradient(cand)); k=cand; }
                        }
                        
                        if (k != -1) {
                            // Shift dans la direction opposée au gradient
                            int shift = (gradient(k) > 0) ? -2 : 2;
                            if (dist_4(rng) == 0) shift = -shift; // Un peu de bruit
                            
                            int new_val = std::max(L_res, std::min(U_res, current_vec(k) + shift));
                            int step = new_val - current_vec(k);
                            if(step != 0) {
                                current_vec(k) = new_val;
                                gradient += step * M.col(k);
                            }
                        }
                    }
                    tabu.clear();
                }

                double best_run_val = 1e20;

                for (int iter = 0; iter < max_iter_per_run; ++iter) {
                    int best_k = -1; int best_step = 0; double best_delta = 1e20;
                    bool move_found = false;

                    for (int k = 0; k < r; ++k) {
                        double grad_k = gradient(k);
                        double m_kk = M_diag(k);
                        // Pas de Newton
                        double ideal_step = -grad_k / (m_kk + 1e-12);
                        
                        int cv = current_vec(k);
                        int c1 = std::floor(cv + ideal_step);
                        int c2 = std::ceil(cv + ideal_step);
                        int candidates[2] = {c1, c2};

                        for(int target : candidates) {
                            if(target < L_res) target = L_res;
                            if(target > U_res) target = U_res;
                            int step = target - cv;
                            if(step == 0) continue;

                            double delta = (step * step * m_kk) + (2 * step * grad_k);
                            
                            bool aspiration = (delta < -0.01 && (current_obj_val + delta < best_run_val));
                            if(aspiration || !tabu.is_tabu(k, iter)) {
                                if(delta < best_delta) {
                                    best_delta = delta; best_k = k; best_step = step; move_found = true;
                                }
                            }
                        }
                    }

                    if(move_found) {
                        current_vec(best_k) += best_step;
                        gradient += best_step * M.col(best_k);
                        current_obj_val += best_delta;
                        tabu.make_tabu(best_k, iter, rng);

                        if(current_obj_val < best_run_val) {
                            best_run_val = current_obj_val;
                            best_local_vec = current_vec;
                            stagnation = 0;
                        } else stagnation++;
                    } else break;

                    if(stagnation > (effort==1 ? 8 : 25)) break;
                }
            }
            current_vec = best_local_vec;
        } else {
            // Fallback Greedy (Effort 0)
             for (int iter = 0; iter < 10; ++iter) {
                int best_k = -1; int best_step = 0; double best_delta = -1e-9;
                bool improved = false;
                for (int k = 0; k < r; ++k) {
                     double delta_base = -gradient(k) / (M_diag(k) + 1e-12);
                     int target = std::round(current_vec(k) + delta_base);
                     target = std::max(L_res, std::min(U_res, target));
                     int step = target - current_vec(k);
                     if(step == 0) continue;
                     double delta = (step * step * M_diag(k)) + (2 * step * gradient(k));
                     if (delta < best_delta) { best_delta = delta; best_k = k; best_step = step; improved = true; }
                }
                if(improved) {
                    current_vec(best_k) += best_step;
                    gradient += best_step * M.col(best_k);
                } else break;
             }
        }

        Result.col(j) = current_vec;
        VectorXd final_res = Target.col(j) - Base * current_vec.cast<double>();
        total_final_error += final_res.squaredNorm();
    }

    return total_final_error;
}

// --- ALIGNMENT (Coupled W-H) ---
// Aligns the columns of W2 to match W1, and permutes H2 accordingly.
// This is done in-place to minimize memory allocations.
void align_in_place(const MatrixXi& W1, MatrixXi& W2_to_align, MatrixXi& H2_to_align) {
    int r = W1.cols();
    int m = W1.rows();
    MatrixXi W2_copy = W2_to_align;
    MatrixXi H2_copy = H2_to_align;
    
    std::vector<bool> used(r, false);
    
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
            H2_to_align.row(k1) = H2_copy.row(best_k2); // Sync H rows
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

// Calculates the aligned Hamming distance between W1 and W2.
int get_aligned_distance(const MatrixXi& W1, MatrixXi W2) {
    // Dummy H for distance check (we only care about W distance)
    MatrixXi H_dummy = MatrixXi::Zero(W2.cols(), 1); 
    align_in_place(W1, W2, H_dummy);
    return count_diff(W1, W2);
}

// --- BATCH GENERATION (Coupled Crossover) ---
// Generates a batch of children in parallel using OpenMP.
// Performs selection, crossover (Uniform or Mean), mutation (Swap, Reset, Noise), and local optimization.
std::vector<std::tuple<MatrixXi, MatrixXi, double, int, int, int, int>> generate_children_batch(
    const MatrixXd& X, 
    const std::vector<MatrixXi>& Pop_W, 
    const std::vector<MatrixXi>& Pop_H,
    const std::vector<double>& Pop_Fitness,
    int num_children,
    int tournament_size,
    double mutation_rate,
    int LW, int UW, int LH, int UH
) {
    std::vector<std::tuple<MatrixXi, MatrixXi, double, int, int, int, int>> results(num_children);
    int pop_size = Pop_W.size();
    int m = X.rows();
    int r = Pop_W[0].cols();
    
    #pragma omp parallel 
    {
        std::random_device rd;
        std::mt19937 gen(rd() ^ (omp_get_thread_num() + 1)); 
        std::uniform_int_distribution<> dist_idx(0, pop_size - 1);
        std::uniform_real_distribution<> dist_prob(0.0, 1.0);
        std::uniform_int_distribution<> dist_val_W(LW, UW);
        std::uniform_int_distribution<> dist_col(0, r - 1);
        std::uniform_int_distribution<> dist_row(0, m - 1);

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
            
            // Alignement COUPLE (W et H bougent ensemble)
            align_in_place(W1, W2, H2);

            MatrixXi Child_W(m, r);
            MatrixXi Child_H(r, H1.cols());

            // 2. Crossover Strategy (Adaptive)
            if (dist_prob(gen) < 0.6) {
                // UNIFORM CROSSOVER (Preserves Integer Structure better)
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
                Child_W = (W1 + W2).array().round() / 2;
                Child_H = (H1 + H2).array().round() / 2;
            }
            
            // Ensure bounds
            Child_W = Child_W.cwiseMax(LW).cwiseMin(UW);
            Child_H = Child_H.cwiseMax(LH).cwiseMin(UH);

            // 3. Mutation
            // a) Swap Mutation (W cols & H rows)
            if (dist_prob(gen) < 0.20) { // Increased slightly
                int c1 = dist_col(gen);
                int c2 = dist_col(gen);
                Child_W.col(c1).swap(Child_W.col(c2));
                Child_H.row(c1).swap(Child_H.row(c2));
            }

            // b) Greedy Column / Reset
            if (dist_prob(gen) < 0.30) { 
                int col = dist_col(gen);
                double type = dist_prob(gen);
                
                if (type < 0.3) {
                     // Reset to Lower Bound
                     for(int row=0; row<m; ++row) Child_W(row, col) = LW; 
                } else if (type < 0.6) {
                    // Random Reset
                    for(int row=0; row<m; ++row) Child_W(row, col) = dist_val_W(gen);
                } else {
                    // Multiplicative Mutation (Scaling)
                    double alpha = (dist_prob(gen) < 0.5) ? 0.5 : 2.0;
                    for(int row=0; row<m; ++row) {
                         int val = std::round(Child_W(row, col) * alpha);
                         Child_W(row, col) = std::max(LW, std::min(UW, val));
                    }
                }
            }
            
            // c) Noise Mutation (Pointwise)
            if (dist_prob(gen) < 0.7) { 
                for(int row=0; row<m; ++row) {
                    for(int col=0; col<r; ++col) {
                        if (dist_prob(gen) < mutation_rate) {
                            int shift = (dist_prob(gen) < 0.5) ? -1 : 1;
                            // Occasional larger jump
                            if (dist_prob(gen) < 0.1) shift *= 2;
                            
                            int val = Child_W(row, col) + shift;
                            Child_W(row, col) = std::max(LW, std::min(UW, val));
                        }
                    }
                }
            }

            // 4. Optimisation Rapide (Light Tabu)
            double f_obj = 0.0;
            int max_iters_child = 5; // Slightly increased
            
            for(int iter=0; iter<max_iters_child; ++iter) {
                f_obj = solve_matrix_step_gram(X, Child_W.cast<double>(), Child_H, LH, UH, 1);
                MatrixXd XT = X.transpose();
                MatrixXd HT = Child_H.transpose().cast<double>();
                MatrixXi WT = Child_W.transpose();
                double f_w = solve_matrix_step_gram(XT, HT, WT, LW, UW, 1);
                Child_W = WT.transpose();
                f_obj = f_w;
            }
            
            // Calculate distances to parents (Deterministic Crowding)
            int dist_p1 = count_diff(Child_W, W1);
            int dist_p2 = count_diff(Child_W, W2);

            results[i] = std::make_tuple(Child_W, Child_H, f_obj, best_p1, best_p2, dist_p1, dist_p2);
        }
    }

    return results;
}

// --- BINDINGS ---

// Wrapper for Alternating Coordinate Descent optimization.
// Optimizes W and H iteratively to minimize the objective function.
std::tuple<MatrixXi, MatrixXi, double> optimize_alternating_cpp(
    MatrixXd X, MatrixXi W_init, MatrixXi H_init, 
    int LW, int UW, int LH, int UH, 
    int max_global_iters, int effort, double time_limit_seconds) {

    auto start_time = std::chrono::high_resolution_clock::now();

    MatrixXi W = W_init;
    MatrixXi H = H_init;
    double f_obj = 1e20;
    MatrixXd XT = X.transpose();

    for (int global_iter = 0; global_iter < max_global_iters; ++global_iter) {
        auto current_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = current_time - start_time;
        if (elapsed.count() > time_limit_seconds) break;

        double prev_f = f_obj;
        f_obj = solve_matrix_step_gram(X, W.cast<double>(), H, LH, UH, effort);
        
        MatrixXd HT = H.transpose().cast<double>();
        MatrixXi WT = W.transpose();
        double f_w = solve_matrix_step_gram(XT, HT, WT, LW, UW, effort);
        W = WT.transpose();
        f_obj = f_w;

        if (std::abs(prev_f - f_obj) < 1e-3 && global_iter > 3) break;
    }
    return std::make_tuple(W, H, f_obj);
}

// Optimizes H for a fixed W.
std::pair<MatrixXi, double> optimize_h_cpp(MatrixXd X, MatrixXd W, int LW, int UW, int LH, int UH, bool use_vnd) {
    MatrixXi H = MatrixXi::Zero(W.cols(), X.cols());
    int effort = use_vnd ? 1 : 0; 
    double f = solve_matrix_step_gram(X, W, H, LH, UH, effort);
    return {H, f};
}

// Aligns W2 to W1 and returns the aligned W2.
MatrixXi align_parents_cpp(MatrixXi W1, MatrixXi W2) {
    MatrixXi W2_copy = W2;
    MatrixXi H_dummy = MatrixXi::Zero(W2.cols(), 1); // Dummy
    align_in_place(W1, W2_copy, H_dummy);
    return W2_copy;
}

int get_aligned_distance_binding(MatrixXi W1, MatrixXi W2) {
    return get_aligned_distance(W1, W2);
}

PYBIND11_MODULE(fast_solver, m) {
    m.doc() = "Solver C++ FactInZ v2.0 (Coupled Crossover + Optimized RNG)";
    m.def("optimize_h", &optimize_h_cpp, py::call_guard<py::gil_scoped_release>());
    m.def("optimize_alternating", &optimize_alternating_cpp, py::call_guard<py::gil_scoped_release>(),
          py::arg("X"), py::arg("W_init"), py::arg("H_init"), 
          py::arg("LW"), py::arg("UW"), py::arg("LH"), py::arg("UH"), 
          py::arg("max_global_iters"), py::arg("effort"), py::arg("time_limit_seconds") = 3600.0);
    m.def("align_parents_cpp", &align_parents_cpp, py::call_guard<py::gil_scoped_release>());
    m.def("generate_children_batch", &generate_children_batch, py::call_guard<py::gil_scoped_release>());
    m.def("get_aligned_distance", &get_aligned_distance_binding, py::call_guard<py::gil_scoped_release>());
}