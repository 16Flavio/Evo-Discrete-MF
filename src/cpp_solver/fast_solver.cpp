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

// Counts the number of differences (Hamming Distance) between two matrices
int count_diff(const MatrixXi& A, const MatrixXi& B) {
    return (int)(A - B).cwiseAbs().cast<bool>().count();
}

// --- OPTIMISATION LOCALE ---

// --- 1. SOLVE IMF (Arrondi simple) ---

double solve_matrix_imf(const MatrixXd& Target, const MatrixXd& Base, MatrixXi& Result, int L_res, int U_res, int effort) {
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
        mt19937 rng(12345 ^ (omp_get_thread_num() + 1) ^ j); 
        uniform_int_distribution<int> dist_r(0, r - 1);
        uniform_int_distribution<int> dist_4(0, 3);

        VectorXi current_vec = Result.col(j);
        
        // Initialisation si vide ou si effort élevé (pour sortir des minimums)
        if (current_vec.isZero()) {
            for (int k = 0; k < r; ++k) {
                int val = round(Float_Init(k, j));
                current_vec(k) = max(L_res, min(U_res, val));
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
            int tabu_tenure = max(3, r / 3);
            TabuState tabu(r, tabu_tenure);

            for (int restart = 0; restart <= max_restarts; ++restart) {
                int stagnation = 0;

                // --- SMART KICK (Restart Stratégique) ---
                if (restart > 0) {
                    current_vec = best_local_vec;
                    gradient = (M * current_vec.cast<double>()) - V.col(j);
                    
                    // On perturbe les indices à fort gradient
                    int n_perturb = max(1, r / 6);
                    for(int p=0; p<n_perturb; ++p) {
                        int k = -1;
                        double max_g = -1.0;
                        // Tournoi pour trouver une variable tendue
                        for(int t=0; t<3; ++t) {
                            int cand = dist_r(rng);
                            if (abs(gradient(cand)) > max_g) { max_g = abs(gradient(cand)); k=cand; }
                        }
                        
                        if (k != -1) {
                            // Shift dans la direction opposée au gradient
                            int shift = (gradient(k) > 0) ? -2 : 2;
                            if (dist_4(rng) == 0) shift = -shift; // Un peu de bruit
                            
                            int new_val = max(L_res, min(U_res, current_vec(k) + shift));
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
                        int c1 = floor(cv + ideal_step);
                        int c2 = ceil(cv + ideal_step);
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
                     int target = round(current_vec(k) + delta_base);
                     target = max(L_res, min(U_res, target));
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

// --- 2. SOLVE BMF ---

double solve_matrix_bmf(const MatrixXd& X, const MatrixXi& Fixed, MatrixXi& Target, int L, int U) {
    int m = (int)X.rows();
    int n = (int)X.cols();
    int r = (int)Fixed.cols();
    
    MatrixXi P = Fixed * Target; 
    MatrixXd Pd = P.cast<double>();
    
    bool improved = true;
    int iter = 0;
    int max_iters = 10;
    
    while (improved && iter < max_iters) {
        improved = false;
        iter++;
        
        for (int j = 0; j < n; ++j) {       
            for (int k = 0; k < r; ++k) {   
                
                int current_val = Target(k, j);
                int new_val = 1 - current_val; 
                int delta = new_val - current_val; 
                
                double gain = 0.0;
                
                for (int i = 0; i < m; ++i) {
                    if (Fixed(i, k) != 0) {
                        double x_ij = X(i, j);
                        double p_ij = Pd(i, j);
                        double p_new_ij = p_ij + (double)(Fixed(i, k) * delta);
                        
                        gain += (x_ij - p_ij)*(x_ij - p_ij) - (x_ij - p_new_ij)*(x_ij - p_new_ij);
                    }
                }
                
                if (gain > 1e-6) { 
                    Target(k, j) = new_val;
                    for (int i = 0; i < m; ++i) {
                        if (Fixed(i, k) != 0) {
                            P(i, j) += Fixed(i, k) * delta;
                            Pd(i, j) = (double)P(i, j);
                        }
                    }
                    improved = true;
                }
            }
        }
    }
    return (X - Pd).squaredNorm();
}

// --- 3. SOLVE RELU ---

double solve_matrix_relu(const MatrixXd& X, const MatrixXi& Fixed, MatrixXi& Target, int L, int U) {
    int m = (int)X.rows();
    int n = (int)X.cols();
    int r = (int)Fixed.cols();
    
    MatrixXd Fd = Fixed.cast<double>();
    MatrixXd Td = Target.cast<double>();
    MatrixXd P = Fd * Td; 
    
    bool improved = true;
    int iter = 0;
    int max_iters = 15;
    
    while (improved && iter < max_iters) {
        improved = false;
        iter++;
        
        for (int k = 0; k < r; ++k) {
            for (int j = 0; j < n; ++j) {
                
                double numerator = 0.0;
                double denominator = 0.0;
                
                for (int i = 0; i < m; ++i) {
                    double fix_val = Fd(i, k);
                    if (abs(fix_val) < 1e-9) continue;
                    
                    double p_val = P(i, j);
                    
                    if (p_val > 0 || (p_val <= 0 && X(i, j) > 0)) {
                        numerator += (X(i, j) - p_val) * fix_val;
                        denominator += fix_val * fix_val;
                    }
                }
                
                if (denominator > 1e-9) {
                    double delta = numerator / denominator;
                    
                    double old_val = Td(k, j);
                    double new_val_raw = old_val + delta;
                    
                    int new_val_int = (int)round(new_val_raw);
                    if (new_val_int < L) new_val_int = L;
                    if (new_val_int > U) new_val_int = U;
                    
                    if (new_val_int != Target(k, j)) {
                        int diff = new_val_int - Target(k, j);
                        Target(k, j) = new_val_int;
                        Td(k, j) = (double)new_val_int;
                        
                        for (int i = 0; i < m; ++i) {
                            if (Fd(i, k) != 0) {
                                P(i, j) += Fd(i, k) * diff;
                            }
                        }
                        improved = true;
                    }
                }
            }
        }
    }
    return (X - P.cwiseMax(0.0)).squaredNorm();
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
                    f_obj = solve_matrix_imf(X, W_float, Child_H, LH, UH, 1);
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
                    f_w = solve_matrix_imf(XT, HT_float, WT, LW, UW, 1);
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
    int max_global_iters, int effort, double time_limit_seconds, string mode_opti) {

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
            f_obj = solve_matrix_imf(X, W_float, H, LH, UH, effort);
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
            f_w = solve_matrix_imf(XT, HT, WT, LW, UW, effort);
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

pair<MatrixXi, double> optimize_h_cpp(MatrixXd X, MatrixXd W, int LW, int UW, int LH, int UH, bool use_vnd, string mode_opti) {
    MatrixXi H = MatrixXi::Zero(W.cols(), X.cols());
    double f = 1e20;
    MatrixXi W_int = W.cast<int>();
    if (mode_opti == "IMF") {
        f = solve_matrix_imf(X, W, H, LH, UH, 3);
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
          py::arg("max_global_iters"), py::arg("effort"), py::arg("time_limit_seconds") = 3600.0,
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