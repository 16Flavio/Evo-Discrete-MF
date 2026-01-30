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

VectorXi k_best_search_qr(const ColPivHouseholderQR<MatrixXd>& qr, const MatrixXd& R, const VectorXd& y, int L, int U, int K_beam) {
    int r = (int)R.rows();
    std::vector<BeamNode> current_level;
    current_level.push_back({0.0, std::vector<int>(r, 0)}); 
    
    for (int k = r - 1; k >= 0; --k) {
        std::priority_queue<BeamNode> best_candidates;
        
        for (const auto& node : current_level) {
            double val = y(k);
            for (int j = k + 1; j < r; ++j) {
                val -= R(k, j) * (double)node.h[j];
            }
            
            double diag = R(k, k);
            double center = 0.0;
            if (std::abs(diag) > 1e-9) center = val / diag;
            else center = (double)L;
            
            int center_int = (int)std::round(center);
            
            for (int offset = -2; offset <= 2; ++offset) {
                int candidate_val = center_int + offset;
                if (candidate_val < L || candidate_val > U) continue;
                
                double diff = (double)candidate_val - center;
                double term_cost = diff * diff * diag * diag;
                double new_total_cost = node.cost + term_cost;
                
                BeamNode new_node = node;
                new_node.cost = new_total_cost;
                new_node.h[k] = candidate_val;
                
                if (best_candidates.size() < (size_t)K_beam) {
                    best_candidates.push(new_node);
                } else if (new_total_cost < best_candidates.top().cost) {
                    best_candidates.pop();
                    best_candidates.push(new_node);
                }
            }
        }
        
        current_level.clear();
        while(!best_candidates.empty()) {
            current_level.push_back(best_candidates.top());
            best_candidates.pop();
        }
    }
    
    if (current_level.empty()) return VectorXi::Constant(r, L);
    
    auto best_it = std::min_element(current_level.begin(), current_level.end(), 
        [](const BeamNode& a, const BeamNode& b){ return a.cost < b.cost; });
        
    VectorXi z = VectorXi::Map(best_it->h.data(), r);
    return qr.colsPermutation() * z;
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

    int max_passes = 3;
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

void integer_cd_bmf(const MatrixXd& W, const VectorXd& X_col, VectorXi& h_int, int L, int U, mt19937& gen) {
    int r = (int)h_int.size();
    int m = (int)X_col.size();
    VectorXd wh = W * h_int.cast<double>();
    
    std::vector<int> indices(r);
    std::iota(indices.begin(), indices.end(), 0);

    for(int pass=0; pass<3; ++pass) {
        std::shuffle(indices.begin(), indices.end(), gen);
        bool changed = false;
        
        for(int k : indices) {
            int current_v = h_int(k);
            double best_err = -1.0;
            int best_v = current_v;

            // Tester les valeurs dans [L, U] autour de la valeur actuelle
            for(int v = L; v <= U; ++v) {
                double diff = (double)v - (double)current_v;
                double step_err = 0;
                
                for(int i=0; i<m; ++i) {
                    double new_wh_i = wh(i) + W(i, k) * diff;
                    double prediction = std::min(1.0, new_wh_i);
                    double d = X_col(i) - prediction;
                    step_err += d * d;
                    
                    // Astuce : Petite pénalité pour éviter les WH inutilement grands (> 1)
                    // Cela aide à choisir la plus petite valeur de H qui sature le pixel
                    if (new_wh_i > 1.0) step_err += 1e-5 * (new_wh_i - 1.0);
                }

                if (best_err < 0 || step_err < best_err) {
                    best_err = step_err;
                    best_v = v;
                }
            }

            if (best_v != current_v) {
                wh += W.col(k) * (double)(best_v - current_v);
                h_int(k) = best_v;
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
    MatrixXd W = Fixed.cast<double>();
    int n = (int)X.cols();
    int r = (int)Fixed.cols();

    ColPivHouseholderQR<MatrixXd> qr(W);
    MatrixXd R = qr.matrixQR().topRows(r).triangularView<Upper>();
    int K_beam = 8; 

    VectorXd W_norms_sq = W.colwise().squaredNorm(); 
    double total_error = 0.0;

    #pragma omp parallel reduction(+:total_error)
    {
        int thread_id = omp_get_thread_num();
        std::mt19937 gen(9781 + thread_id * 1337);

        #pragma omp for 
        for (int j = 0; j < n; ++j) {
            VectorXd X_col = X.col(j);
            VectorXd y = qr.householderQ().transpose() * X_col;

            VectorXi h_int = k_best_search_qr(qr, R, y, L, U, K_beam);
            
            integer_cd_imf(W, W_norms_sq, X_col, h_int, L, U, gen);

            Target.col(j) = h_int;
            total_error += (X_col - W * h_int.cast<double>()).squaredNorm();
        }
    }
    return total_error;
}

// SOLVEUR BMF MODIFIÉ : BEAM SEARCH + PRUNING + FLIP
double solve_matrix_bmf(const MatrixXd& X, const MatrixXi& Fixed, MatrixXi& Target, int L, int U) {
    MatrixXd W = Fixed.cast<double>();
    int n = (int)X.cols();
    int r = (int)Fixed.cols();

    // On utilise la même stratégie que IMF : QR + Beam Search + CD
    ColPivHouseholderQR<MatrixXd> qr(W);
    MatrixXd R = qr.matrixQR().topRows(r).triangularView<Upper>();
    int K_beam = 8; 

    VectorXd W_norms_sq = W.colwise().squaredNorm(); 
    double total_error = 0.0;

    #pragma omp parallel reduction(+:total_error)
    {
        int thread_id = omp_get_thread_num();
        std::mt19937 gen(9781 + thread_id * 1337);

        #pragma omp for 
        for (int j = 0; j < n; ++j) {
            VectorXd X_col = X.col(j);
            VectorXd y = qr.householderQ().transpose() * X_col;

            // 1. Recherche initiale via Beam Search (Logique IMF)
            VectorXi h_int = k_best_search_qr(qr, R, y, L, U, K_beam);
            
            // 2. Raffinement par Descente de Coordonnées Entière (Logique IMF)
            integer_cd_bmf(W, X_col, h_int, L, U, gen);

            Target.col(j) = h_int;
            
            // 3. Calcul de l'erreur finale avec la fonction SATURÉE : || X - min(1, WH) ||^2
            // On calcule d'abord WH
            VectorXd wh = W * h_int.cast<double>();
            // On applique la saturation min(1, x)
            for(int i = 0; i < wh.size(); ++i) {
                if (wh(i) > 1.0) wh(i) = 1.0;
                // Note: min(1, WH) dans le contexte BMF suppose souvent WH >= 0
            }
            
            total_error += (X_col - wh).squaredNorm();
        }
    }
    return total_error;
}

double solve

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
        
        if(mode_opti == "IMF") f_obj = solve_matrix_imf(X, W, H, LH, UH);
        else if(mode_opti == "BMF") f_obj = solve_matrix_bmf(X, W, H, LH, UH);
        else f_obj = solve_matrix_relu(X, W, H, LH, UH);
        
        MatrixXi WT = W.transpose();
        if(mode_opti == "IMF") f_obj = solve_matrix_imf(XT, H.transpose(), WT, LW, UW);
        else if(mode_opti == "BMF") f_obj = solve_matrix_bmf(XT, H.transpose(), WT, LW, UW);
        else f_obj = solve_matrix_relu(XT, H.transpose(), WT, LW, UW);
        W = WT.transpose();

        if (abs(prev_f - f_obj) < 1e-3 && global_iter > 2) break;
    }
    return make_tuple(W, H, f_obj);
}

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
            int best_p = -1; 
            double best_f = 1e20;
            
            for(int t=0; t<tournament_size; ++t) {
                int idx = dist_idx(gen);
                if (Pop_Fitness[idx] < best_f) { 
                    best_f = Pop_Fitness[idx]; 
                    best_p = idx; 
                }
            }
            
            MatrixXi Child_W = Pop_W[best_p];
            MatrixXi Child_H = Pop_H[best_p];
            
            for (int col = 0; col < r; ++col) {
                if (dist_prob(gen) < mutation_rate) {
                    for (int row = 0; row < m; ++row) {
                        if (mode_opti == "BMF") {
                            if (dist_prob(gen) < 0.1){
                                Child_W(row, col) = (Child_W(row, col) == LW) ? UW : LW;
                            }
                        } else {
                            int delta = (dist_prob(gen) < 0.5) ? 1 : -1;
                            int new_val = Child_W(row, col) + delta;
                            Child_W(row, col) = std::max(LW, std::min(UW, new_val));
                        }
                    }
                }
            }

            double f_obj = 0.0;
            int max_refine = 1;
            for(int iter=0; iter<max_refine; ++iter) {
                if(mode_opti == "IMF") f_obj = solve_matrix_imf(X, Child_W, Child_H, LH, UH);
                else if(mode_opti == "BMF") f_obj = solve_matrix_bmf(X, Child_W, Child_H, LH, UH);
                else f_obj = solve_matrix_relu(X, Child_W, Child_H, LH, UH);

                MatrixXi WT = Child_W.transpose();
                if(mode_opti == "IMF") f_obj = solve_matrix_imf(XT, Child_H.transpose(), WT, LW, UW);
                else if(mode_opti == "BMF") f_obj = solve_matrix_bmf(XT, Child_H.transpose(), WT, LW, UW);
                else f_obj = solve_matrix_relu(XT, Child_H.transpose(), WT, LW, UW);
                Child_W = WT.transpose();
            }
            
            results[i] = make_tuple(Child_W, Child_H, f_obj, best_p, best_p, count_diff(Child_W, Pop_W[best_p]), 0);
        }
    }
    return results;
}

pair<MatrixXi, double> optimize_h_cpp(MatrixXd X, MatrixXd W, int LW, int UW, int LH, int UH, string mode_opti) {
    MatrixXi W_int = W.cast<int>();
    MatrixXi H = MatrixXi::Zero(W.cols(), X.cols());
    double f = 1e20;
    if (mode_opti == "IMF") f = solve_matrix_imf(X, W_int, H, LH, UH);
    else if (mode_opti == "BMF") f = solve_matrix_bmf(X, W_int, H, LH, UH);
    else f = solve_matrix_relu(X, W_int, H, LH, UH);
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
    m.def("optimize_alternating", &optimize_alternating_cpp, py::call_guard<py::gil_scoped_release>());
    m.def("generate_children_batch", &generate_children_batch);
    m.def("align_parents_cpp", &align_parents_cpp, py::call_guard<py::gil_scoped_release>());
    m.def("get_aligned_distance", &get_aligned_distance, py::call_guard<py::gil_scoped_release>());
}