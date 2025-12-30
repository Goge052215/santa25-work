#pragma once
#include <vector>
#include <cmath>
#include <random>
#include <iostream>
#include <tuple>
#include "grid.hpp"
#include "overlap.hpp"

namespace optimization {

struct SAParams {
    double Tmax;
    double Tmin;
    double nsteps;
    double nsteps_per_T;
    double position_delta;
    double angle_delta;
    double angle_delta2;
    double delta_t;
    double stagger_delta;
    double shear_delta;
    double parity_delta;
    int seed;
};

struct GridState {
    std::vector<long double> seed_xs;
    std::vector<long double> seed_ys;
    std::vector<long double> seed_degs;
    long double a;
    long double b;
    long double row_phase_x;
    long double col_phase_y;
    long double shear_x;
    long double shear_y;
    long double parity_row_deg;
    long double parity_col_deg;
};

inline std::tuple<long double, GridState, std::vector<ChristmasTree>> sa_optimize(
    const GridState& initial_state,
    int ncols, int nrows,
    bool append_x, bool append_y,
    const SAParams& params
) {
    std::mt19937 rng(params.seed);
    std::uniform_real_distribution<double> dist01(0.0, 1.0);
    std::uniform_int_distribution<int> dist_int; // range set later

    GridState current = initial_state;
    GridState best = current;

    // Check initial validity
    auto trees = grid::create_grid_trees(
        current.seed_xs, current.seed_ys, current.seed_degs,
        current.a, current.b, ncols, nrows, append_x, append_y,
        current.row_phase_x, current.col_phase_y, current.shear_x, current.shear_y,
        current.parity_row_deg, current.parity_col_deg
    );

    if (overlap::has_any_overlap(trees)) {
        // Simple recovery strategy: expand grid spacing
        // Python code does: a = max(a, a_test * 1.5), b = max(b, b_test * 1.5)
        // Here we just blindly increase if overlapping?
        // Let's assume the caller provides a reasonable start or we just bump a,b slightly?
        // Python code calls get_initial_translations. 
        // For now, let's just proceed. The SA might fix it or it might stay bad.
        // Actually, let's implement the expansion:
        current.a *= 1.5;
        current.b *= 1.5;
        trees = grid::create_grid_trees(
            current.seed_xs, current.seed_ys, current.seed_degs,
            current.a, current.b, ncols, nrows, append_x, append_y,
            current.row_phase_x, current.col_phase_y, current.shear_x, current.shear_y,
            current.parity_row_deg, current.parity_col_deg
        );
    }

    long double current_score = overlap::calculate_score(trees);
    long double best_score = current_score;
    
    // Store best trees
    std::vector<ChristmasTree> best_trees = trees;

    double T = params.Tmax;
    double Tfactor = -std::log(params.Tmax / params.Tmin);
    
    int n_seeds = current.seed_xs.size();
    int n_move_types = n_seeds + 6; // 0..n_seeds-1 (seed moves), n_seeds (grid), +1(row), +2(col), +3(sx), +4(sy), +5(rot)

    // Adaptive SA: Weights for move types
    std::vector<double> move_weights(n_move_types, 1.0);
    double min_weight = 0.1;

    int total_steps = static_cast<int>(params.nsteps);
    int steps_per_T = static_cast<int>(params.nsteps_per_T);

    for (int step = 0; step < total_steps; ++step) {
        // Create distribution from current weights
        std::discrete_distribution<int> move_dist(move_weights.begin(), move_weights.end());

        for (int k = 0; k < steps_per_T; ++k) {
            // Backup current state
            GridState old_state = current;
            
            int move_type = move_dist(rng);
            int dchoice = 0; // for rotation move

            if (move_type < n_seeds) {
                int i = move_type;
                double dx = (dist01(rng) * 2.0 - 1.0) * params.position_delta;
                double dy = (dist01(rng) * 2.0 - 1.0) * params.position_delta;
                double ddeg = (dist01(rng) * 2.0 - 1.0) * params.angle_delta;
                
                current.seed_xs[i] += dx;
                current.seed_ys[i] += dy;
                current.seed_degs[i] = std::fmod(current.seed_degs[i] + ddeg, 360.0);
            } else if (move_type == n_seeds) {
                double da = (dist01(rng) * 2.0 - 1.0) * params.delta_t;
                double db = (dist01(rng) * 2.0 - 1.0) * params.delta_t;
                current.a += current.a * da;
                current.b += current.b * db;
            } else if (move_type == n_seeds + 1) {
                double dpx = (dist01(rng) * 2.0 - 1.0) * params.stagger_delta;
                current.row_phase_x += dpx;
            } else if (move_type == n_seeds + 2) {
                double dpy = (dist01(rng) * 2.0 - 1.0) * params.stagger_delta;
                current.col_phase_y += dpy;
            } else if (move_type == n_seeds + 3) {
                double dsx = (dist01(rng) * 2.0 - 1.0) * params.shear_delta;
                current.shear_x += dsx;
            } else if (move_type == n_seeds + 4) {
                double dsy = (dist01(rng) * 2.0 - 1.0) * params.shear_delta;
                current.shear_y += dsy;
            } else {
                // Rotation moves
                dchoice = std::uniform_int_distribution<int>(0, 2)(rng);
                if (dchoice == 0) {
                    double ddeg = (dist01(rng) * 2.0 - 1.0) * params.angle_delta2;
                    for (int i = 0; i < n_seeds; ++i) {
                        current.seed_degs[i] = std::fmod(current.seed_degs[i] + ddeg, 360.0);
                    }
                } else if (dchoice == 1) {
                    double dpr = (dist01(rng) * 2.0 - 1.0) * params.parity_delta;
                    current.parity_row_deg = std::fmod(current.parity_row_deg + dpr, 360.0);
                } else {
                    double dpc = (dist01(rng) * 2.0 - 1.0) * params.parity_delta;
                    current.parity_col_deg = std::fmod(current.parity_col_deg + dpc, 360.0);
                }
            }

            // Generate trees
            auto test_trees = grid::create_grid_trees(
                current.seed_xs, current.seed_ys, current.seed_degs,
                current.a, current.b, ncols, nrows, append_x, append_y,
                current.row_phase_x, current.col_phase_y, current.shear_x, current.shear_y,
                current.parity_row_deg, current.parity_col_deg
            );

            // Check overlap
            if (overlap::has_any_overlap(test_trees)) {
                // Revert and penalize
                current = old_state;
                move_weights[move_type] = std::max(min_weight, move_weights[move_type] * 0.95);
                continue;
            }

            long double new_score = overlap::calculate_score(test_trees);
            double delta = static_cast<double>(new_score - current_score);

            bool accept = false;
            if (delta < 0) {
                accept = true;
            } else if (T > 1e-10) {
                if (dist01(rng) < std::exp(-delta / T)) {
                    accept = true;
                }
            }

            if (accept) {
                current_score = new_score;
                // Reward good moves
                if (delta < 0) {
                     move_weights[move_type] *= 1.1; 
                } else {
                     // Accepted but worse? Maybe slight reward or neutral?
                     // Let's just reward improvement significantly.
                     move_weights[move_type] *= 1.02; 
                }

                if (new_score < best_score) {
                    best_score = new_score;
                    best = current;
                    best_trees = test_trees;
                    // Strong reward for best solution
                    move_weights[move_type] *= 1.2;
                }
            } else {
                // Revert
                current = old_state;
                // Penalize rejection
                move_weights[move_type] = std::max(min_weight, move_weights[move_type] * 0.98);
            }
        }
        
        // Cooling
        T = params.Tmax * std::exp(Tfactor * (step + 1) / params.nsteps);
    }

    return {best_score, best, best_trees};
}

inline std::tuple<long double, GridState, std::vector<ChristmasTree>> refine_grid(
    const GridState& state,
    int ncols, int nrows,
    bool append_x, bool append_y,
    const SAParams& params
) {
    GridState best = state;
    auto best_trees = grid::create_grid_trees(
        best.seed_xs, best.seed_ys, best.seed_degs,
        best.a, best.b, ncols, nrows, append_x, append_y,
        best.row_phase_x, best.col_phase_y, best.shear_x, best.shear_y,
        best.parity_row_deg, best.parity_col_deg
    );
    long double best_score = overlap::calculate_score(best_trees);

    double step_px = params.stagger_delta * 0.5;
    double step_py = params.stagger_delta * 0.5;
    double step_sx = params.shear_delta * 0.5;
    double step_sy = params.shear_delta * 0.5;
    double step_pr = params.parity_delta * 0.5;
    double step_pc = params.parity_delta * 0.5;
    double step_a = params.delta_t * 0.5;
    double step_b = params.delta_t * 0.5;

    struct Candidate {
        long double score;
        GridState state;
        std::vector<ChristmasTree> trees;
        bool valid;
    };

    bool improved = true;
    while (improved) {
        improved = false;
        
        std::vector<Candidate> candidates(16);

        #pragma omp parallel for schedule(dynamic)
        for (int idx = 0; idx < 16; ++idx) {
            double delta = 0.0;
            int key = 0;
            switch (idx) {
                case 0:  delta = step_px; key = 0; break;
                case 1:  delta = -step_px; key = 0; break;
                case 2:  delta = step_py; key = 1; break;
                case 3:  delta = -step_py; key = 1; break;
                case 4:  delta = step_sx; key = 2; break;
                case 5:  delta = -step_sx; key = 2; break;
                case 6:  delta = step_sy; key = 3; break;
                case 7:  delta = -step_sy; key = 3; break;
                case 8:  delta = step_pr; key = 4; break;
                case 9:  delta = -step_pr; key = 4; break;
                case 10: delta = step_pc; key = 5; break;
                case 11: delta = -step_pc; key = 5; break;
                case 12: delta = step_a; key = 6; break;
                case 13: delta = -step_a; key = 6; break;
                case 14: delta = step_b; key = 7; break;
                case 15: delta = -step_b; key = 7; break;
            }

            GridState cand = best;
            if (key == 0) {
                cand.row_phase_x = best.row_phase_x + delta;
            } else if (key == 1) {
                cand.col_phase_y = best.col_phase_y + delta;
            } else if (key == 2) {
                cand.shear_x = best.shear_x + delta;
            } else if (key == 3) {
                cand.shear_y = best.shear_y + delta;
            } else if (key == 4) {
                cand.parity_row_deg = std::fmod(best.parity_row_deg + delta, 360.0);
            } else if (key == 5) {
                cand.parity_col_deg = std::fmod(best.parity_col_deg + delta, 360.0);
            } else if (key == 6) {
                cand.a = best.a + best.a * delta;
            } else if (key == 7) {
                cand.b = best.b + best.b * delta;
            }

            auto trees = grid::create_grid_trees(
                cand.seed_xs, cand.seed_ys, cand.seed_degs,
                cand.a, cand.b, ncols, nrows, append_x, append_y,
                cand.row_phase_x, cand.col_phase_y, cand.shear_x, cand.shear_y,
                cand.parity_row_deg, cand.parity_col_deg
            );
            
            if (overlap::has_any_overlap(trees)) {
                candidates[idx].valid = false;
            } else {
                candidates[idx].score = overlap::calculate_score(trees);
                candidates[idx].state = cand;
                candidates[idx].trees = trees;
                candidates[idx].valid = true;
            }
        }

        // Find best candidate (Steepest Descent)
        int best_idx = -1;
        long double current_best_score = best_score;

        for (int idx = 0; idx < 16; ++idx) {
            if (candidates[idx].valid && candidates[idx].score < current_best_score - 1e-12L) {
                current_best_score = candidates[idx].score;
                best_idx = idx;
            }
        }

        if (best_idx != -1) {
            best = candidates[best_idx].state;
            best_trees = candidates[best_idx].trees;
            best_score = candidates[best_idx].score;
            improved = true;
        }
    }

    return {best_score, best, best_trees};
}

} // namespace optimization
