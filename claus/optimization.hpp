#pragma once
#include <vector>
#include <cmath>
#include <random>
#include <iostream>
#include <tuple>
#include "grid.hpp"
#include "overlap.hpp"
#include "gpu/gpu_context.hpp"

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
        // Recovery: expand grid spacing
        current.a *= 1.5;
        current.b *= 1.5;
        trees = grid::create_grid_trees(
            current.seed_xs, current.seed_ys, current.seed_degs,
            current.a, current.b, ncols, nrows, append_x, append_y,
            current.row_phase_x, current.col_phase_y, 
            current.shear_x, current.shear_y,
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

    // Adaptive Step Sizes
    double scale_linear = 1.0;
    double scale_angle = 1.0;
    int accepted_moves = 0;
    int total_moves_in_window = 0;
    const int adaptation_window = 50; // Check every 50 moves

    int total_steps = static_cast<int>(params.nsteps);
    int steps_per_T = static_cast<int>(params.nsteps_per_T);

    for (int step = 0; step < total_steps; ++step) {
        // Create distribution from current weights
        std::discrete_distribution<int> move_dist(move_weights.begin(), move_weights.end());

        for (int k = 0; k < steps_per_T; ++k) {
            // Adaptive Step Update
            if (++total_moves_in_window >= adaptation_window) {
                double rate = (double)accepted_moves / total_moves_in_window;
                // Target ~0.4
                if (rate > 0.5) {
                    scale_linear *= 1.05;
                    scale_angle *= 1.05;
                } else if (rate < 0.25) {
                    scale_linear *= 0.95;
                    scale_angle *= 0.95;
                }
                // Clamp
                scale_linear = std::max(1e-3, std::min(scale_linear, 5.0));
                scale_angle = std::max(1e-3, std::min(scale_angle, 5.0));
                
                accepted_moves = 0;
                total_moves_in_window = 0;
            }

            // Backup current state
            GridState old_state = current;
            
            int move_type = move_dist(rng);
            int dchoice = 0; // for rotation move

            if (move_type < n_seeds) {
                int i = move_type;
                double dx = (dist01(rng) * 2.0 - 1.0) * params.position_delta * scale_linear;
                double dy = (dist01(rng) * 2.0 - 1.0) * params.position_delta * scale_linear;
                double ddeg = (dist01(rng) * 2.0 - 1.0) * params.angle_delta * scale_angle;
                
                current.seed_xs[i] += dx;
                current.seed_ys[i] += dy;
                current.seed_degs[i] = std::fmod(current.seed_degs[i] + ddeg, 360.0);
            } else if (move_type == n_seeds) {
                double da = (dist01(rng) * 2.0 - 1.0) * params.delta_t * scale_linear;
                double db = (dist01(rng) * 2.0 - 1.0) * params.delta_t * scale_linear;
                current.a += current.a * da;
                current.b += current.b * db;
            } else if (move_type == n_seeds + 1) {
                double dpx = (dist01(rng) * 2.0 - 1.0) * params.stagger_delta * scale_linear;
                current.row_phase_x += dpx;
            } else if (move_type == n_seeds + 2) {
                double dpy = (dist01(rng) * 2.0 - 1.0) * params.stagger_delta * scale_linear;
                current.col_phase_y += dpy;
            } else if (move_type == n_seeds + 3) {
                double dsx = (dist01(rng) * 2.0 - 1.0) * params.shear_delta * scale_linear;
                current.shear_x += dsx;
            } else if (move_type == n_seeds + 4) {
                double dsy = (dist01(rng) * 2.0 - 1.0) * params.shear_delta * scale_linear;
                current.shear_y += dsy;
            } else {
                // Rotation moves
                dchoice = std::uniform_int_distribution<int>(0, 2)(rng);
                if (dchoice == 0) {
                    double ddeg = (dist01(rng) * 2.0 - 1.0) * params.angle_delta2 * scale_angle;
                    for (int i = 0; i < n_seeds; ++i) {
                        current.seed_degs[i] = std::fmod(current.seed_degs[i] + ddeg, 360.0);
                    }
                } else if (dchoice == 1) {
                    double dpr = (dist01(rng) * 2.0 - 1.0) * params.parity_delta * scale_angle;
                    current.parity_row_deg = std::fmod(current.parity_row_deg + dpr, 360.0);
                } else {
                    double dpc = (dist01(rng) * 2.0 - 1.0) * params.parity_delta * scale_angle;
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

inline std::tuple<long double, std::vector<ChristmasTree>> sa_optimize_individual(
    const std::vector<ChristmasTree>& initial_trees,
    const SAParams& params
) {
    std::mt19937 rng(params.seed);
    std::uniform_real_distribution<double> dist01(0.0, 1.0);
    
    // Work in-place on current_trees to avoid copying vector every step
    std::vector<ChristmasTree> current_trees = initial_trees;
    std::vector<ChristmasTree> best_trees = current_trees;
    
    long double current_main_score = overlap::calculate_score(current_trees);
    long double current_secondary_score = overlap::calculate_moment_of_inertia(current_trees);
    
    // Weight for secondary score. 
    double lambda = 1e-4; 
    
    long double current_total_score = current_main_score + lambda * current_secondary_score;
    long double best_total_score = current_total_score;
    long double best_main_score = current_main_score;

    double T = params.Tmax;
    double Tfactor = -std::log(params.Tmax / params.Tmin);
    int total_steps = static_cast<int>(params.nsteps);
    int steps_per_T = static_cast<int>(params.nsteps_per_T);
    int n_trees = (int)current_trees.size();

    // Adaptive Step Sizes
    double current_pos_delta = params.position_delta;
    double current_angle_delta = params.angle_delta;
    int accepted_moves = 0;
    int total_moves_in_window = 0;
    const int adaptation_window = 100;

    for (int step = 0; step < total_steps; ++step) {
        for (int k = 0; k < steps_per_T; ++k) {
            // Adaptive Step Update
            if (++total_moves_in_window >= adaptation_window) {
                double rate = (double)accepted_moves / total_moves_in_window;
                // Target acceptance ~0.4-0.5. 
                // If too high, step is too small -> Increase.
                // If too low, step is too large -> Decrease.
                if (rate > 0.5) {
                    current_pos_delta *= 1.05;
                    current_angle_delta *= 1.05;
                } else if (rate < 0.3) {
                    current_pos_delta *= 0.95;
                    current_angle_delta *= 0.95;
                }
                // Clamp
                current_pos_delta = std::max(1e-4, std::min(current_pos_delta, 5.0));
                current_angle_delta = std::max(1e-2, std::min(current_angle_delta, 45.0));
                
                accepted_moves = 0;
                total_moves_in_window = 0;
            }

            // Pick a move type: 0 = Perturb, 1 = Compress
            int move_type = (dist01(rng) < 0.05) ? 1 : 0; // 5% chance to compress
            
            if (move_type == 1) {
                 // Compression Move: Scale everything down slightly
                 double scale = 1.0 - (current_pos_delta * 0.1); // Small shrinkage
                 std::vector<ChristmasTree> backup = current_trees;
                 bool possible = true;
                 
                 for(auto& t : current_trees) {
                     t.center_x *= scale;
                     t.center_y *= scale;
                     t.angle_deg = t.angle_deg; // Rotation invariant
                     t = ChristmasTree(t.center_x, t.center_y, t.angle_deg);
                 }
                 
                 if (overlap::has_any_overlap(current_trees)) {
                     current_trees = backup;
                     possible = false;
                 }
                 
                 if (possible) {
                    long double new_main = overlap::calculate_score(current_trees);
                    long double new_sec = overlap::calculate_moment_of_inertia(current_trees);
                    long double new_total = new_main + lambda * new_sec;
                     
                    // Always accept compression if valid? Yes, it reduces objective.
                    current_main_score = new_main;
                    current_secondary_score = new_sec;
                    current_total_score = new_total;
                     
                    if (current_main_score < best_main_score) {
                        best_main_score = current_main_score;
                        best_total_score = current_total_score;
                        best_trees = current_trees;
                    }
                    accepted_moves++;
                }
            } else {
                // Perturbation Move
                int idx = std::uniform_int_distribution<int>(0, n_trees - 1)(rng);
                ChristmasTree old_tree = current_trees[idx];

                double dx = (dist01(rng) * 2.0 - 1.0) * current_pos_delta;
                double dy = (dist01(rng) * 2.0 - 1.0) * current_pos_delta;
                double ddeg = (dist01(rng) * 2.0 - 1.0) * current_angle_delta;
                
                current_trees[idx].center_x += dx;
                current_trees[idx].center_y += dy;
                current_trees[idx].angle_deg = std::fmod(current_trees[idx].angle_deg + ddeg, 360.0);
                
                current_trees[idx] = ChristmasTree(
                    current_trees[idx].center_x, 
                    current_trees[idx].center_y, 
                    current_trees[idx].angle_deg
                );
                
                bool valid = true;
                if (current_trees[idx].center_x < -100.0L || current_trees[idx].center_x > 100.0L ||
                    current_trees[idx].center_y < -100.0L || current_trees[idx].center_y > 100.0L) {
                    valid = false;
                } else {
                    if (overlap::has_overlap_with_others(current_trees, idx)) {
                        valid = false;
                    }
                }

                if (!valid) {
                    current_trees[idx] = old_tree;
                    continue;
                }

                long double new_main = overlap::calculate_score(current_trees);
                long double new_sec = overlap::calculate_moment_of_inertia(current_trees);
                long double new_total = new_main + lambda * new_sec;
                
                double delta = static_cast<double>(new_total - current_total_score);

                bool accept = false;
                if (delta < 0) {
                    accept = true;
                } else if (T > 1e-10) {
                    if (dist01(rng) < std::exp(-delta / T)) {
                        accept = true;
                    }
                }

                if (accept) {
                    current_main_score = new_main;
                    current_secondary_score = new_sec;
                    current_total_score = new_total;
                    
                    if (current_main_score < best_main_score) {
                        best_main_score = current_main_score;
                        best_total_score = current_total_score;
                        best_trees = current_trees;
                    }
                    accepted_moves++;
                } else {
                    current_trees[idx] = old_tree;
                }
            }
        }
        T = params.Tmax * std::exp(Tfactor * (step + 1) / params.nsteps);
    }
    
    // std::cout << "SA Finished. Best Main: " << best_main_score << " Initial: " << overlap::calculate_score(initial_trees) << std::endl;
    return {best_main_score, best_trees};
}

inline std::vector<ChristmasTree> coordinate_descent_polish(std::vector<ChristmasTree> trees) {
    long double current_score = overlap::calculate_score(trees);
    bool improved = true;
    double step_size = 0.1;
    
    // Iteratively refine with decreasing step size
    while (step_size > 1e-6) {
        improved = false;
        // Try to move each tree in 4 directions + rotation
        for (size_t i = 0; i < trees.size(); ++i) {
            ChristmasTree original = trees[i];
            
            // Candidates: dx+, dx-, dy+, dy-, rot+, rot-
            struct Move { double dx; double dy; double ddeg; };
            std::vector<Move> moves = {
                {step_size, 0, 0}, {-step_size, 0, 0},
                {0, step_size, 0}, {0, -step_size, 0},
                {0, 0, step_size * 10.0}, {0, 0, -step_size * 10.0} // Rotation scale
            };
            
            for (const auto& m : moves) {
                trees[i].center_x += m.dx;
                trees[i].center_y += m.dy;
                trees[i].angle_deg = std::fmod(trees[i].angle_deg + m.ddeg, 360.0);
                trees[i] = ChristmasTree(
                    trees[i].center_x, 
                    trees[i].center_y, 
                    trees[i].angle_deg
                );
                
                bool valid = true;
                if (trees[i].center_x < -100.0L || trees[i].center_x > 100.0L ||
                    trees[i].center_y < -100.0L || trees[i].center_y > 100.0L) {
                    valid = false;
                } else if (overlap::has_overlap_with_others(trees, i)) {
                    valid = false;
                }
                
                if (valid) {
                    long double new_score = overlap::calculate_score(trees);
                    if (new_score < current_score) {
                        current_score = new_score;
                        improved = true;
                        // Keep the change and continue to next tree/move
                        // Greedily accept first improvement or best? 
                        // Greedy first is faster.
                        original = trees[i]; // Update baseline for next moves
                    } else {
                        // Revert
                        trees[i] = original;
                    }
                } else {
                    // Revert
                    trees[i] = original;
                }
            }
        }
        
        // If no improvement at this resolution, shrink step
        if (!improved) {
            step_size *= 0.5;
        }
    }
    return trees;
}

inline std::vector<ChristmasTree> compact_trees(std::vector<ChristmasTree> trees, int steps = 200, double initial_step = 0.01) {
    double step_size = initial_step;
    std::mt19937 rng(123);
    std::uniform_real_distribution<double> djitter(-1.0, 1.0);

    for (int s = 0; s < steps; ++s) {
        bool any_moved = false;
        // Shuffle order
        std::vector<int> indices(trees.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), rng);

        for (int i : indices) {
            ChristmasTree original = trees[i];
            
            // Vector to center
            double dx = -trees[i].center_x;
            double dy = -trees[i].center_y;
            double len = std::sqrt(dx*dx + dy*dy);
            if (len < 1e-9) continue;
            
            dx /= len;
            dy /= len;
            
            // Add slight jitter to direction to avoid getting stuck
            if (s % 10 == 0) {
                dx += djitter(rng) * 0.5;
                dy += djitter(rng) * 0.5;
                double nlen = std::sqrt(dx*dx + dy*dy);
                dx /= nlen; dy /= nlen;
            }

            trees[i].center_x += dx * step_size;
            trees[i].center_y += dy * step_size;
            trees[i] = ChristmasTree(trees[i].center_x, trees[i].center_y, trees[i].angle_deg);
            
            if (overlap::has_overlap_with_others(trees, i)) {
                trees[i] = original; // Revert
            } else {
                any_moved = true;
            }
        }
        
        // Adaptive step size
        if (!any_moved) {
            step_size *= 0.5;
            if (step_size < 1e-6) break;
        } else if (s % 50 == 0) {
             // Occasionally reset step size to try larger jumps again?
             // Or just keep it.
        }
    }
    return trees;
}

// Physics-based optimization: Apply repulsion forces to separate overlapping trees
inline std::vector<ChristmasTree> physics_polish(std::vector<ChristmasTree> trees, int steps = 1000, double initial_lr = 0.01) {
    // Try GPU acceleration first
    if (GpuContext::getInstance().is_valid()) {
        // std::cout << "Using GPU physics polish" << std::endl;
        return GpuContext::getInstance().physics_polish(trees, steps, initial_lr);
    }

    size_t n = trees.size();
    if (n < 2) return trees;

    // Parameters
    double repulsion_strength = 1.0;
    double gravity_strength = 0.001; // Pull towards center to keep compact
    double learning_rate = initial_lr;
    double decay = 0.999;

    for (int s = 0; s < steps; ++s) {
        bool any_overlap = false;
        std::vector<std::pair<double, double>> forces(n, {0.0, 0.0});

        // 1. Compute forces
        for (size_t i = 0; i < n; ++i) {
            auto box_i = trees[i].aabb();
            
            // Gravity (Centripetal force)
            forces[i].first -= trees[i].center_x * gravity_strength;
            forces[i].second -= trees[i].center_y * gravity_strength;

            for (size_t j = 0; j < n; ++j) {
                if (i == j) continue;
                
                // Repulsion only if close
                double dx = trees[i].center_x - trees[j].center_x;
                double dy = trees[i].center_y - trees[j].center_y;
                double dist_sq = dx*dx + dy*dy;
                
                // Effective radius approximation (trees are roughly size 1x1)
                // If dist < 2.0, they might overlap.
                if (dist_sq < 4.0) {
                    double dist = std::sqrt(dist_sq);
                    if (dist < 1e-6) {
                        dx = (double(rand()) / RAND_MAX) - 0.5;
                        dy = (double(rand()) / RAND_MAX) - 0.5;
                        dist = 1e-3;
                    }
                    
                    bool is_overlapping = false;
                    auto box_j = trees[j].aabb();
                    if (overlap::boxes_overlap(box_i, box_j)) {
                        if (overlap::polygons_strict_overlap(trees[i], trees[j])) {
                            is_overlapping = true;
                            any_overlap = true;
                        }
                    }

                    if (is_overlapping) {
                        // Strong Repulsion
                        double force = repulsion_strength * (2.0 - dist) / dist; // Simple linear spring
                        forces[i].first += dx * force;
                        forces[i].second += dy * force;
                    } else if (dist < 1.2) {
                        // Weak Repulsion buffer
                        double force = repulsion_strength * 0.1 * (1.2 - dist) / dist;
                        forces[i].first += dx * force;
                        forces[i].second += dy * force;
                    }
                }
            }
        }

        // 2. Apply forces
        if (!any_overlap && s > 100) {
            // Early exit if stable and valid? 
            // No, keep compressing via gravity.
        }

        for (size_t i = 0; i < n; ++i) {
            trees[i].center_x += forces[i].first * learning_rate;
            trees[i].center_y += forces[i].second * learning_rate;
            
            // Bounds check
            if (trees[i].center_x < -100.0) trees[i].center_x = -100.0;
            if (trees[i].center_x > 100.0) trees[i].center_x = 100.0;
            if (trees[i].center_y < -100.0) trees[i].center_y = -100.0;
            if (trees[i].center_y > 100.0) trees[i].center_y = 100.0;
        }

        learning_rate *= decay;
        if (learning_rate < 1e-6) break;
    }
    
    return trees;
}

// Soft Constraint Optimization: Shrink box, allow overlap, minimize overlap
inline std::vector<ChristmasTree> squeeze_optimization(std::vector<ChristmasTree> trees, double shrink_factor = 0.02, int steps = 5000) {
    // 1. Shrink
    double scale = 1.0 - shrink_factor;
    for(auto& t : trees) {
        t.center_x *= scale;
        t.center_y *= scale;
        t = ChristmasTree(t.center_x, t.center_y, t.angle_deg);
    }
    
    // 2. Resolve Overlaps via SA
    SAParams params;
    params.nsteps = steps;
    params.Tmax = 10.0; // High temp to cross barriers
    params.Tmin = 0.01;
    params.seed = 12345;
    params.position_delta = 0.1;
    params.angle_delta = 5.0;
    
    std::mt19937 rng(params.seed);
    std::uniform_real_distribution<double> dist01(0.0, 1.0);
    
    // Adaptive Step Sizes
    double scale_linear = 1.0;
    double scale_angle = 1.0;
    int accepted_moves = 0;
    int total_moves_in_window = 0;
    const int adaptation_window = 100;

    // Initial cost: Proxy objective
    // Cost = OverlapPairs * 1000 + Repulsion + Compaction
    
    // Incremental Cost Calculation
    
    // Calculate cost contribution of a single tree 'idx' against all others
    auto get_tree_contribution = [&](const std::vector<ChristmasTree>& t, int idx) {
        double overlaps = 0.0;
        double repulsion = 0.0;
        double compaction = 0.0;
        
        auto box_idx = t[idx].aabb();

        // Check against all other trees
        for(size_t j=0; j<t.size(); ++j) {
            if((int)j == idx) continue;
            
            // AABB Check
            auto box_j = t[j].aabb();
            if (overlap::boxes_overlap(box_idx, box_j)) {
                 // Repulsion (approximate distance for gradient)
                 double dx = t[idx].center_x - t[j].center_x;
                 double dy = t[idx].center_y - t[j].center_y;
                 double d2 = dx*dx + dy*dy;
                 double dist = std::sqrt(d2);
                 if (dist < 0.8) {
                     repulsion += (0.8 - dist) * 10.0;
                 }
                 
                 // Strict Overlap
                 if (overlap::polygons_strict_overlap(t[idx], t[j])) {
                     overlaps += 1.0;
                 }
            } else {
                // Even if AABB doesn't overlap, check distance for repulsion field
                double dx = t[idx].center_x - t[j].center_x;
                double dy = t[idx].center_y - t[j].center_y;
                double d2 = dx*dx + dy*dy;
                if (d2 < 1.0) { // Interaction radius
                     double dist = std::sqrt(d2);
                     if (dist < 0.8) {
                         repulsion += (0.8 - dist) * 10.0;
                     }
                }
            }
        }
        
        compaction = (t[idx].center_x * t[idx].center_x + t[idx].center_y * t[idx].center_y) * 0.001;
        
        return overlaps * 10000.0 + repulsion + compaction;
    };
    
    // Initial Cost
    double current_cost = 0.0;
    for(size_t i=0; i<trees.size(); ++i) {
        current_cost += get_tree_contribution(trees, (int)i);
    }
    
    double best_cost = current_cost;
    std::vector<ChristmasTree> best_trees = trees;
    
    double T = params.Tmax;
    double Tfactor = -std::log(params.Tmax / params.Tmin);
    
    for(int s=0; s<steps; ++s) {
        if (current_cost == 0) break; // Might not be exactly 0 due to compaction/repulsion
        
        // Adaptive Step Update
        if (++total_moves_in_window >= adaptation_window) {
            double rate = (double)accepted_moves / total_moves_in_window;
            // Target ~0.4
            if (rate > 0.5) {
                scale_linear *= 1.05;
                scale_angle *= 1.05;
            } else if (rate < 0.25) {
                scale_linear *= 0.95;
                scale_angle *= 0.95;
            }
            // Clamp
            scale_linear = std::max(1e-3, std::min(scale_linear, 5.0));
            scale_angle = std::max(1e-3, std::min(scale_angle, 5.0));
            
            accepted_moves = 0;
            total_moves_in_window = 0;
        }

        int idx = std::uniform_int_distribution<int>(0, trees.size()-1)(rng);
        ChristmasTree old_tree = trees[idx];
        
        // Calculate OLD contribution of this tree
        double old_contrib = get_tree_contribution(trees, idx);
        
        // Guided Mutation: Calculate repulsion gradient from overlaps
        double grad_x = 0.0;
        double grad_y = 0.0;
        auto box_idx = trees[idx].aabb();
        for(size_t j=0; j<trees.size(); ++j) {
            if((int)j == idx) continue;
            // Use AABB check first
            auto box_j = trees[j].aabb();
            if(overlap::boxes_overlap(box_idx, box_j)) {
                 // Check if strict overlap or just close
                 if (overlap::polygons_strict_overlap(trees[idx], trees[j])) {
                     double diff_x = trees[idx].center_x - trees[j].center_x;
                     double diff_y = trees[idx].center_y - trees[j].center_y;
                     double d2 = diff_x*diff_x + diff_y*diff_y;
                     
                     if (d2 < 1e-9) {
                         // Exact overlap (duplicate): Strong random repulsion
                         grad_x += (dist01(rng) * 2.0 - 1.0) * 100.0;
                         grad_y += (dist01(rng) * 2.0 - 1.0) * 100.0;
                     } else {
                         // Strong repulsion for overlap
                         grad_x += diff_x / d2;
                         grad_y += diff_y / d2;
                     }
                 }
            }
        }
        
        // Perturb
        double dx = (dist01(rng)*2.0-1.0) * params.position_delta * scale_linear;
        double dy = (dist01(rng)*2.0-1.0) * params.position_delta * scale_linear;
        double ddeg = (dist01(rng)*2.0-1.0) * params.angle_delta * scale_angle;
        
        // Apply gradient bias if overlapping
        double g_len = std::sqrt(grad_x*grad_x + grad_y*grad_y);
        if(g_len > 1e-9) {
            grad_x /= g_len;
            grad_y /= g_len;
            // Bias the move significantly if overlapping
            double bias_strength = 0.8; 
            dx = dx * (1.0 - bias_strength) + grad_x * params.position_delta * scale_linear * bias_strength;
            dy = dy * (1.0 - bias_strength) + grad_y * params.position_delta * scale_linear * bias_strength;
        }

        trees[idx].center_x += dx;
        trees[idx].center_y += dy;
        trees[idx].angle_deg = std::fmod(trees[idx].angle_deg + ddeg, 360.0);
        trees[idx] = ChristmasTree(trees[idx].center_x, trees[idx].center_y, trees[idx].angle_deg);
        
        // Check bounds
        if (trees[idx].center_x < -100 || trees[idx].center_x > 100 || 
            trees[idx].center_y < -100 || trees[idx].center_y > 100) {
            trees[idx] = old_tree;
            continue;
        }
        
        // Calculate NEW contribution
        double new_contrib = get_tree_contribution(trees, idx);
        
        double new_total_cost = current_cost - old_contrib + new_contrib;
        double delta = new_total_cost - current_cost;
        
        if (delta < 0 || dist01(rng) < std::exp(-delta/T)) {
            current_cost = new_total_cost;
            if (current_cost < best_cost) {
                best_cost = current_cost;
                best_trees = trees;
            }
            accepted_moves++;
        } else {
            trees[idx] = old_tree;
        }
        
        T = params.Tmax * std::exp(Tfactor * (s+1) / params.nsteps);
    }
    
    // Return best found. If best_cost == 0, we successfully squeezed.
    // If best_cost > 0, we failed to resolve, caller should probably discard or try less shrink.
    return best_trees;
}

inline std::vector<ChristmasTree> crossover_spatial(
    const std::vector<ChristmasTree>& p1,
    const std::vector<ChristmasTree>& p2,
    std::mt19937& rng
) {
    std::uniform_real_distribution<double> dist01(0.0, 1.0);
    // Determine bounds from p1 (assuming p1/p2 similar bounds)
    double min_x = 1e9, max_x = -1e9, min_y = 1e9, max_y = -1e9;
    for(const auto& t : p1) {
        min_x = std::min(min_x, (double)t.center_x);
        max_x = std::max(max_x, (double)t.center_x);
        min_y = std::min(min_y, (double)t.center_y);
        max_y = std::max(max_y, (double)t.center_y);
    }
    
    // Pick split axis and value
    bool split_x = dist01(rng) < 0.5;
    double split_val = 0;
    if (split_x) {
        split_val = min_x + (max_x - min_x) * dist01(rng);
    } else {
        split_val = min_y + (max_y - min_y) * dist01(rng);
    }
    
    std::vector<ChristmasTree> child;
    child.reserve(p1.size());
    
    // Take from P1 if < split, P2 if >= split
    for(const auto& t : p1) {
        double val = split_x ? t.center_x : t.center_y;
        if (val < split_val) {
            child.push_back(t);
        }
    }
    
    for(const auto& t : p2) {
        double val = split_x ? t.center_x : t.center_y;
        if (val >= split_val) {
            child.push_back(t);
        }
    }
    
    // Fix count
    int target_n = (int)p1.size();
    if (child.size() > target_n) {
        // Too many trees: Remove random ones (or could remove those closest to cut)
        // Shuffling is fair
        std::shuffle(child.begin(), child.end(), rng);
        child.resize(target_n);
    } else {
        while(child.size() < target_n) {
            // Missing trees: Pick random from P1/P2 to fill
            const auto& source = (dist01(rng) < 0.5) ? p1 : p2;
            int idx = std::uniform_int_distribution<int>(0, source.size()-1)(rng);
            child.push_back(source[idx]);
        }
    }
    
    return child;
}

inline std::tuple<long double, std::vector<ChristmasTree>> ga_optimize(
    const std::vector<ChristmasTree>& initial_trees,
    const SAParams& params
) {
    int population_size = 12; // Increased slightly
    int generations = 20;     // Increased generations
    
    std::mt19937 rng(params.seed);
    std::uniform_real_distribution<double> dist01(0.0, 1.0);
    
    // Initialize Population
    std::vector<std::vector<ChristmasTree>> population;
    population.reserve(population_size);
    population.push_back(initial_trees);
    
    // Add 1 very compressed version
    std::vector<ChristmasTree> compressed = compact_trees(initial_trees, 500, 0.05);
    population.push_back(compressed);

    // Add 1 slightly perturbed version
    SAParams p_perturb = params;
    p_perturb.nsteps = 1000;
    p_perturb.Tmax = 1.0; 
    auto res_p = sa_optimize_individual(initial_trees, p_perturb);
    population.push_back(std::get<1>(res_p));

    // Fill rest with more aggressive mutations of initial
    while(population.size() < population_size) {
        SAParams init_params = params;
        init_params.nsteps = 2000; 
        init_params.Tmax = 2.0; 
        init_params.seed = params.seed + (int)population.size() * 100;
        auto res = sa_optimize_individual(initial_trees, init_params);
        population.push_back(std::get<1>(res));
    }
    
    std::vector<ChristmasTree> global_best_trees = initial_trees;
    long double global_best_score = overlap::calculate_score(global_best_trees);
    
    for (int gen = 0; gen < generations; ++gen) {
        // 1. Evaluate
        std::vector<std::pair<long double, int>> scores;
        for (int i = 0; i < population_size; ++i) {
            long double s = overlap::calculate_score(population[i]);
            scores.push_back({s, i});
            if (s < global_best_score) {
                global_best_score = s;
                global_best_trees = population[i];
            }
        }
        std::sort(scores.begin(), scores.end()); // Ascending (minimize)
        
        // Elitism: Keep best 2
        std::vector<std::vector<ChristmasTree>> next_pop;
        next_pop.push_back(population[scores[0].second]);
        next_pop.push_back(population[scores[1].second]);
        
        // 2. Crossover & Mutation
        while (next_pop.size() < population_size) {
            // Tournament Selection
            int p1_idx = scores[std::uniform_int_distribution<int>(0, population_size/2)(rng)].second;
            int p2_idx = scores[std::uniform_int_distribution<int>(0, population_size/2)(rng)].second;
            
            // Spatial Crossover
            std::vector<ChristmasTree> child = crossover_spatial(population[p1_idx], population[p2_idx], rng);
            
            // 3. Soft Constraint Repair (Squeeze/Resolve Overlaps)
            // Use squeeze_optimization with 0 shrink to just resolve overlaps, or small shrink to improve
            // We use a small shrink to encourage compaction
            child = squeeze_optimization(child, 0.005, 2000); 
            
            // 4. Hard Constraint Polish (SA)
            SAParams mut_params = params;
            mut_params.nsteps = params.nsteps / generations; 
            mut_params.seed = params.seed + gen * 100 + (int)next_pop.size();

            // sa_optimize_individual will further refine the solution.
            // Even if squeeze_optimization left some overlaps, SA can still operate (it fixes local overlaps when moving trees).
            auto res = sa_optimize_individual(child, mut_params);
            next_pop.push_back(std::get<1>(res));
        }
        population = next_pop;
    }
    
    return {global_best_score, global_best_trees};
}

inline std::vector<ChristmasTree> optimize_cluster(int n_trees, int seed) {
    std::mt19937 rng(seed);
    std::vector<ChristmasTree> cluster;
    cluster.reserve(n_trees);
    
    // Heuristic initialization: Spiral
    double spacing = 0.5;
    for(int i=0; i<n_trees; ++i) {
        double angle = i * 2.4; // Golden angle-ish
        double r = spacing * std::sqrt(i);
        double x = r * std::cos(angle);
        double y = r * std::sin(angle);
        cluster.emplace_back(x, y, (double)(i * 30));
    }
    
    // Optimize cluster
    SAParams params;
    params.nsteps = 10000;
    params.Tmax = 2.0;
    params.Tmin = 0.001;
    params.seed = seed;
    params.position_delta = 0.1;
    params.angle_delta = 5.0;
    params.nsteps_per_T = 1;
    
    // Squeeze first to compact
    cluster = squeeze_optimization(cluster, 0.01, 5000);
    
    // Then polish
    auto res = sa_optimize_individual(cluster, params);
    cluster = std::get<1>(res);
    
    return cluster;
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
            switch (key) {
                case 0: cand.row_phase_x = best.row_phase_x + delta; break;
                case 1: cand.col_phase_y = best.col_phase_y + delta; break;
                case 2: cand.shear_x = best.shear_x + delta; break;
                case 3: cand.shear_y = best.shear_y + delta; break;
                case 4: cand.parity_row_deg = std::fmod(best.parity_row_deg + delta, 360.0); break;
                case 5: cand.parity_col_deg = std::fmod(best.parity_col_deg + delta, 360.0); break;
                case 6: cand.a = best.a + best.a * delta; break;
                case 7: cand.b = best.b + best.b * delta; break;
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

inline GridState create_tiled_state(const std::vector<ChristmasTree>& cluster, double a, double b) {
    GridState s;
    s.a = a;
    s.b = b;
    s.row_phase_x = 0;
    s.col_phase_y = 0;
    s.shear_x = 0;
    s.shear_y = 0;
    s.parity_row_deg = 0;
    s.parity_col_deg = 0;
    
    for(const auto& t : cluster) {
        s.seed_xs.push_back(t.center_x);
        s.seed_ys.push_back(t.center_y);
        s.seed_degs.push_back(t.angle_deg);
    }
    return s;
}

} // namespace optimization
