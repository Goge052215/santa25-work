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
    // Side^2 is around 60-100. Moment is around 10000.
    // We want secondary to be a tie-breaker, not driver.
    // Let's say we want 1.0 improvement in main score to be worth 1000 improvement in secondary.
    double lambda = 1e-4; 
    
    long double current_total_score = current_main_score + lambda * current_secondary_score;
    long double best_total_score = current_total_score;
    long double best_main_score = current_main_score;

    double T = params.Tmax;
    double Tfactor = -std::log(params.Tmax / params.Tmin);
    int total_steps = static_cast<int>(params.nsteps);
    int steps_per_T = static_cast<int>(params.nsteps_per_T);
    int n_trees = (int)current_trees.size();

    for (int step = 0; step < total_steps; ++step) {
        for (int k = 0; k < steps_per_T; ++k) {
            // Pick a move type: 0 = Perturb, 1 = Compress
            int move_type = (dist01(rng) < 0.05) ? 1 : 0; // 5% chance to compress
            
            if (move_type == 1) {
                 // Compression Move: Scale everything down slightly
                 double scale = 1.0 - (params.position_delta * 0.1); // Small shrinkage
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
                 }
            } else {
                // Perturbation Move
                int idx = std::uniform_int_distribution<int>(0, n_trees - 1)(rng);
                ChristmasTree old_tree = current_trees[idx];

                double dx = (dist01(rng) * 2.0 - 1.0) * params.position_delta;
                double dy = (dist01(rng) * 2.0 - 1.0) * params.position_delta;
                double ddeg = (dist01(rng) * 2.0 - 1.0) * params.angle_delta;
                
                current_trees[idx].center_x += dx;
                current_trees[idx].center_y += dy;
                current_trees[idx].angle_deg = std::fmod(current_trees[idx].angle_deg + ddeg, 360.0);
                
                current_trees[idx] = ChristmasTree(current_trees[idx].center_x, current_trees[idx].center_y, current_trees[idx].angle_deg);
                
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
                trees[i] = ChristmasTree(trees[i].center_x, trees[i].center_y, trees[i].angle_deg);
                
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
    
    // Initial cost: Overlap Area
    // Monte Carlo is too slow for 5000 steps. 
    // Use Proxy: Number of overlapping pairs? Or Penetration Depth approximation?
    // Let's use Number of Overlapping Pairs + Distance to Center
    
    // Actually, simple SA with "Count Overlaps" objective is often enough to find a valid config if density permits.
    // Cost = OverlapPairs * 1000 + Repulsion (to guide separation)
    
    // Helper for fast triangle overlap (SAT)
    auto is_triangle_overlap = [](const ChristmasTree& t1, const ChristmasTree& t2) {
        auto get_tri = [](const ChristmasTree& t) {
            double rad = t.angle_deg * 3.14159265358979323846 / 180.0;
            double c = std::cos(rad);
            double s = std::sin(rad);
            // Triangle approximation of the tree
            double pts[3][2] = {{-0.35, 0.0}, {0.35, 0.0}, {0.0, 0.8}};
            std::vector<std::pair<double, double>> out(3);
            for(int k=0; k<3; ++k) {
                out[k].first = t.center_x + pts[k][0]*c - pts[k][1]*s;
                out[k].second = t.center_y + pts[k][0]*s + pts[k][1]*c;
            }
            return out;
        };
        auto tri1 = get_tri(t1);
        auto tri2 = get_tri(t2);
        
        std::vector<std::pair<double,double>> axes;
        for(int i=0; i<3; ++i) {
            double dx = tri1[(i+1)%3].first - tri1[i].first;
            double dy = tri1[(i+1)%3].second - tri1[i].second;
            axes.push_back({-dy, dx});
            dx = tri2[(i+1)%3].first - tri2[i].first;
            dy = tri2[(i+1)%3].second - tri2[i].second;
            axes.push_back({-dy, dx});
        }
        
        for(const auto& axis : axes) {
            double min1 = 1e18, max1 = -1e18;
            double min2 = 1e18, max2 = -1e18;
            for(const auto& p : tri1) {
                double val = p.first * axis.first + p.second * axis.second;
                if(val < min1) min1 = val;
                if(val > max1) max1 = val;
            }
            for(const auto& p : tri2) {
                double val = p.first * axis.first + p.second * axis.second;
                if(val < min2) min2 = val;
                if(val > max2) max2 = val;
            }
            if(max1 < min2 || max2 < min1) return false;
        }
        return true;
    };

    auto calc_cost = [&](const std::vector<ChristmasTree>& t) {
        double overlaps = 0.0;
        double repulsion = 0.0;
        
        // Only check triangle overlap for speed (approximation)
        // Check only i < j to avoid double counting and speed up
        int n = (int)t.size();
        for(int i=0; i<n; ++i) {
            for(int j=i+1; j<n; ++j) {
                // AABB Check (Manual for speed)
                // Tree is roughly unit size, so dist > 2 is safe
                double dx = t[i].center_x - t[j].center_x;
                double dy = t[i].center_y - t[j].center_y;
                double d2 = dx*dx + dy*dy;
                
                if (d2 < 4.0) {
                     // Repulsion
                     double dist = std::sqrt(d2);
                     if (dist < 1.0) repulsion += (1.0 - dist);
                     
                     // Strict(ish) Overlap
                     if (is_triangle_overlap(t[i], t[j])) {
                         overlaps += 1.0;
                     }
                }
            }
        }
        return overlaps * 1000.0 + repulsion;
    };
    
    double current_cost = calc_cost(trees);
    double best_cost = current_cost;
    std::vector<ChristmasTree> best_trees = trees;
    
    double T = params.Tmax;
    double Tfactor = -std::log(params.Tmax / params.Tmin);
    
    for(int s=0; s<steps; ++s) {
        if (current_cost == 0) break; // Solved!
        
        int idx = std::uniform_int_distribution<int>(0, trees.size()-1)(rng);
        ChristmasTree old_tree = trees[idx];
        
        // Perturb
        double dx = (dist01(rng)*2.0-1.0) * params.position_delta;
        double dy = (dist01(rng)*2.0-1.0) * params.position_delta;
        double ddeg = (dist01(rng)*2.0-1.0) * params.angle_delta;
        
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
        
        double new_cost = calc_cost(trees);
        double delta = new_cost - current_cost;
        
        if (delta < 0 || dist01(rng) < std::exp(-delta/T)) {
            current_cost = new_cost;
            if (current_cost < best_cost) {
                best_cost = current_cost;
                best_trees = trees;
            }
        } else {
            trees[idx] = old_tree;
        }
        
        T = params.Tmax * std::exp(Tfactor * (s+1) / params.nsteps);
    }
    
    // Return best found. If best_cost == 0, we successfully squeezed.
    // If best_cost > 0, we failed to resolve, caller should probably discard or try less shrink.
    return best_trees;
}

inline std::tuple<long double, std::vector<ChristmasTree>> ga_optimize(
    const std::vector<ChristmasTree>& initial_trees,
    const SAParams& params
) {
    int population_size = 6; 
    int generations = 10;
    
    std::mt19937 rng(params.seed);
    std::uniform_real_distribution<double> dist01(0.0, 1.0);
    
    // Initialize Population
    // Pop[0] is the input
    // Pop[1..N] are mutated versions
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
        init_params.Tmax = 2.0; // Very hot
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
            
            const auto& p1 = population[p1_idx];
            const auto& p2 = population[p2_idx];
            std::vector<ChristmasTree> child = p1;
            
            // Uniform Crossover
            for (size_t t = 0; t < child.size(); ++t) {
                if (dist01(rng) < 0.5) {
                    // Try taking from P2
                    ChristmasTree backup = child[t];
                    child[t] = p2[t];
                    // If creates overlap with CURRENT child state, revert
                    if (overlap::has_overlap_with_others(child, t)) {
                        child[t] = backup;
                        // If backup also overlaps (because other trees changed), try to perturb?
                        if (overlap::has_overlap_with_others(child, t)) {
                            // Try a small jiggle
                             child[t].center_x += (dist01(rng)*2.0-1.0)*0.1;
                             child[t].center_y += (dist01(rng)*2.0-1.0)*0.1;
                             child[t] = ChristmasTree(child[t].center_x, child[t].center_y, child[t].angle_deg);
                             if (overlap::has_overlap_with_others(child, t)) {
                                 child[t] = backup; // Give up
                             }
                        }
                    }
                }
            }
            
            // Mutation (SA)
            SAParams mut_params = params;
            mut_params.nsteps = params.nsteps / generations; // Distribute budget
            mut_params.seed = params.seed + gen * 100 + (int)next_pop.size();
            auto res = sa_optimize_individual(child, mut_params);
            next_pop.push_back(std::get<1>(res));
        }
        population = next_pop;
    }
    
    return {global_best_score, global_best_trees};
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
