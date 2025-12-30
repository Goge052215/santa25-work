#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <sys/stat.h>
#include <sys/types.h>
#include <omp.h>
#include <map>
#include "tree.hpp"
#include "submission.hpp"
#include "grid.hpp"
#include "overlap.hpp"
#include "optimization.hpp"
#include "gpu/gpu_context.hpp"

using namespace optimization;

struct GridConfig {
    int ncols;
    int nrows;
    bool append_x;
    bool append_y;
};

// Helper to get side length of a set of trees
long double get_side_length(const std::vector<ChristmasTree>& trees) {
    if (trees.empty()) return 0.0L;
    auto box_first = trees[0].aabb();
    long double min_x = box_first.first.x;
    long double min_y = box_first.first.y;
    long double max_x = box_first.second.x;
    long double max_y = box_first.second.y;

    for (size_t i = 1; i < trees.size(); ++i) {
        auto box = trees[i].aabb();
        if (box.first.x < min_x) min_x = box.first.x;
        if (box.first.y < min_y) min_y = box.first.y;
        if (box.second.x > max_x) max_x = box.second.x;
        if (box.second.y > max_y) max_y = box.second.y;
    }

    long double sf = ChristmasTree::scale_factor;
    return std::max((max_x - min_x) / sf, (max_y - min_y) / sf);
}

void deletion_cascade(
    std::vector<std::pair<long double, std::vector<ChristmasTree>>>& solutions
) {
    // solutions is 0-indexed, so solutions[n-1] corresponds to group size n.
    // We go from n=200 down to 2.
    // indices: 199 down to 1.
    
    // Pre-calculate side lengths
    std::vector<long double> side_lengths(201);
    for (int n = 1; n <= 200; ++n) {
        side_lengths[n] = get_side_length(solutions[n - 1].second);
    }

    for (int n = 200; n > 1; --n) {
        const auto& current_sol = solutions[n - 1].second; // size n
        long double best_prev_side = side_lengths[n - 1];
        int best_delete_idx = -1;

        // Try deleting each tree from current_sol to form candidate for n-1
        for (int i = 0; i < n; ++i) {
            std::vector<ChristmasTree> candidate;
            candidate.reserve(n - 1);
            for (int j = 0; j < n; ++j) {
                if (i == j) continue;
                candidate.push_back(current_sol[j]);
            }
            
            long double cand_side = get_side_length(candidate);
            if (cand_side < best_prev_side) {
                best_prev_side = cand_side;
                best_delete_idx = i;
            }
        }

        if (best_delete_idx != -1) {
            // Found a better subset from N that beats current solution for N-1
            std::vector<ChristmasTree> new_sol;
            new_sol.reserve(n - 1);
            for (int j = 0; j < n; ++j) {
                if (j == best_delete_idx) continue;
                new_sol.push_back(current_sol[j]);
            }
            solutions[n - 2].second = new_sol; // update solution for n-1
            side_lengths[n - 1] = best_prev_side;
        }
    }
}

int main() {
    // Initialize GPU context (thread-safe)
    if (GpuContext::getInstance().is_valid()) {
        std::cout << "GPU Acceleration Enabled." << std::endl;
    } else {
        std::cout << "GPU Acceleration Disabled (Metal not available or library not found)." << std::endl;
    }

    // Initial seeds from Python code
    std::vector<long double> seed_xs = {-4.191683864412409, -4.92202045352307};
    std::vector<long double> seed_ys = {-4.498489528496051, -4.727639556649786};
    std::vector<long double> seed_degs = {74.54421568660419, 254.5401905706735};
    
    long double a_init = 0.8744896974945239;
    long double b_init = 0.7499641699190263;

    SAParams params;
    params.Tmax = 0.11179901333601554;
    params.Tmin = 0.047444327977129414;
    params.nsteps = 1200;
    params.nsteps_per_T = 12; // Reduced from 24
    params.position_delta = 0.061618140065500766;
    params.angle_delta = 2.930163420191516;
    params.angle_delta2 = 20.526990795364707;
    params.delta_t = 0.08859034625418066;
    params.stagger_delta = 0.02;
    params.shear_delta = 0.02;
    params.parity_delta = 0.5;
    params.seed = 42;

    // Generate grid configs
    std::vector<GridConfig> grid_configs;
    grid_configs.push_back({3, 5, false, false});
    grid_configs.push_back({4, 5, false, false});
    grid_configs.push_back({4, 6, false, false});
    grid_configs.push_back({4, 7, false, false});
    grid_configs.push_back({5, 7, false, true});
    grid_configs.push_back({5, 8, false, false});
    grid_configs.push_back({6, 7, false, false});
    grid_configs.push_back({7, 11, false, true});
    grid_configs.push_back({8, 12, false, true});

    for (int ncols = 2; ncols <= 10; ++ncols) {
        for (int nrows = ncols; nrows <= 14; ++nrows) {
            int n_trees = 2 * ncols * nrows;
            if (n_trees >= 20 && n_trees <= 200) {
                // Check if exists
                bool exists = false;
                for (const auto& c : grid_configs) {
                    if (c.ncols == ncols && c.nrows == nrows && !c.append_x && !c.append_y) {
                        exists = true; break;
                    }
                }
                if (!exists) grid_configs.push_back({ncols, nrows, false, false});

                int n_wy = n_trees + ncols;
                if (n_wy <= 200) {
                    bool exists_y = false;
                    for (const auto& c : grid_configs) {
                        if (c.ncols == ncols && c.nrows == nrows && !c.append_x && c.append_y) {
                            exists_y = true; break;
                        }
                    }
                    if (!exists_y) grid_configs.push_back({ncols, nrows, false, true});
                }

                int n_wx = n_trees + nrows;
                if (n_wx <= 200) {
                     bool exists_x = false;
                    for (const auto& c : grid_configs) {
                        if (c.ncols == ncols && c.nrows == nrows && c.append_x && !c.append_y) {
                            exists_x = true; break;
                        }
                    }
                    if (!exists_x) grid_configs.push_back({ncols, nrows, true, false});
                }
            }
        }
    }

    // Sort grid configs
    std::sort(grid_configs.begin(), grid_configs.end(), [](const GridConfig& a, const GridConfig& b) {
        int na = 2 * a.ncols * a.nrows + (a.append_x ? a.nrows : 0) + (a.append_y ? a.ncols : 0);
        int nb = 2 * b.ncols * b.nrows + (b.append_x ? b.nrows : 0) + (b.append_y ? b.ncols : 0);
        return na < nb;
    });

    // Remove duplicates? Logic above checks existence, but simplistic.
    // The sorting key was primary.
    
    // Prepare tasks
    struct Task {
        GridConfig config;
        int seed;
    };
    std::vector<Task> tasks;
    int num_starts = 4;
    for (size_t i = 0; i < grid_configs.size(); ++i) {
        int n_base = 2 * grid_configs[i].ncols * grid_configs[i].nrows;
        int n_add = (grid_configs[i].append_x ? grid_configs[i].nrows : 0) + 
                    (grid_configs[i].append_y ? grid_configs[i].ncols : 0);
        if (n_base + n_add > 200) continue;

        for (int k = 0; k < num_starts; ++k) {
            tasks.push_back({grid_configs[i], 42 + (int)i * 1000 + k});
        }
    }

    // Results map: n_trees -> (score, trees)
    // We need thread-safe storage or merge after.
    // Since n_trees is key, we can use a vector of bests.
    std::vector<std::pair<long double, std::vector<ChristmasTree>>> best_results(201);
    for (int i = 0; i <= 200; ++i) best_results[i].first = 1e18L;

    std::cout << "Running " << tasks.size() << " optimization tasks..." << std::endl;

    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < tasks.size(); ++i) {
        GridState initial_state;
        {
            std::mt19937 rng(tasks[i].seed * 101 + 7);
            std::uniform_real_distribution<long double> dpos(-0.3L, 0.3L);
            std::uniform_real_distribution<long double> ddeg(-15.0L, 15.0L);
            initial_state.seed_xs = seed_xs;
            initial_state.seed_ys = seed_ys;
            initial_state.seed_degs = seed_degs;
            for (size_t si = 0; si < initial_state.seed_xs.size(); ++si) {
                initial_state.seed_xs[si] += dpos(rng);
                initial_state.seed_ys[si] += dpos(rng);
                initial_state.seed_degs[si] = std::fmod(initial_state.seed_degs[si] + ddeg(rng), 360.0);
            }
        }
        initial_state.a = a_init;
        initial_state.b = b_init;
        initial_state.row_phase_x = 0;
        initial_state.col_phase_y = 0;
        initial_state.shear_x = 0;
        initial_state.shear_y = 0;
        initial_state.parity_row_deg = 0;
        initial_state.parity_col_deg = 0;

        SAParams local_params = params;
        local_params.seed = tasks[i].seed;

        auto result = sa_optimize(
            initial_state, 
            tasks[i].config.ncols, 
            tasks[i].config.nrows, 
            tasks[i].config.append_x, 
            tasks[i].config.append_y, 
            local_params
        );

        auto refined = refine_grid(
            std::get<1>(result),
            tasks[i].config.ncols,
            tasks[i].config.nrows,
            tasks[i].config.append_x,
            tasks[i].config.append_y,
            local_params
        );

        long double score = std::get<0>(refined);
        const auto& trees = std::get<2>(refined);
        int n_trees = (int)trees.size();

        if (n_trees <= 200) {
            #pragma omp critical
            {
                if (score < best_results[n_trees].first) {
                    best_results[n_trees] = {score, trees};
                    // std::cout << "New best for " << n_trees << ": " << score << std::endl;
                }
            }
        }
    }

    // Check if we have a result for 200
    if (best_results[200].second.empty()) {
        std::cout << "Running fallback for N=200" << std::endl;
        GridState initial_state;
        initial_state.seed_xs = seed_xs;
        initial_state.seed_ys = seed_ys;
        initial_state.seed_degs = seed_degs;
        initial_state.a = a_init;
        initial_state.b = b_init;
        initial_state.row_phase_x = 0; initial_state.col_phase_y = 0;
        initial_state.shear_x = 0; initial_state.shear_y = 0;
        initial_state.parity_row_deg = 0; initial_state.parity_col_deg = 0;

        auto result = sa_optimize(initial_state, 8, 12, false, true, params);
        best_results[200] = {std::get<0>(result), std::get<2>(result)};
    }

    // Assemble final solutions
    std::vector<std::pair<long double, std::vector<ChristmasTree>>> solutions(200);
    const auto& sol200 = best_results[200].second;

    for (int n = 1; n <= 200; ++n) {
        if (!best_results[n].second.empty()) {
            solutions[n - 1] = {best_results[n].first, best_results[n].second};
        } else {
            // Take first n from sol200
            std::vector<ChristmasTree> sub(sol200.begin(), sol200.begin() + n);
            solutions[n - 1] = {0.0L, sub}; // Score recalculated later
        }
    }

    // Deletion cascade
    std::cout << "Running deletion cascade..." << std::endl;
    deletion_cascade(solutions);

    // Final scoring
    long double overall_score = 0.0L;
    std::vector<std::pair<long double, std::vector<ChristmasTree>>> final_output;
    final_output.reserve(200);

    for (int n = 1; n <= 200; ++n) {
        long double s = get_side_length(solutions[n - 1].second);
        overall_score += (s * s) / n;
        final_output.push_back({s, solutions[n - 1].second});
    }

    std::cout << "Overall Score: " << std::setprecision(12) << overall_score << std::endl;

    struct stat st;
    if (stat("data", &st) != 0) {
        mkdir("data", 0755);
    }
    submission::write_csv(final_output, "data/submission.csv");
    std::cout << "Saved to data/submission.csv" << std::endl;

    return 0;
}
