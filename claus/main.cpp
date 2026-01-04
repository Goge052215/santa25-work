#include <iostream>
#include <fstream>
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
#include "beam_search.hpp"
#include "symmetry.hpp"
// #include "placement_model.hpp"

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

ChristmasTree find_valid_placement(const std::vector<ChristmasTree>& existing, int seed) {
    std::mt19937 rng(seed);
    
    // 1. Try ML-guided placement first
    /*
    for(int i=0; i<5000; ++i) {
        ChristmasTree cand = ml_policy::propose_placement(existing, seed + i);
        
        // Check bounds
        if (cand.center_x < -100.0L || cand.center_x > 100.0L ||
            cand.center_y < -100.0L || cand.center_y > 100.0L) continue;

        // Check overlap with existing
        if (!overlap::has_overlap_with_others(existing, cand)) {
            return cand;
        }
    }
    */

    // 2. Fallback to random if ML fails
    // Determine bounds
    long double min_x = 100.0L, max_x = -100.0L;
    long double min_y = 100.0L, max_y = -100.0L;
    
    if (existing.empty()) {
        min_x = -1.0L; max_x = 1.0L;
        min_y = -1.0L; max_y = 1.0L;
    } else {
        auto box = existing[0].aabb();
        min_x = box.first.x; max_x = box.second.x;
        min_y = box.first.y; max_y = box.second.y;
        for (const auto& t : existing) {
            auto b = t.aabb();
            min_x = std::min(min_x, b.first.x);
            max_x = std::max(max_x, b.second.x);
            min_y = std::min(min_y, b.first.y);
            max_y = std::max(max_y, b.second.y);
        }
    }

    // Expand search area slightly
    long double pad = 2.0L; 
    std::uniform_real_distribution<long double> dx(min_x - pad, max_x + pad);
    std::uniform_real_distribution<long double> dy(min_y - pad, max_y + pad);
    std::uniform_real_distribution<long double> ddeg(0.0L, 360.0L);

    // Try random placements
    for (int i = 0; i < 5000; ++i) {
        ChristmasTree cand(dx(rng), dy(rng), ddeg(rng));
        
        // Check bounds
        if (cand.center_x < -100.0L || cand.center_x > 100.0L ||
            cand.center_y < -100.0L || cand.center_y > 100.0L) continue;

        // Check overlap with existing
        if (!overlap::has_overlap_with_others(existing, cand)) {
            return cand;
        }
    }
    
    // Fallback: Just place it somewhere and hope optimizer fixes it? 
    // Or return a tree far away?
    return ChristmasTree(max_x + 2.0L, max_y + 2.0L, 0.0L);
}

void greedy_insertion(
    std::vector<std::pair<long double, std::vector<ChristmasTree>>>& solutions,
    const SAParams& params
) {
    #pragma omp parallel for schedule(dynamic)
    for (int n = 1; n < 200; ++n) {
        if (solutions[n-1].second.empty()) continue;
        
        const auto& current_sol = solutions[n-1].second;
        
        // Create candidate for n+1
        std::vector<ChristmasTree> candidate = current_sol;
        ChristmasTree new_tree = find_valid_placement(candidate, n * 12345);
        candidate.push_back(new_tree);
        
        // Optimize
        SAParams local_params = params;
        local_params.seed = n * 54321;
        local_params.nsteps = 5000; // Fast optimization
        local_params.Tmax = 0.2;
        
        auto res = ga_optimize(candidate, local_params);
        auto optimized = std::get<1>(res);
        optimized = compact_trees(optimized, 500, 0.01);
        optimized = coordinate_descent_polish(optimized);
        
        long double score = get_side_length(optimized);
        long double existing_score = 1e18L;
        
        #pragma omp critical
        {
            if (!solutions[n].second.empty()) {
                existing_score = get_side_length(solutions[n].second);
            }
            
            if (score < existing_score) {
                solutions[n] = {score, optimized};
                std::cout << "Greedy Insert Improved N=" << (n+1) << ": " << existing_score << " -> " << score << std::endl;
            }
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
    // (Disabled in favor of Beam Search for initialization, but kept for fallback or specific cases if needed)
    /*
    std::vector<GridConfig> grid_configs;
    grid_configs.push_back({3, 5, false, false});
    // ... (rest of grid configs)
    */

    // Prepare tasks
    // Replaced by Beam Search initialization
    
    // Results map: n_trees -> (score, trees)
    // We need thread-safe storage or merge after.
    // Since n_trees is key, we can use a vector of bests.
    std::vector<std::pair<long double, std::vector<ChristmasTree>>> best_results(201);
    for (int i = 0; i <= 200; ++i) best_results[i].first = 1e18L;

    // Load existing solutions and refine
    mkdir("data/solutions", 0755);
    std::string submission_path = "data/submission.csv";
    struct stat buffer;
    if (stat(submission_path.c_str(), &buffer) == 0) {
        std::cout << "Loading existing submission from " << submission_path << "..." << std::endl;
        auto loaded = submission::read_csv(submission_path);
        
        int loaded_count = 0;
        for(const auto& l : loaded) if(!l.second.empty()) loaded_count++;
        std::cout << "Loaded solutions for " << loaded_count << " N values." << std::endl;
        
        // Fill best_results with loaded
        for(int n=1; n<=200; ++n) {
             if(!loaded[n-1].second.empty()) {
                 long double score = get_side_length(loaded[n-1].second);
                 best_results[n] = {score, loaded[n-1].second};
             }
        }
    }

    // Try Symmetric Solutions for specific N
    std::vector<int> sym_targets = {14, 18, 20, 22};
    std::cout << "Generating symmetric candidates for N={14, 18, 20, 22}..." << std::endl;
    
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < sym_targets.size(); ++i) {
        int n = sym_targets[i];
        // Try multiple seeds
        for(int k=0; k<10; ++k) {
            auto sym_trees = symmetry::optimize_symmetry(n, 12345 + n * 100 + k);
            
            // Refine further using general optimizer to break symmetry if beneficial
            // Or just polish
            
            // Fix overlaps if any (symmetry search penalizes but might leave slight overlap)
            if (overlap::has_any_overlap(sym_trees)) {
                sym_trees = squeeze_optimization(sym_trees, 0.01, 10000);
            }
            
            // Polish
            sym_trees = coordinate_descent_polish(sym_trees);
            sym_trees = physics_polish(sym_trees, 1000, 0.05); // Use physics polish
            sym_trees = coordinate_descent_polish(sym_trees);

            long double score = get_side_length(sym_trees);
            
            #pragma omp critical
            {
                if (score < best_results[n].first) {
                    best_results[n] = {score, sym_trees};
                    std::cout << "Symmetric Improved N=" << n << ": " << score << std::endl;
                }
            }
        }
    }

    /*
    std::cout << "Refining existing solutions..." << std::endl;
    #pragma omp parallel for schedule(dynamic)
    for (int n = 1; n <= 200; ++n) {
        // ... (rest of refinement logic)
    }
    */
    
    // Refine existing solutions logic (simplified for integration)
    std::cout << "Refining existing solutions..." << std::endl;
    #pragma omp parallel for schedule(dynamic)
    for (int n = 1; n <= 200; ++n) {
        if (!best_results[n].second.empty()) {
             auto trees = best_results[n].second;
             // ... Refinement Logic from original code ...
             // (Copying relevant parts back or assuming they are effectively unchanged but using best_results source)
             
            // Fix initial overlaps if any
            if (overlap::has_any_overlap(trees)) {
                trees = squeeze_optimization(trees, -0.01, 50000);
                if (overlap::has_any_overlap(trees)) {
                    trees = squeeze_optimization(trees, -0.05, 100000);
                }
            }

            long double score = get_side_length(trees);
            
            // Refine
            SAParams local_params = params;
            local_params.seed = n * 999;
            local_params.Tmax = 2.0; 
            local_params.Tmin = 1e-6;
            local_params.nsteps = 200000; 
            local_params.nsteps_per_T = 1; 
            
            auto refined_res = ga_optimize(trees, local_params);
            auto refined_trees = std::get<1>(refined_res);
            
            // Final hard compaction
            refined_trees = compact_trees(refined_trees, 5000, 0.005);
            
            // Physics-based Polish (New SOTA approach)
            refined_trees = physics_polish(refined_trees, 2000, 0.05);

            // Coordinate Descent Polish
            refined_trees = coordinate_descent_polish(refined_trees);
            
            // Iterative Squeeze
            bool squeeze_success = true;
            int squeeze_iter = 0;
            while (squeeze_success && squeeze_iter < 30) { 
                double factor = 0.005; 
                if (squeeze_iter > 5) factor = 0.002; 
                if (squeeze_iter > 15) factor = 0.001; 

                auto squeezed = squeeze_optimization(refined_trees, factor, 25000);
                
                if (!overlap::has_any_overlap(squeezed)) {
                    refined_trees = squeezed;
                    refined_trees = coordinate_descent_polish(refined_trees);
                    
                    long double refined_score = get_side_length(refined_trees);
                    if (refined_score < score) {
                        trees = refined_trees;
                        score = refined_score;
                        #pragma omp critical
                        {
                            std::cout << "Squeeze Improved N=" << n << ": " << score << std::endl;
                            // Save intermediate
                            std::string fname = "data/solutions/" + std::to_string(n) + ".csv";
                            std::ofstream out(fname);
                            out << "id,x,y,deg\n";
                            for(size_t i=0; i<trees.size(); ++i) {
                                out << n << "_" << (i+1) << ",s" << trees[i].center_x << ",s" << trees[i].center_y << ",s" << trees[i].angle_deg << "\n";
                            }
                            out.close();
                        }
                    }
                    squeeze_iter++;
                } else {
                    squeeze_success = false;
                }
            }
            
            #pragma omp critical
            {
                if (score < best_results[n].first) {
                    best_results[n] = {score, trees};
                }
                std::cout << "N=" << n << " done. Score: " << score << std::endl;
            }
        }
    }

    // Assemble final solutions
    std::vector<std::pair<long double, std::vector<ChristmasTree>>> solutions(200);

    // Final scoring
    long double overall_score = 0.0L;
    std::vector<std::pair<long double, std::vector<ChristmasTree>>> final_output;
    final_output.reserve(200);

    for (int n = 1; n <= 200; ++n) {
        if (!best_results[n].second.empty()) {
            solutions[n - 1] = {best_results[n].first, best_results[n].second};
        }
    }
    
    // Deletion cascade
    std::cout << "Running deletion cascade (Pass 1)..." << std::endl;
    deletion_cascade(solutions);

    // Greedy Insertion (Bottom-Up)
    // std::cout << "Running greedy insertion (Bottom-Up)..." << std::endl;
    // Increase params for greedy insertion
    // SAParams greedy_params = params;
    // greedy_params.nsteps = 10000;
    // greedy_params.Tmax = 0.5;
    // greedy_insertion(solutions, greedy_params);

    // Deletion cascade (Pass 2)
    std::cout << "Running deletion cascade (Pass 2)..." << std::endl;
    deletion_cascade(solutions);

    for (int n = 1; n <= 200; ++n) {
        if (solutions[n-1].second.empty()) {
             // Shouldn't happen if we loaded 200 solutions
        }
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
