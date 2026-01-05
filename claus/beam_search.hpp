#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <omp.h>
#include "tree.hpp"
#include "overlap.hpp"

// Represents a relative geometric relationship between two trees (a "lock")
struct PlacementPattern {
    double dx;      // Relative x in anchor's coordinate frame
    double dy;      // Relative y in anchor's coordinate frame
    double d_deg;   // Relative angle change
    double weight;  // Quality score of this pattern (optional)
};

// Represents a partial solution in the beam
struct BeamState {
    std::vector<ChristmasTree> trees;
    double side_length; // The metric we want to minimize (max(w, h))
    
    // Helper to update score after adding a tree
    void update_score() {
        if (trees.empty()) { side_length = 0; return; }
        
        // We use the raw float coordinates for scoring to be faster, 
        // relying on ChristmasTree::aabb() which uses the polygon points.
        long double min_x = 1e18, min_y = 1e18, max_x = -1e18, max_y = -1e18;
        
        for (const auto& t : trees) {
            auto box = t.aabb(); // Returns scaled coordinates (1e15)
            min_x = std::min(min_x, box.first.x);
            min_y = std::min(min_y, box.first.y);
            max_x = std::max(max_x, box.second.x);
            max_y = std::max(max_y, box.second.y);
        }
        
        long double sf = ChristmasTree::scale_factor;
        double w = (double)((max_x - min_x) / sf);
        double h = (double)((max_y - min_y) / sf);
        side_length = std::max(w, h);
    }
};

class BeamSearch {
public:
    std::vector<PlacementPattern> patterns;

    // Load patterns from the placement_model CSV/file
    // Format assumed: dx, dy, d_deg, weight
    void load_patterns(const std::string& filepath) {
        std::ifstream infile(filepath);
        std::string line;
        // Skip header if exists
        // std::getline(infile, line); 

        while (std::getline(infile, line)) {
            if (line.empty()) continue;
            std::stringstream ss(line);
            std::string val;
            PlacementPattern p;
            
            // Example parsing logic - adjust to your specific file format
            std::vector<double> row;
            while (std::getline(ss, val, ',')) {
                row.push_back(std::stod(val));
            }
            if (row.size() >= 3) {
                p.dx = row[0];
                p.dy = row[1];
                p.d_deg = row[2];
                p.weight = (row.size() > 3) ? row[3] : 1.0;
                patterns.push_back(p);
            }
        }
        std::cout << "Loaded " << patterns.size() << " interaction patterns." << std::endl;
    }

    // Apply a pattern to an anchor tree to get a new candidate tree
    ChristmasTree apply_pattern(const ChristmasTree& anchor, const PlacementPattern& pat) const {
        // Convert anchor angle to radians
        double rad = anchor.angle_deg * (M_PI / 180.0);
        double c = std::cos(rad);
        double s = std::sin(rad);

        // Rotate the pattern offset into global frame
        double global_dx = pat.dx * c - pat.dy * s;
        double global_dy = pat.dx * s + pat.dy * c;

        // Create new tree
        return ChristmasTree(
            anchor.center_x + global_dx,
            anchor.center_y + global_dy,
            anchor.angle_deg + pat.d_deg
        );
    }

    std::vector<ChristmasTree> solve(int target_n, int beam_width, 
                                   std::vector<std::pair<long double, std::vector<ChristmasTree>>>* best_results = nullptr) {
        // 1. Initialize Beam with a single tree at (0,0,0)
        std::vector<BeamState> current_beam;
        BeamState initial_state;
        initial_state.trees.push_back(ChristmasTree(0, 0, 0));
        initial_state.update_score();
        current_beam.push_back(initial_state);

        // Record initial state for N=1
        if (best_results && best_results->size() > 1) {
            if (initial_state.side_length < (*best_results)[1].first) {
                (*best_results)[1] = {initial_state.side_length, initial_state.trees};
            }
        }

        bool use_gpu = GpuContext::getInstance().is_valid();
        if (use_gpu) {
            std::cout << "Beam Search: GPU Acceleration Enabled." << std::endl;
        } else {
            std::cout << "Beam Search: GPU Unavailable, using CPU." << std::endl;
        }

        std::cout << "Starting Beam Search (Target N=" << target_n 
                  << ", Beam=" << beam_width << ")..." << std::endl;

        // 2. Iterate until we reach target size
        for (int n = 1; n < target_n; ++n) {
            
            std::vector<BeamState> all_next_states;

            if (use_gpu) {
                // GPU Path: Serial over beam states, batch over patterns
                // To avoid massive memory usage, we process one state at a time
                // and prune locally if needed, though with width=20 it's fine.
                
                const int BATCH_SIZE = 500000; // 500k candidates per batch

                for (const auto& state : current_beam) {
                    // For this state, we generate candidates from all anchors and all patterns
                    // We can iterate patterns and for each pattern apply to all anchors
                    // Or iterate anchors.
                    // Total candidates = n * patterns.size()
                    // This can be huge (10 * 1.9M = 19M).
                    // We must batch.
                    
                    std::vector<ChristmasTree> candidate_batch;
                    candidate_batch.reserve(BATCH_SIZE);
                    
                    for (size_t pat_idx = 0; pat_idx < patterns.size(); ++pat_idx) {
                        const auto& pat = patterns[pat_idx];
                        
                        for (const auto& anchor : state.trees) {
                            ChristmasTree cand = apply_pattern(anchor, pat);
                            
                            // A. Bounds Check (Fast fail on CPU)
                            if (std::abs(cand.center_x) > 50.0 || std::abs(cand.center_y) > 50.0) continue;
                            
                            candidate_batch.push_back(cand);
                            
                            if (candidate_batch.size() >= BATCH_SIZE) {
                                // Flush batch
                                auto results = GpuContext::getInstance().check_candidates_overlap(state.trees, candidate_batch);
                                
                                for(size_t k=0; k<candidate_batch.size(); ++k) {
                                    if (results[k] == 0) { // Valid (0 = no overlap)
                                        BeamState new_state = state;
                                        new_state.trees.push_back(candidate_batch[k]);
                                        new_state.update_score();
                                        all_next_states.push_back(std::move(new_state));
                                    }
                                }
                                candidate_batch.clear();
                                
                                // Intermediate pruning if list gets too large
                                if (all_next_states.size() > (size_t)(beam_width * 20)) {
                                     std::partial_sort(all_next_states.begin(), 
                                                       all_next_states.begin() + beam_width * 5, 
                                                       all_next_states.end(), 
                                        [](const BeamState& a, const BeamState& b) { return a.side_length < b.side_length; });
                                     all_next_states.resize(beam_width * 5);
                                }
                            }
                        }
                    }
                    
                    // Flush remaining
                    if (!candidate_batch.empty()) {
                        auto results = GpuContext::getInstance().check_candidates_overlap(state.trees, candidate_batch);
                        for(size_t k=0; k<candidate_batch.size(); ++k) {
                            if (results[k] == 0) {
                                BeamState new_state = state;
                                new_state.trees.push_back(candidate_batch[k]);
                                new_state.update_score();
                                all_next_states.push_back(std::move(new_state));
                            }
                        }
                    }
                }
                
            } else {
                // CPU Path (OpenMP)
                // Thread-local storage for next states
                std::vector<std::vector<BeamState>> next_states_per_thread(omp_get_max_threads());
    
                #pragma omp parallel for schedule(dynamic)
                for (size_t i = 0; i < current_beam.size(); ++i) {
                    const auto& state = current_beam[i];
                    int tid = omp_get_thread_num();
                    
                    // Try to attach a new tree to EVERY existing tree in this state
                    for (const auto& anchor : state.trees) {
                        
                        // Try ALL patterns (or a random subset if 69k is too slow)
                        for (const auto& pat : patterns) {
                            
                            ChristmasTree candidate = apply_pattern(anchor, pat);
    
                            // A. Bounds Check (Fast fail)
                            if (std::abs(candidate.center_x) > 50.0 || std::abs(candidate.center_y) > 50.0) continue;
    
                            // B. Overlap Check (Expensive)
                            if (overlap::has_overlap_with_others(state.trees, candidate)) {
                                continue;
                            }
    
                            // C. Create new state
                            BeamState new_state = state;
                            new_state.trees.push_back(candidate);
                            new_state.update_score();
    
                            next_states_per_thread[tid].push_back(std::move(new_state));
                            
                            // Optimization: Prune thread-local buffer if it gets too huge 
                            if (next_states_per_thread[tid].size() > (size_t)(beam_width * 2)) {
                                 auto& vec = next_states_per_thread[tid];
                                 std::partial_sort(vec.begin(), vec.begin() + beam_width, vec.end(), 
                                    [](const BeamState& a, const BeamState& b) { return a.side_length < b.side_length; });
                                 vec.resize(beam_width);
                            }
                        }
                    }
                }
                
                // Merge candidates from all threads
                for (auto& local_vec : next_states_per_thread) {
                    all_next_states.insert(all_next_states.end(), 
                                         std::make_move_iterator(local_vec.begin()), 
                                         std::make_move_iterator(local_vec.end()));
                }
            }

            if (all_next_states.empty()) {
                std::cerr << "Beam Search died at N=" << n << " (No valid placements found)." << std::endl;
                break;
            }

            // 4. Prune / Select Top K
            size_t keep_count = std::min((size_t)beam_width, all_next_states.size());
            
            std::partial_sort(all_next_states.begin(), 
                            all_next_states.begin() + keep_count, 
                            all_next_states.end(), 
                            [](const BeamState& a, const BeamState& b) {
                                return a.side_length < b.side_length;
                            });
            
            all_next_states.resize(keep_count);
            current_beam = std::move(all_next_states);

            // Save best result for N = n + 1
            if (best_results && !current_beam.empty()) {
                int current_N = n + 1;
                if (current_N < (int)best_results->size()) {
                    if (current_beam[0].side_length < (*best_results)[current_N].first) {
                        (*best_results)[current_N] = {current_beam[0].side_length, current_beam[0].trees};
                    }
                }
            }

            std::cout << "  N=" << (n + 1) 
                      << " | Best Score: " << current_beam[0].side_length 
                      << " | Beam Size: " << current_beam.size() << std::endl;
        }

        return current_beam.empty() ? std::vector<ChristmasTree>{} : current_beam[0].trees;
    }
};