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
struct Pattern {
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
    std::vector<Pattern> patterns;

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
            Pattern p;
            
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
    ChristmasTree apply_pattern(const ChristmasTree& anchor, const Pattern& pat) const {
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

    std::vector<ChristmasTree> solve(int target_n, int beam_width) {
        // 1. Initialize Beam with a single tree at (0,0,0)
        std::vector<BeamState> current_beam;
        BeamState initial_state;
        initial_state.trees.push_back(ChristmasTree(0, 0, 0));
        initial_state.update_score();
        current_beam.push_back(initial_state);

        std::cout << "Starting Beam Search (Target N=" << target_n 
                  << ", Beam=" << beam_width << ")..." << std::endl;

        // 2. Iterate until we reach target size
        for (int n = 1; n < target_n; ++n) {
            
            // Thread-local storage for next states
            std::vector<std::vector<BeamState>> next_states_per_thread(omp_get_max_threads());

            #pragma omp parallel for schedule(dynamic)
            for (size_t i = 0; i < current_beam.size(); ++i) {
                const auto& state = current_beam[i];
                int tid = omp_get_thread_num();
                
                // Heuristic: Don't expand states that are already much worse than the best
                // (Optional implementation detail)

                // Try to attach a new tree to EVERY existing tree in this state
                for (const auto& anchor : state.trees) {
                    
                    // Try ALL patterns (or a random subset if 69k is too slow)
                    for (const auto& pat : patterns) {
                        
                        ChristmasTree candidate = apply_pattern(anchor, pat);

                        // A. Bounds Check (Fast fail)
                        // If candidate pushes side length > current_max * threshold, skip
                        if (std::abs(candidate.center_x) > 50.0 || std::abs(candidate.center_y) > 50.0) continue;

                        // B. Overlap Check (Expensive)
                        // Note: overlaps::has_overlap_with_others checks if 'candidate' overlaps any tree in 'state.trees'
                        if (overlap::has_overlap_with_others(state.trees, candidate)) {
                            continue;
                        }

                        // C. Create new state
                        BeamState new_state = state;
                        new_state.trees.push_back(candidate);
                        new_state.update_score();

                        next_states_per_thread[tid].push_back(std::move(new_state));
                        
                        // Optimization: Prune thread-local buffer if it gets too huge 
                        // to save memory, keeping only top 2*beam_width locally
                        if (next_states_per_thread[tid].size() > (size_t)(beam_width * 2)) {
                             auto& vec = next_states_per_thread[tid];
                             std::partial_sort(vec.begin(), vec.begin() + beam_width, vec.end(), 
                                [](const BeamState& a, const BeamState& b) { return a.side_length < b.side_length; });
                             vec.resize(beam_width);
                        }
                    }
                }
            }

            // 3. Merge candidates from all threads
            std::vector<BeamState> all_next_states;
            for (auto& local_vec : next_states_per_thread) {
                all_next_states.insert(all_next_states.end(), 
                                     std::make_move_iterator(local_vec.begin()), 
                                     std::make_move_iterator(local_vec.end()));
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

            std::cout << "  N=" << (n + 1) 
                      << " | Best Score: " << current_beam[0].side_length 
                      << " | Beam Size: " << current_beam.size() << std::endl;
        }

        return current_beam.empty() ? std::vector<ChristmasTree>{} : current_beam[0].trees;
    }
};