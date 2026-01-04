#pragma once
#include <vector>
#include <cmath>
#include <random>
#include <iostream>
#include <algorithm>
#include "tree.hpp"
#include "overlap.hpp"

namespace symmetry {

struct RingConfig {
    int count;
    double radius;
    double phase;          // Start angle
    double angle_offset;   // Orientation of trees relative to position angle (tangential offset)
    double global_angle;   // Global orientation (if not tangential)
    bool tangential;       // If true, tree rotates with position
};

struct SymmetryState {
    std::vector<RingConfig> rings;
    double global_scale = 1.0;
};

inline std::vector<ChristmasTree> generate_from_symmetry(const SymmetryState& state) {
    std::vector<ChristmasTree> trees;
    double scale = state.global_scale;
    
    for (const auto& ring : state.rings) {
        if (ring.count == 0) continue;
        
        double d_theta = 360.0 / ring.count;
        for (int i = 0; i < ring.count; ++i) {
            double theta = ring.phase + i * d_theta;
            double rad = theta * (M_PI / 180.0);
            
            double x = ring.radius * scale * std::cos(rad);
            double y = ring.radius * scale * std::sin(rad);
            
            double tree_angle = 0.0;
            if (ring.tangential) {
                tree_angle = theta + ring.angle_offset;
            } else {
                tree_angle = ring.global_angle + (i % 2 == 0 ? 0 : ring.angle_offset); // Alternating? Or just global?
                // Let's stick to simple global for now, maybe with alternating parity logic later
                tree_angle = ring.global_angle;
            }
            
            // Normalize angle
            tree_angle = std::fmod(tree_angle, 360.0);
            
            trees.push_back(ChristmasTree(x, y, tree_angle));
        }
    }
    return trees;
}

inline long double score_symmetric(const std::vector<ChristmasTree>& trees) {
    if (overlap::has_any_overlap(trees)) {
        return 1e18; // Penalty
    }
    
    long double min_x = 1e18, min_y = 1e18, max_x = -1e18, max_y = -1e18;
    for (const auto& t : trees) {
        auto box = t.aabb();
        min_x = std::min(min_x, box.first.x);
        min_y = std::min(min_y, box.first.y);
        max_x = std::max(max_x, box.second.x);
        max_y = std::max(max_y, box.second.y);
    }
    
    long double sf = ChristmasTree::scale_factor;
    long double w = (max_x - min_x) / sf;
    long double h = (max_y - min_y) / sf;
    return std::max(w, h);
}

inline std::vector<ChristmasTree> optimize_symmetry(int n, int seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist_rad(0.5, 10.0);
    std::uniform_real_distribution<double> dist_angle(0.0, 360.0);
    std::uniform_real_distribution<double> dist01(0.0, 1.0);
    
    // Partitions of N into rings
    std::vector<std::vector<int>> partitions;
    
    // 1 Ring
    partitions.push_back({n});
    
    // 2 Rings
    for (int i = 1; i <= n/2; ++i) {
        partitions.push_back({i, n - i});
    }
    
    // 3 Rings (for larger N)
    if (n >= 10) {
        for (int i = 1; i < n/3; ++i) {
            for (int j = i; j < (n-i)/2; ++j) {
                int k = n - i - j;
                partitions.push_back({i, j, k});
            }
        }
    }
    
    // Center point? (Ring of 1 with radius 0)
    // Handled by partition logic if we allow radius -> 0 optimization
    
    double best_score = 1e18;
    std::vector<ChristmasTree> best_trees;
    
    for (const auto& part : partitions) {
        // Initialize state
        SymmetryState current;
        for (int count : part) {
            RingConfig r;
            r.count = count;
            r.radius = (count == 1) ? 0.0 : dist_rad(rng) * (count / 6.0); // Heuristic radius
            r.phase = dist_angle(rng);
            r.angle_offset = dist_angle(rng);
            r.global_angle = dist_angle(rng);
            r.tangential = (dist01(rng) > 0.5);
            current.rings.push_back(r);
        }
        
        // SA Loop for parameters
        double T = 1.0;
        double Tmin = 1e-4;
        double decay = 0.99;
        int steps = 2000;
        
        // Initial shrinking to find valid state
        // Actually, just run SA directly on score (with penalty)
        
        double current_obj = score_symmetric(generate_from_symmetry(current));
        SymmetryState best_local = current;
        double best_local_obj = current_obj;
        
        for (int s = 0; s < steps; ++s) {
            SymmetryState next = current;
            // Mutate
            int r_idx = std::uniform_int_distribution<int>(0, (int)part.size() - 1)(rng);
            int type = std::uniform_int_distribution<int>(0, 4)(rng);
            double mag = 0.1 * T;
            
            if (type == 0) next.rings[r_idx].radius += (dist01(rng) - 0.5) * mag * 5.0;
            if (type == 1) next.rings[r_idx].phase += (dist01(rng) - 0.5) * mag * 360.0;
            if (type == 2) next.rings[r_idx].angle_offset += (dist01(rng) - 0.5) * mag * 360.0;
            if (type == 3) next.rings[r_idx].global_angle += (dist01(rng) - 0.5) * mag * 360.0;
            if (type == 4 && dist01(rng) < 0.05) next.rings[r_idx].tangential = !next.rings[r_idx].tangential;
            
            // Constrain radius
            if (next.rings[r_idx].radius < 0) next.rings[r_idx].radius = 0;
            
            auto trees = generate_from_symmetry(next);
            double next_obj = score_symmetric(trees);
            
            // Accept/Reject
            if (next_obj < current_obj || std::exp((current_obj - next_obj) / T) > dist01(rng)) {
                current = next;
                current_obj = next_obj;
                if (current_obj < best_local_obj) {
                    best_local = current;
                    best_local_obj = current_obj;
                }
            }
            
            T *= decay;
            if (T < Tmin) break;
        }
        
        if (best_local_obj < best_score) {
            best_score = best_local_obj;
            best_trees = generate_from_symmetry(best_local);
        }
    }
    
    return best_trees;
}

} // namespace symmetry
