#pragma once
#include <vector>
#include <array>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <omp.h>
#include "tree.hpp"
#include "gpu/gpu_context.hpp"

namespace overlap {

struct DPoint {
    long double x;
    long double y;
};

// Convert tree polygon to unit-scale doubles for geometric checks, with optional buffering
static inline std::array<DPoint, 15> to_unit_poly_buffered(const ChristmasTree& tree, double buffer = 0.0) {
    std::array<DPoint, 15> out;
    long double sf = ChristmasTree::scale_factor;
    long double cx = tree.center_x;
    long double cy = tree.center_y;
    long double scale = 1.0 + buffer;

    for (size_t i = 0; i < 15; ++i) {
        const auto& p = tree.polygon[i];
        long double px = static_cast<long double>(p.x) / sf;
        long double py = static_cast<long double>(p.y) / sf;
        // Expand relative to center
        out[i] = {cx + (px - cx) * scale, cy + (py - cy) * scale};
    }
    return out;
}

static inline long double cross(const DPoint& a, const DPoint& b) {
    return a.x * b.y - a.y * b.x;
}

static inline DPoint sub(const DPoint& a, const DPoint& b) {
    return {a.x - b.x, a.y - b.y};
}

// Check if point p lies on segment ab (with epsilon tolerance)
static inline bool on_segment(const DPoint& a, const DPoint& b, const DPoint& p, long double eps) {
    long double minx = std::min(a.x, b.x) - eps;
    long double maxx = std::max(a.x, b.x) + eps;
    long double miny = std::min(a.y, b.y) - eps;
    long double maxy = std::max(a.y, b.y) + eps;
    long double c = std::abs(cross(sub(b, a), sub(p, a)));
    return c <= eps && p.x >= minx && p.x <= maxx && p.y >= miny && p.y <= maxy;
}

// Strict segment intersection: returns true only if segments cross strictly (excluding endpoints)
static inline bool segments_strict_intersect(const DPoint& p1, const DPoint& p2, const DPoint& q1, const DPoint& q2, long double eps) {
    DPoint r = sub(p2, p1);
    DPoint s = sub(q2, q1);
    long double d = cross(r, s);
    long double o1 = cross(sub(p2, p1), sub(q1, p1));
    long double o2 = cross(sub(p2, p1), sub(q2, p1));

    if (std::abs(d) < eps) {
        return false; // Parallel segments treated as non-intersecting (touching allowed)
    }

    long double t = cross(sub(q1, p1), s) / d;
    long double u = cross(sub(q1, p1), r) / d;
    return t > eps && t < 1.0L - eps && u > eps && u < 1.0L - eps;
}

// Strict point-in-polygon: returns true if point is strictly inside
template<size_t N>
static inline bool point_in_polygon_strict(const std::array<DPoint, N>& poly, const DPoint& p, long double eps) {
    bool inside = false;
    for (size_t i = 0; i < N; ++i) {
        const DPoint& a = poly[i];
        const DPoint& b = poly[(i + 1) % N];
        // If on boundary, not strictly inside
        if (on_segment(a, b, p, eps)) return false;
        
        // Ray casting with epsilon shift (from preprocess.py)
        long double py_adj = p.y + eps;
        if ((a.y > py_adj) != (b.y > py_adj)) {
            long double x_intersect = (b.x - a.x) * (py_adj - a.y) / (b.y - a.y) + a.x;
            if (p.x < x_intersect - eps) {
                inside = !inside;
            }
        }
    }
    return inside;
}

// Main overlap check: mirrors validate_overlap.py logic (strict intersection + point-in-poly)
static inline bool polygons_strict_overlap(const ChristmasTree& A, const ChristmasTree& B, double buffer = 0.0) {
    // Check for identical or nearly identical pose (stacked trees)
    long double dx = A.center_x - B.center_x;
    long double dy = A.center_y - B.center_y;
    long double dist_sq = dx*dx + dy*dy;
    
    // Angle difference normalized
    long double da = std::abs(A.angle_deg - B.angle_deg);
    while (da >= 360.0L) da -= 360.0L;
    if (da > 180.0L) da = 360.0L - da;

    // If practically identical, they overlap
    if (dist_sq < 1e-10L && da < 1e-4L) {
        return true;
    }

    auto Ad = to_unit_poly_buffered(A, buffer);
    auto Bd = to_unit_poly_buffered(B, buffer);
    // Use epsilon 1e-12 as requested (relaxed from 1e-14)
    long double eps = 1e-12L; 

    size_t na = 15, nb = 15;
    
    // 1. Check strict edge intersections
    for (size_t i = 0; i < na; ++i) {
        DPoint a1 = Ad[i];
        DPoint a2 = Ad[(i + 1) % na];
        for (size_t j = 0; j < nb; ++j) {
            DPoint b1 = Bd[j];
            DPoint b2 = Bd[(j + 1) % nb];
            if (segments_strict_intersect(a1, a2, b1, b2, eps)) return true;
        }
    }

    // 2. Check if any vertex of B is strictly inside A
    for (const auto& p : Bd) {
        if (point_in_polygon_strict(Ad, p, eps)) return true;
    }

    // 3. Check if any vertex of A is strictly inside B
    for (const auto& p : Ad) {
        if (point_in_polygon_strict(Bd, p, eps)) return true;
    }

    return false;
}

// Approximate overlap area (very expensive if high res, use Monte Carlo for speed)
static inline double calculate_overlap_area_monte_carlo(const std::vector<ChristmasTree>& trees, int samples_per_tree = 100) {
    // Monte Carlo approach:
    // For each tree, generate random points inside its bounding box.
    // If point is inside the tree AND inside another tree, it counts as overlap.
    // This is approximate but differentiable-ish for SA.
    
    double total_overlap = 0.0;
    std::mt19937 rng(12345);
    
    for (size_t i = 0; i < trees.size(); ++i) {
        auto box = trees[i].aabb();
        double min_x = (double)box.first.x / (double)ChristmasTree::scale_factor;
        double max_x = (double)box.second.x / (double)ChristmasTree::scale_factor;
        double min_y = (double)box.first.y / (double)ChristmasTree::scale_factor;
        double max_y = (double)box.second.y / (double)ChristmasTree::scale_factor;
        
        double area_box = (max_x - min_x) * (max_y - min_y);
        if (area_box < 1e-9) continue;
        
        std::uniform_real_distribution<double> dx(min_x, max_x);
        std::uniform_real_distribution<double> dy(min_y, max_y);
        
        auto poly_i = to_unit_poly_buffered(trees[i]);
        int overlap_hits = 0;
        int inside_hits = 0;
        
        for (int k = 0; k < samples_per_tree; ++k) {
            DPoint p = {dx(rng), dy(rng)};
            
            // Check if inside tree i
            if (point_in_polygon_strict(poly_i, p, 0.0)) {
                inside_hits++;
                // Check if inside any other tree
                bool overlap = false;
                for (size_t j = 0; j < trees.size(); ++j) {
                    if (i == j) continue;
                    // Fast AABB check
                    auto box_j = trees[j].aabb();
                    double j_min_x = (double)box_j.first.x / (double)ChristmasTree::scale_factor;
                    double j_max_x = (double)box_j.second.x / (double)ChristmasTree::scale_factor;
                    double j_min_y = (double)box_j.first.y / (double)ChristmasTree::scale_factor;
                    double j_max_y = (double)box_j.second.y / (double)ChristmasTree::scale_factor;
                    
                    if (p.x < j_min_x || p.x > j_max_x || p.y < j_min_y || p.y > j_max_y) continue;
                    
                    auto poly_j = to_unit_poly_buffered(trees[j]);
                    if (point_in_polygon_strict(poly_j, p, 0.0)) {
                        overlap = true;
                        break;
                    }
                }
                if (overlap) overlap_hits++;
            }
        }
        
        if (inside_hits > 0) {
            // Overlap ratio * Tree Area (approx)
            // Tree area is constant ~0.2. 
            // Better: (overlap_hits / samples) * box_area
            total_overlap += (double)overlap_hits / (double)samples_per_tree * area_box;
        }
    }
    return total_overlap; // Double counting? Yes, A-B and B-A. OK for penalty.
}

static inline bool boxes_overlap(const std::pair<TreePoint, TreePoint>& a, const std::pair<TreePoint, TreePoint>& b) {
    return !(
        a.second.x < b.first.x ||
        b.second.x < a.first.x ||
        a.second.y < b.first.y ||
        b.second.y < a.first.y
    );
}

struct AABBInfo {
    long double min_x, max_x, min_y, max_y;
    int id;
};

static inline bool has_any_overlap(const std::vector<ChristmasTree>& trees, double buffer = 0.0) {
    size_t n = trees.size();
    if (n < 2) return false;

    // Use GPU for large N if valid (disabled for precision reasons currently)
    if (n > 5000 && GpuContext::getInstance().is_valid()) {
       return GpuContext::getInstance().has_overlap(trees, buffer);
    }

    // 1. Precompute AABBs
    std::vector<AABBInfo> boxes(n);
    long double sf = ChristmasTree::scale_factor;
    long double scale = 1.0 + buffer;

    for(size_t i=0; i<n; ++i) {
        auto pair = trees[i].aabb(); 
        if (buffer != 0.0) {
            long double cx = trees[i].center_x * sf;
            long double cy = trees[i].center_y * sf;
            boxes[i] = {
                cx + (pair.first.x - cx) * scale,
                cx + (pair.second.x - cx) * scale,
                cy + (pair.first.y - cy) * scale,
                cy + (pair.second.y - cy) * scale,
                (int)i
            };
        } else {
            boxes[i] = {pair.first.x, pair.second.x, pair.first.y, pair.second.y, (int)i};
        }
    }

    // 2. Sort by min_x
    std::sort(boxes.begin(), boxes.end(), [](const AABBInfo& a, const AABBInfo& b) {
        return a.min_x < b.min_x;
    });

    // 3. Sweep
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
             // Prune based on sorted X
             if (boxes[j].min_x > boxes[i].max_x) break; 
             
             // Check Y overlap
             if (boxes[j].min_y > boxes[i].max_y || boxes[j].max_y < boxes[i].min_y) continue;
             
             // Detailed check
             int id1 = boxes[i].id;
             int id2 = boxes[j].id;
             if (polygons_strict_overlap(trees[id1], trees[id2], buffer)) return true;
        }
    }
    return false;
}

static inline long double calculate_moment_of_inertia(const std::vector<ChristmasTree>& trees) {
    long double sum_sq = 0.0L;
    for (const auto& t : trees) {
        sum_sq += (t.center_x * t.center_x + t.center_y * t.center_y);
    }
    return sum_sq;
}

static inline bool has_overlap_with_others(const std::vector<ChristmasTree>& trees, size_t idx) {
    size_t n = trees.size();
    if (idx >= n) return false;

    const auto& target = trees[idx];
    auto box_a = target.aabb();

    // Check against all others
    for (size_t i = 0; i < n; ++i) {
        if (i == idx) continue;
        
        auto box_b = trees[i].aabb();
        if (boxes_overlap(box_a, box_b)) {
            if (polygons_strict_overlap(target, trees[i])) {
                return true;
            }
        }
    }
    return false;
}

// Overload for checking a candidate tree against existing set
static inline bool has_overlap_with_others(const std::vector<ChristmasTree>& trees, const ChristmasTree& target) {
    auto box_a = target.aabb();
    for (const auto& t : trees) {
        auto box_b = t.aabb();
        if (boxes_overlap(box_a, box_b)) {
            if (polygons_strict_overlap(target, t)) {
                return true;
            }
        }
    }
    return false;
}

static inline long double calculate_score(const std::vector<ChristmasTree>& trees) {
    if (trees.empty()) return 0.0L;
    
    long double min_x = 1e18L, min_y = 1e18L;
    long double max_x = -1e18L, max_y = -1e18L;

    for (const auto& t : trees) {
        auto box = t.aabb();
        if (box.first.x < min_x) min_x = box.first.x;
        if (box.first.y < min_y) min_y = box.first.y;
        if (box.second.x > max_x) max_x = box.second.x;
        if (box.second.y > max_y) max_y = box.second.y;
    }

    // Convert back from scaled coords for score calculation?
    // ChristmasTree stores scaled coords in polygon.
    // Scale factor is 1e15.
    // Score should be based on unscaled coords.
    // The AABB returns scaled coords.
    
    long double sf = ChristmasTree::scale_factor;
    long double w = (max_x - min_x) / sf;
    long double h = (max_y - min_y) / sf;
    long double side = std::max(w, h);
    
    return (side * side) / static_cast<long double>(trees.size());
}

} // namespace overlap
