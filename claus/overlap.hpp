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
        if (std::abs(o1) > eps || std::abs(o2) > eps) return false;
        long double rr = r.x * r.x + r.y * r.y;
        if (rr < eps) return false;
        long double t0 = ((q1.x - p1.x) * r.x + (q1.y - p1.y) * r.y) / rr;
        long double t1 = ((q2.x - p1.x) * r.x + (q2.y - p1.y) * r.y) / rr;
        long double smin = std::min(t0, t1);
        long double smax = std::max(t0, t1);
        long double overlap_len = std::min(1.0L, smax) - std::max(0.0L, smin);
        return overlap_len > eps;
    }

    long double t = cross(sub(q1, p1), s) / d;
    long double u = cross(sub(q1, p1), r) / d;
    return t > eps && t < 1.0L - eps && u > eps && u < 1.0L - eps;
}

// Strict point-in-polygon: returns true if point is strictly inside
template<size_t N>
static inline bool point_in_polygon_strict(const std::array<DPoint, N>& poly, const DPoint& p, long double eps) {
    int wn = 0;
    for (size_t i = 0; i < N; ++i) {
        const DPoint& a = poly[i];
        const DPoint& b = poly[(i + 1) % N];
        // If on boundary, not strictly inside
        if (on_segment(a, b, p, eps)) return false;
        
        bool cond = ((a.y <= p.y) && (b.y > p.y)) || ((a.y > p.y) && (b.y <= p.y));
        if (cond) {
            long double x_intersect = a.x + (p.y - a.y) * (b.x - a.x) / (b.y - a.y);
            if (x_intersect > p.x) wn += (b.y > a.y) ? 1 : -1;
        }
    }
    return wn != 0;
}

// Main overlap check: mirrors validate_overlap.py logic (strict intersection + point-in-poly)
static inline bool polygons_strict_overlap(const ChristmasTree& A, const ChristmasTree& B, double buffer = 0.0) {
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

// R-Tree like structure for overlap detection
struct Box {
    double min_x, min_y, max_x, max_y;
    
    bool overlaps(const Box& other) const {
        return !(
            max_x < other.min_x || 
            min_x > other.max_x || 
            max_y < other.min_y || 
            min_y > other.max_y
        );
    }
    
    void expand(const Box& other) {
        min_x = std::min(min_x, other.min_x);
        min_y = std::min(min_y, other.min_y);
        max_x = std::max(max_x, other.max_x);
        max_y = std::max(max_y, other.max_y);
    }
    
    double area() const {
        return (max_x - min_x) * (max_y - min_y);
    }
};

struct RTreeNode {
    Box box;
    std::vector<std::pair<int, Box>> items; // ID + Box (for leaf)
    std::vector<RTreeNode*> children;       // Children (for internal)
    bool is_leaf;
    int height;

    RTreeNode() : is_leaf(true), height(0) {
        box = {1e18, 1e18, -1e18, -1e18};
    }
    
    ~RTreeNode() {
        for(auto c : children) delete c;
    }
};

class RTree {
public:
    RTree(const std::vector<ChristmasTree>& trees, int max_leaf_size = 8) : max_leaf_size(max_leaf_size) {
        // Bulk loading (STR-like)
        std::vector<int> indices(trees.size());
        std::iota(indices.begin(), indices.end(), 0);
        if (indices.empty()) {
            root = new RTreeNode();
            return;
        }
        root = build_recursive(trees, indices);
    }
    
    ~RTree() { delete root; }
    
    void query(const ChristmasTree& target, std::vector<int>& result) const {
        auto t_box = target.aabb();
        double sf = static_cast<double>(ChristmasTree::scale_factor);
        Box q_box;
        q_box.min_x = static_cast<double>(t_box.first.x) / sf;
        q_box.min_y = static_cast<double>(t_box.first.y) / sf;
        q_box.max_x = static_cast<double>(t_box.second.x) / sf;
        q_box.max_y = static_cast<double>(t_box.second.y) / sf;
        
        query_recursive(root, q_box, result);
    }

    // Dynamic update support
    void insert(int id, const ChristmasTree& tree) {
        auto t_box = tree.aabb();
        double sf = static_cast<double>(ChristmasTree::scale_factor);
        Box box;
        box.min_x = static_cast<double>(t_box.first.x) / sf;
        box.min_y = static_cast<double>(t_box.first.y) / sf;
        box.max_x = static_cast<double>(t_box.second.x) / sf;
        box.max_y = static_cast<double>(t_box.second.y) / sf;
        
        insert_recursive(root, id, box);
    }
    
    void remove(int id, const ChristmasTree& tree) {
        // We need the box to find it efficiently
        auto t_box = tree.aabb();
        double sf = static_cast<double>(ChristmasTree::scale_factor);
        Box box;
        box.min_x = static_cast<double>(t_box.first.x) / sf;
        box.min_y = static_cast<double>(t_box.first.y) / sf;
        box.max_x = static_cast<double>(t_box.second.x) / sf;
        box.max_y = static_cast<double>(t_box.second.y) / sf;
        
        remove_recursive(root, id, box);
    }

private:
    RTreeNode* root;
    int max_leaf_size;
    
    RTreeNode* build_recursive(const std::vector<ChristmasTree>& trees, std::vector<int>& indices) {
        RTreeNode* node = new RTreeNode();
        
        // Compute MBR of all items and store items
        for(int idx : indices) {
            auto t_box = trees[idx].aabb();
            double sf = static_cast<double>(ChristmasTree::scale_factor);
            Box item_box;
            item_box.min_x = static_cast<double>(t_box.first.x) / sf;
            item_box.min_y = static_cast<double>(t_box.first.y) / sf;
            item_box.max_x = static_cast<double>(t_box.second.x) / sf;
            item_box.max_y = static_cast<double>(t_box.second.y) / sf;
            
            node->box.expand(item_box);
            
            if (indices.size() <= (size_t)max_leaf_size) {
                node->items.push_back({idx, item_box});
            }
        }
        
        if (indices.size() <= (size_t)max_leaf_size) {
            node->is_leaf = true;
            node->height = 0;
            return node;
        }
        
        node->is_leaf = false;
        // Sort by x center
        std::sort(indices.begin(), indices.end(), [&](int a, int b) {
            return trees[a].center_x < trees[b].center_x;
        });
        
        int n_slices = std::ceil(std::sqrt((double)indices.size() / max_leaf_size));
        int slice_size = indices.size() / n_slices;
        if (slice_size < 1) slice_size = 1;

        for(int i=0; i<n_slices; ++i) {
            int start = i * slice_size;
            int end = (i == n_slices - 1) ? indices.size() : (i + 1) * slice_size;
            if (start >= indices.size()) break;
            std::vector<int> slice(indices.begin() + start, indices.begin() + end);
            
            // Sort slice by y center
            std::sort(slice.begin(), slice.end(), [&](int a, int b) {
                return trees[a].center_y < trees[b].center_y;
            });
            
            int n_children = std::ceil((double)slice.size() / max_leaf_size);
            int child_size = slice.size() / n_children;
            if (child_size < 1) child_size = 1;
            
            for(int j=0; j<n_children; ++j) {
                int c_start = j * child_size;
                int c_end = (j == n_children - 1) ? slice.size() : (j + 1) * child_size;
                if (c_start >= slice.size()) break;
                
                std::vector<int> child_indices(slice.begin() + c_start, slice.begin() + c_end);
                if(child_indices.empty()) continue;
                
                RTreeNode* child = build_recursive(trees, child_indices);
                node->children.push_back(child);
                node->box.expand(child->box);
                node->height = std::max(node->height, child->height + 1);
            }
        }
        return node;
    }
    
    void query_recursive(const RTreeNode* node, const Box& q_box, std::vector<int>& result) const {
        if (!node->box.overlaps(q_box)) return;
        
        if (node->is_leaf) {
            for(const auto& item : node->items) {
                if (item.second.overlaps(q_box)) {
                    result.push_back(item.first);
                }
            }
        } else {
            for(auto c : node->children) {
                query_recursive(c, q_box, result);
            }
        }
    }

    void insert_recursive(RTreeNode* node, int id, const Box& box) {
        node->box.expand(box);
        if (node->is_leaf) {
            node->items.push_back({id, box});
            if (node->items.size() > (size_t)max_leaf_size * 2) { // Allow some overflow or split
                 // Splitting not implemented for dynamic yet, just allow growth for simple SA
                 // or implement simple split.
                 // For now, allow growth. Performance degrades but correctness holds.
            }
        } else {
            // Choose best child
            RTreeNode* best_child = nullptr;
            double min_expansion = 1e18;
            for(auto c : node->children) {
                // Calculate expansion needed
                Box new_box = c->box;
                new_box.expand(box);
                double expansion = new_box.area() - c->box.area();
                if (expansion < min_expansion) {
                    min_expansion = expansion;
                    best_child = c;
                }
            }
            if (best_child) insert_recursive(best_child, id, box);
            else { 
                // Should not happen if built correctly, but if node has no children?
                // Create one?
            }
        }
    }
    
    bool remove_recursive(RTreeNode* node, int id, const Box& box) {
        if (!node->box.overlaps(box)) return false;
        
        if (node->is_leaf) {
            for(auto it = node->items.begin(); it != node->items.end(); ++it) {
                if (it->first == id) {
                    node->items.erase(it);
                    // Recompute MBR
                    node->box = {1e18, 1e18, -1e18, -1e18};
                    for(const auto& item : node->items) node->box.expand(item.second);
                    return true;
                }
            }
        } else {
            for(auto c : node->children) {
                if (remove_recursive(c, id, box)) {
                    // Recompute MBR
                    node->box = {1e18, 1e18, -1e18, -1e18};
                    for(auto child : node->children) node->box.expand(child->box);
                    return true;
                }
            }
        }
        return false;
    }
};

static inline bool has_any_overlap(const std::vector<ChristmasTree>& trees, double buffer = 0.0) {
    size_t n = trees.size();
    if (n == 0) return false;

    // Use GPU for large N
    // GPU now supports custom buffer and strict overlap logic.
    if (n > 50 && GpuContext::getInstance().is_valid()) {
        return GpuContext::getInstance().has_overlap(trees, buffer);
    }

    // Build RTree
    RTree rt(trees, 8);

    bool found = false;

    #pragma omp parallel for schedule(dynamic) shared(found)
    for (size_t i = 0; i < n; ++i) {
        if (found) continue;
        
        std::vector<int> candidates;
        candidates.reserve(32);
        
        // Const cast to call query (which is logically const but not marked const in my impl)
        const_cast<RTree&>(rt).query(trees[i], candidates);
        
        for (int j : candidates) {
            if (found) break;
            if ((size_t)j <= i) continue; // Only check j > i

            if (polygons_strict_overlap(trees[i], trees[j], buffer)) {
                #pragma omp atomic write
                found = true;
            }
        }
    }
    return found;
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
