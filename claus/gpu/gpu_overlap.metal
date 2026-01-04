#include <metal_stdlib>
using namespace metal;

struct TreeData {
    float x;
    float y;
    float angle; // in degrees
    float padding; // Align to 16 bytes
};

struct Point {
    float x;
    float y;
};

constant Point base_poly[15] = {
    {0.0, 0.8},
    {0.125, 0.5},
    {0.0625, 0.5},
    {0.2, 0.25},
    {0.1, 0.25},
    {0.35, 0.0},
    {0.075, 0.0},
    {0.075, -0.2},
    {-0.075, -0.2},
    {-0.075, 0.0},
    {-0.35, 0.0},
    {-0.1, 0.25},
    {-0.2, 0.25},
    {-0.0625, 0.5},
    {-0.125, 0.5}
};

float cross_prod(Point a, Point b) {
    return a.x * b.y - a.y * b.x;
}

Point sub(Point a, Point b) {
    return {a.x - b.x, a.y - b.y};
}

bool segments_strict_intersect(Point p1, Point p2, Point q1, Point q2) {
    float eps = 1e-12;
    Point r = sub(p2, p1);
    Point s = sub(q2, q1);
    float d = cross_prod(r, s);
    
    if (abs(d) < eps) {
        return false;
    }

    float t = cross_prod(sub(q1, p1), s) / d;
    float u = cross_prod(sub(q1, p1), r) / d;
    return t > eps && t < 1.0 - eps && u > eps && u < 1.0 - eps;
}

bool point_in_polygon(Point p, Point poly[15]) {
    bool inside = false;
    float eps = 1e-12;
    for (int i = 0; i < 15; ++i) {
        Point a = poly[i];
        Point b = poly[(i + 1) % 15];
        
        if (abs(cross_prod(sub(b, a), sub(p, a))) <= eps) {
            float minx = min(a.x, b.x) - eps;
            float maxx = max(a.x, b.x) + eps;
            float miny = min(a.y, b.y) - eps;
            float maxy = max(a.y, b.y) + eps;
            if (p.x >= minx && p.x <= maxx && p.y >= miny && p.y <= maxy) return false;
        }
        
        float py_adj = p.y + eps;
        if ((a.y > py_adj) != (b.y > py_adj)) {
            float x_intersect = (b.x - a.x) * (py_adj - a.y) / (b.y - a.y) + a.x;
            if (p.x < x_intersect - eps) {
                inside = !inside;
            }
        }
    }
    return inside;
}

kernel void check_overlaps(
    device const TreeData* trees [[ buffer(0) ]],
    device atomic_int* result [[ buffer(1) ]],
    constant int& n_trees [[ buffer(2) ]],
    constant float& buffer_size [[ buffer(3) ]],
    uint2 id [[ thread_position_in_grid ]]
) {
    if (atomic_load_explicit(result, memory_order_relaxed) > 0) return;

    int n = n_trees;
    int i = id.x;
    int j = id.y;

    if (i >= n || j >= n) return;
    if (i >= j) return; 

    TreeData t1 = trees[i];
    TreeData t2 = trees[j];
    
    float dx = t1.x - t2.x;
    float dy = t1.y - t2.y;
    float dist_sq = dx*dx + dy*dy;
    float da = abs(t1.angle - t2.angle);
    if (dist_sq < 1e-8f && da < 1e-3f) {
        atomic_store_explicit(result, 1, memory_order_relaxed);
        return;
    }

    float scale = 1.0 + buffer_size;

    Point poly1[15];
    Point poly2[15];
    
    float rad1 = t1.angle * 3.14159265358979323846f / 180.0f;
    float c1 = cos(rad1);
    float s1 = sin(rad1);
    
    float min_x1 = 1e9, min_y1 = 1e9, max_x1 = -1e9, max_y1 = -1e9;
    
    for (int k = 0; k < 15; ++k) {
        Point p = base_poly[k];
        float rx = (p.x * c1 - p.y * s1) * scale;
        float ry = (p.x * s1 + p.y * c1) * scale;
        poly1[k] = {rx + t1.x, ry + t1.y};
        min_x1 = min(min_x1, poly1[k].x);
        min_y1 = min(min_y1, poly1[k].y);
        max_x1 = max(max_x1, poly1[k].x);
        max_y1 = max(max_y1, poly1[k].y);
    }
    
    float rad2 = t2.angle * 3.14159265358979323846f / 180.0f;
    float c2 = cos(rad2);
    float s2 = sin(rad2);
    
    float min_x2 = 1e9, min_y2 = 1e9, max_x2 = -1e9, max_y2 = -1e9;
    
    for (int k = 0; k < 15; ++k) {
        Point p = base_poly[k];
        float rx = (p.x * c2 - p.y * s2) * scale;
        float ry = (p.x * s2 + p.y * c2) * scale;
        poly2[k] = {rx + t2.x, ry + t2.y};
        min_x2 = min(min_x2, poly2[k].x);
        min_y2 = min(min_y2, poly2[k].y);
        max_x2 = max(max_x2, poly2[k].x);
        max_y2 = max(max_y2, poly2[k].y);
    }
    
    if (max_x1 < min_x2 || max_x2 < min_x1 || max_y1 < min_y2 || max_y2 < min_y1) return;
    
    for (int a = 0; a < 15; ++a) {
        Point p1 = poly1[a];
        Point p2 = poly1[(a + 1) % 15];
        for (int b = 0; b < 15; ++b) {
            Point q1 = poly2[b];
            Point q2 = poly2[(b + 1) % 15];
            if (segments_strict_intersect(p1, p2, q1, q2)) {
                atomic_store_explicit(result, 1, memory_order_relaxed);
                return;
            }
        }
    }
    
    for (int k = 0; k < 15; ++k) {
        if (point_in_polygon(poly2[k], poly1)) {
            atomic_store_explicit(result, 1, memory_order_relaxed);
            return;
        }
        if (point_in_polygon(poly1[k], poly2)) {
            atomic_store_explicit(result, 1, memory_order_relaxed);
            return;
        }
    }
}

struct PhysicsParams {
    float repulsion_strength;
    float gravity_strength;
    float learning_rate;
    float buffer_val;
};

kernel void physics_step(
    device const TreeData* trees_in [[ buffer(0) ]],
    device TreeData* trees_out [[ buffer(1) ]],
    constant int& n_trees [[ buffer(2) ]],
    constant PhysicsParams& params [[ buffer(3) ]],
    uint id [[ thread_position_in_grid ]]
) {
    if (id >= (uint)n_trees) return;

    TreeData t1 = trees_in[id];
    float2 force = float2(0.0f, 0.0f);
    
    force.x -= t1.x * params.gravity_strength;
    force.y -= t1.y * params.gravity_strength;
    
    Point poly1[15];
    float min_x1 = 1e9, min_y1 = 1e9, max_x1 = -1e9, max_y1 = -1e9;
    
    float rad1 = t1.angle * 3.14159265f / 180.0f;
    float c1 = cos(rad1);
    float s1 = sin(rad1);
    float scale = 1.0f + params.buffer_val; 
    
    for (int k = 0; k < 15; ++k) {
        Point p = base_poly[k];
        float rx = (p.x * c1 - p.y * s1) * scale;
        float ry = (p.x * s1 + p.y * c1) * scale;
        poly1[k] = {rx + t1.x, ry + t1.y};
        min_x1 = min(min_x1, poly1[k].x);
        min_y1 = min(min_y1, poly1[k].y);
        max_x1 = max(max_x1, poly1[k].x);
        max_y1 = max(max_y1, poly1[k].y);
    }
    
    for (int j = 0; j < n_trees; ++j) {
        if (id == (uint)j) continue;
        
        TreeData t2 = trees_in[j];
        float dx = t1.x - t2.x;
        float dy = t1.y - t2.y;
        float dist_sq = dx*dx + dy*dy;
        
        if (dist_sq < 4.0f) { 
            float dist = sqrt(dist_sq);
            if (dist < 1e-6f) {
                float seed = (float)(id * 12345 + j * 6789);
                dx = fract(sin(seed) * 43758.5453) - 0.5f;
                dy = fract(cos(seed) * 43758.5453) - 0.5f;
                dist = 1e-3f;
            }
            
            bool is_overlapping = false;
            
            Point poly2[15];
            float min_x2 = 1e9, min_y2 = 1e9, max_x2 = -1e9, max_y2 = -1e9;
            
            float rad2 = t2.angle * 3.14159265f / 180.0f;
            float c2 = cos(rad2);
            float s2 = sin(rad2);
            
            for (int k = 0; k < 15; ++k) {
                Point p = base_poly[k];
                float rx = (p.x * c2 - p.y * s2) * scale;
                float ry = (p.x * s2 + p.y * c2) * scale;
                poly2[k] = {rx + t2.x, ry + t2.y};
                min_x2 = min(min_x2, poly2[k].x);
                min_y2 = min(min_y2, poly2[k].y);
                max_x2 = max(max_x2, poly2[k].x);
                max_y2 = max(max_y2, poly2[k].y);
            }
            
            if (!(max_x1 < min_x2 || max_x2 < min_x1 || max_y1 < min_y2 || max_y2 < min_y1)) {
                bool overlap_found = false;
                
                for (int a = 0; a < 15 && !overlap_found; ++a) {
                    Point p1 = poly1[a];
                    Point p2 = poly1[(a + 1) % 15];
                    for (int b = 0; b < 15; ++b) {
                        Point q1 = poly2[b];
                        Point q2 = poly2[(b + 1) % 15];
                        if (segments_strict_intersect(p1, p2, q1, q2)) {
                            overlap_found = true;
                            break;
                        }
                    }
                }
                
                if (!overlap_found) {
                     for (int k = 0; k < 15; ++k) {
                         if (point_in_polygon(poly2[k], poly1)) {
                             overlap_found = true; break;
                         }
                     }
                }
                if (!overlap_found) {
                     for (int k = 0; k < 15; ++k) {
                         if (point_in_polygon(poly1[k], poly2)) {
                             overlap_found = true; break;
                         }
                     }
                }
                
                is_overlapping = overlap_found;
            }
            
            if (is_overlapping) {
                float f = params.repulsion_strength * (2.0f - dist) / dist;
                force.x += dx * f;
                force.y += dy * f;
            } else if (dist < 1.2f) {
                float f = params.repulsion_strength * 0.1f * (1.2f - dist) / dist;
                force.x += dx * f;
                force.y += dy * f;
            }
        }
    }
    
    float new_x = t1.x + force.x * params.learning_rate;
    float new_y = t1.y + force.y * params.learning_rate;
    
    new_x = clamp(new_x, -100.0f, 100.0f);
    new_y = clamp(new_y, -100.0f, 100.0f);
    
    trees_out[id].x = new_x;
    trees_out[id].y = new_y;
    trees_out[id].angle = t1.angle;
    trees_out[id].padding = t1.padding;
}

struct SharedPoly {
    Point points[15];
    float min_x, min_y, max_x, max_y;
};

kernel void check_overlaps_shared(
    device const TreeData* trees [[ buffer(0) ]],
    device atomic_int* result [[ buffer(1) ]],
    constant float& buffer_val [[ buffer(2) ]],
    uint tid [[ thread_position_in_threadgroup ]],
    uint block_size [[ threads_per_threadgroup ]]
) {
    threadgroup SharedPoly shared_polys[240];

    if (tid >= block_size) return;

    TreeData t = trees[tid];
    float scale = 1.0 + buffer_val;
    float rad = t.angle * 3.14159265 / 180.0;
    float c = cos(rad);
    float s = sin(rad);

    float min_x = 1e9, min_y = 1e9, max_x = -1e9, max_y = -1e9;

    for (int k = 0; k < 15; ++k) {
        Point p = base_poly[k];
        float rx = (p.x * c - p.y * s) * scale;
        float ry = (p.x * s + p.y * c) * scale;
        Point transformed = {rx + t.x, ry + t.y};
        shared_polys[tid].points[k] = transformed;
        
        min_x = min(min_x, transformed.x);
        min_y = min(min_y, transformed.y);
        max_x = max(max_x, transformed.x);
        max_y = max(max_y, transformed.y);
    }
    shared_polys[tid].min_x = min_x;
    shared_polys[tid].min_y = min_y;
    shared_polys[tid].max_x = max_x;
    shared_polys[tid].max_y = max_y;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (atomic_load_explicit(result, memory_order_relaxed) > 0) return;
    
    float my_min_x = shared_polys[tid].min_x;
    float my_min_y = shared_polys[tid].min_y;
    float my_max_x = shared_polys[tid].max_x;
    float my_max_y = shared_polys[tid].max_y;
    float my_angle = trees[tid].angle;
    
    for (uint j = tid + 1; j < block_size; ++j) {
        float dx = trees[tid].x - trees[j].x;
        float dy = trees[tid].y - trees[j].y;
        float dist_sq = dx*dx + dy*dy;
        float da = abs(my_angle - trees[j].angle);
        if (dist_sq < 1e-8f && da < 1e-3f) {
             atomic_store_explicit(result, 1, memory_order_relaxed);
             return;
        }

        float other_min_x = shared_polys[j].min_x;
        float other_min_y = shared_polys[j].min_y;
        float other_max_x = shared_polys[j].max_x;
        float other_max_y = shared_polys[j].max_y;
        
        if (my_max_x < other_min_x || other_max_x < my_min_x || 
            my_max_y < other_min_y || other_max_y < my_min_y) continue;
            
        for (int a = 0; a < 15; ++a) {
            Point p1 = shared_polys[tid].points[a];
            Point p2 = shared_polys[tid].points[(a + 1) % 15];
            for (int b = 0; b < 15; ++b) {
                Point q1 = shared_polys[j].points[b];
                Point q2 = shared_polys[j].points[(b + 1) % 15];
                if (segments_strict_intersect(p1, p2, q1, q2)) {
                    atomic_store_explicit(result, 1, memory_order_relaxed);
                    return;
                }
            }
        }
        
        for (int k = 0; k < 15; ++k) {
            Point p = shared_polys[j].points[k];
            int wn = 0;
            float eps = 1e-12;
            for (int i = 0; i < 15; ++i) {
                Point a = shared_polys[tid].points[i];
                Point b = shared_polys[tid].points[(i + 1) % 15];
                if (abs(cross_prod(sub(b, a), sub(p, a))) <= eps) {
                     float mx = min(a.x, b.x) - eps;
                     float Mx = max(a.x, b.x) + eps;
                     float my = min(a.y, b.y) - eps;
                     float My = max(a.y, b.y) + eps;
                     if (p.x >= mx && p.x <= Mx && p.y >= my && p.y <= My) {
                         wn = 0; break; 
                     }
                }
                bool cond = ((a.y <= p.y) && (b.y > p.y)) || ((a.y > p.y) && (b.y <= p.y));
                if (cond) {
                    float x_intersect = a.x + (p.y - a.y) * (b.x - a.x) / (b.y - a.y);
                    if (x_intersect > p.x) wn += (b.y > a.y) ? 1 : -1;
                }
            }
            if (wn != 0) {
                atomic_store_explicit(result, 1, memory_order_relaxed);
                return;
            }
        }
        
        for (int k = 0; k < 15; ++k) {
            Point p = shared_polys[tid].points[k];
            int wn = 0;
            float eps = 1e-12;
            for (int i = 0; i < 15; ++i) {
                Point a = shared_polys[j].points[i];
                Point b = shared_polys[j].points[(i + 1) % 15];
                if (abs(cross_prod(sub(b, a), sub(p, a))) <= eps) {
                     float mx = min(a.x, b.x) - eps;
                     float Mx = max(a.x, b.x) + eps;
                     float my = min(a.y, b.y) - eps;
                     float My = max(a.y, b.y) + eps;
                     if (p.x >= mx && p.x <= Mx && p.y >= my && p.y <= My) {
                         wn = 0; break; 
                     }
                }
                bool cond = ((a.y <= p.y) && (b.y > p.y)) || ((a.y > p.y) && (b.y <= p.y));
                if (cond) {
                    float x_intersect = a.x + (p.y - a.y) * (b.x - a.x) / (b.y - a.y);
                    if (x_intersect > p.x) wn += (b.y > a.y) ? 1 : -1;
                }
            }
            if (wn != 0) {
                atomic_store_explicit(result, 1, memory_order_relaxed);
                return;
            }
        }
    }
}

// --- Batch SA Optimization ---

struct SAParams {
    float Tmax;
    float Tmin;
    float cooling_factor;
    int nsteps;
    float position_delta;
    float angle_delta;
};

uint rand_xorshift(thread uint& state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}
float rand_float(thread uint& state) {
    return float(rand_xorshift(state)) * 2.3283064365386963e-10f; 
}

// Check overlap between two transformed polygons using SharedPoly struct (redefined for SA)
// Actually we can reuse SharedPoly but we need to populate it.
// Or just compute on fly. Compute on fly is better for single pair check to save shared mem bandwidth/bank conflicts?
// Actually shared mem is fast.
// Let's use shared mem for trees, but compute poly on fly for the check.

bool check_overlap_pair(TreeData t1, TreeData t2) {
    // Quick AABB
    // Need to transform.
    // Let's implement full check here.
    
    float scale = 1.0; 
    Point poly1[15];
    Point poly2[15];
    float min_x1 = 1e9, min_y1 = 1e9, max_x1 = -1e9, max_y1 = -1e9;
    float min_x2 = 1e9, min_y2 = 1e9, max_x2 = -1e9, max_y2 = -1e9;

    // Transform t1
    float rad1 = t1.angle * 3.14159265f / 180.0f;
    float c1 = cos(rad1); float s1 = sin(rad1);
    for(int k=0; k<15; ++k) {
        Point p = base_poly[k];
        poly1[k] = {p.x*c1 - p.y*s1 + t1.x, p.x*s1 + p.y*c1 + t1.y};
        min_x1 = min(min_x1, poly1[k].x); min_y1 = min(min_y1, poly1[k].y);
        max_x1 = max(max_x1, poly1[k].x); max_y1 = max(max_y1, poly1[k].y);
    }
    
    // Transform t2
    float rad2 = t2.angle * 3.14159265f / 180.0f;
    float c2 = cos(rad2); float s2 = sin(rad2);
    for(int k=0; k<15; ++k) {
        Point p = base_poly[k];
        poly2[k] = {p.x*c2 - p.y*s2 + t2.x, p.x*s2 + p.y*c2 + t2.y};
        min_x2 = min(min_x2, poly2[k].x); min_y2 = min(min_y2, poly2[k].y);
        max_x2 = max(max_x2, poly2[k].x); max_y2 = max(max_y2, poly2[k].y);
    }

    if (max_x1 < min_x2 || max_x2 < min_x1 || max_y1 < min_y2 || max_y2 < min_y1) return false;

    // Edges
    for (int a = 0; a < 15; ++a) {
        Point p1 = poly1[a]; Point p2 = poly1[(a + 1) % 15];
        for (int b = 0; b < 15; ++b) {
            Point q1 = poly2[b]; Point q2 = poly2[(b + 1) % 15];
            if (segments_strict_intersect(p1, p2, q1, q2)) return true;
        }
    }
    
    // Points
    for (int k = 0; k < 15; ++k) {
        if (point_in_polygon(poly2[k], poly1)) return true;
        if (point_in_polygon(poly1[k], poly2)) return true;
    }
    return false;
}

kernel void batch_sa_optimize(
    device const TreeData* initial_trees [[ buffer(0) ]],
    device TreeData* final_trees [[ buffer(1) ]],
    device const int* offsets [[ buffer(2) ]],
    device const int* sizes [[ buffer(3) ]],
    constant SAParams& params [[ buffer(4) ]],
    device uint* seeds [[ buffer(5) ]],
    uint group_id [[ threadgroup_position_in_grid ]],
    uint local_id [[ thread_position_in_threadgroup ]]
) {
    int n = sizes[group_id];
    int offset = offsets[group_id];
    
    threadgroup TreeData shared_trees[256];
    threadgroup int shared_overlap_count;
    threadgroup float shared_score;
    
    // Load trees
    if (local_id < n) {
        shared_trees[local_id] = initial_trees[offset + local_id];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    uint rng_state = seeds[group_id] + local_id * 12345;
    
    // Initial overlap count (parallel reduce needed?)
    // Let's just track delta. Assume initial is valid or handle later.
    // Actually, we need to know if we are in valid state to update best.
    // For now, let's just run SA to minimize energy (Side + Overlap).
    
    // Current Side Length
    float min_x=1e9, max_x=-1e9, min_y=1e9, max_y=-1e9;
    if (local_id < n) {
        // Compute my bounds
        // Just approximate with center? No, need real bounds.
        // We can use AABB of each tree.
        TreeData t = shared_trees[local_id];
        // ... (compute bounds) ...
        // Reduce min/max across threads?
        // This is complex to do every step.
        // Approximation: Minimize Max(x) - Min(x) etc.
    }
    
    // Simplified SA:
    // Leader picks move.
    // Parallel check delta overlaps.
    // Leader evaluates.
    
    threadgroup int move_idx;
    threadgroup TreeData backup_tree;
    threadgroup int delta_overlaps;
    threadgroup float move_dx, move_dy, move_ddeg;
    threadgroup int accepted;
    
    float T = params.Tmax;
    
    for (int step = 0; step < params.nsteps; ++step) {
        // 1. Propose
        if (local_id == 0) {
            move_idx = rand_xorshift(rng_state) % n;
            move_dx = (rand_float(rng_state)*2.0 - 1.0) * params.position_delta;
            move_dy = (rand_float(rng_state)*2.0 - 1.0) * params.position_delta;
            move_ddeg = (rand_float(rng_state)*2.0 - 1.0) * params.angle_delta;
            
            backup_tree = shared_trees[move_idx];
            
            // Apply
            shared_trees[move_idx].x += move_dx;
            shared_trees[move_idx].y += move_dy;
            shared_trees[move_idx].angle += move_ddeg; // Clamp angle?
            
            delta_overlaps = 0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // 2. Parallel Check
        if (local_id < n && local_id != move_idx) {
            bool old_ov = check_overlap_pair(shared_trees[local_id], backup_tree);
            bool new_ov = check_overlap_pair(shared_trees[local_id], shared_trees[move_idx]);
            
            int d = 0;
            if (new_ov && !old_ov) d = 1;
            if (!new_ov && old_ov) d = -1;
            
            if (d != 0) {
                atomic_fetch_add_explicit((threadgroup atomic_int*)&delta_overlaps, d, memory_order_relaxed);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // 3. Eval
        if (local_id == 0) {
            // Simplified Energy: Just Overlaps?
            // We want to compact too.
            // Side length is hard to compute efficiently every step without full reduction.
            // Proxy: Minimize Distance to Center? Gravity?
            // Or just check bounds of the moved tree?
            // If moved tree expands bounds -> Penalty.
            // If moved tree shrinks bounds -> Reward.
            
            // Let's use Gravity proxy for compaction in GPU SA
            float old_dist = sqrt(backup_tree.x*backup_tree.x + backup_tree.y*backup_tree.y);
            float new_dist = sqrt(shared_trees[move_idx].x*shared_trees[move_idx].x + shared_trees[move_idx].y*shared_trees[move_idx].y);
            
            float delta_score = (new_dist - old_dist) * 0.01; // Small weight
            
            float delta_E = delta_score + (float)delta_overlaps * 10.0; // Heavy penalty for overlap
            
            bool accept = false;
            if (delta_E < 0) accept = true;
            else if (rand_float(rng_state) < exp(-delta_E / T)) accept = true;
            
            if (!accept) {
                shared_trees[move_idx] = backup_tree;
            }
            
            // Cooling
            T *= params.cooling_factor;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Write back
    if (local_id < n) {
        final_trees[offset + local_id] = shared_trees[local_id];
    }
}
