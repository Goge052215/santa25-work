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

// Tree definition constants
// constant float trunk_w = 0.15;
// constant float trunk_h = 0.2;
// constant float base_w = 0.7;
// constant float mid_w = 0.4;
// constant float top_w = 0.25;
// constant float tip_y = 0.8;
// constant float tier_1_y = 0.5;
// constant float tier_2_y = 0.25;
// constant float base_y = 0.0;
// constant float trunk_bottom_y = -0.2;

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
    float eps = 1e-12; // Tighter epsilon to match CPU logic as closely as possible
    Point r = sub(p2, p1);
    Point s = sub(q2, q1);
    float d = cross_prod(r, s);
    float o1 = cross_prod(sub(p2, p1), sub(q1, p1));
    float o2 = cross_prod(sub(p2, p1), sub(q2, p1));

    if (abs(d) < eps) {
        if (abs(o1) > eps || abs(o2) > eps) return false;
        float rr = r.x * r.x + r.y * r.y;
        if (rr < eps) return false;
        float t0 = ((q1.x - p1.x) * r.x + (q1.y - p1.y) * r.y) / rr;
        float t1 = ((q2.x - p1.x) * r.x + (q2.y - p1.y) * r.y) / rr;
        float smin = min(t0, t1);
        float smax = max(t0, t1);
        float overlap_len = min(1.0f, smax) - max(0.0f, smin);
        return overlap_len > eps;
    }

    float t = cross_prod(sub(q1, p1), s) / d;
    float u = cross_prod(sub(q1, p1), r) / d;
    return t > eps && t < 1.0 - eps && u > eps && u < 1.0 - eps;
}

bool point_in_polygon(Point p, Point poly[15]) {
    int wn = 0;
    float eps = 1e-12;
    for (int i = 0; i < 15; ++i) {
        Point a = poly[i];
        Point b = poly[(i + 1) % 15];
        
        // On segment check
        if (abs(cross_prod(sub(b, a), sub(p, a))) <= eps) {
            float minx = min(a.x, b.x) - eps;
            float maxx = max(a.x, b.x) + eps;
            float miny = min(a.y, b.y) - eps;
            float maxy = max(a.y, b.y) + eps;
            if (p.x >= minx && p.x <= maxx && p.y >= miny && p.y <= maxy) return false; // On boundary is not strictly inside
        }
        
        bool cond = ((a.y <= p.y) && (b.y > p.y)) || ((a.y > p.y) && (b.y <= p.y));
        if (cond) {
            float x_intersect = a.x + (p.y - a.y) * (b.x - a.x) / (b.y - a.y);
            if (x_intersect > p.x) wn += (b.y > a.y) ? 1 : -1;
        }
    }
    return wn != 0;
}

kernel void check_overlaps(
    device const TreeData* trees [[ buffer(0) ]],
    device atomic_int* result [[ buffer(1) ]],
    constant float& buffer_val [[ buffer(2) ]],
    uint2 id [[ thread_position_in_grid ]],
    uint2 size [[ threads_per_grid ]]
) {
    // We map 2D grid to pairs?
    // Let's assume 1D grid for simplicity: index i
    // Or 2D grid (i, j)
    
    uint i = id.x;
    uint j = id.y;
    
    if (i >= j) return; // Only check upper triangle
    // We need to pass N? Assuming size covers it.
    
    // Check if result is already found
    if (atomic_load_explicit(result, memory_order_relaxed) > 0) return;

    TreeData t1 = trees[i];
    TreeData t2 = trees[j];

    float scale = 1.0 + buffer_val;

    // Bounding box check first
    // Need to compute transformed vertices to get BB
    // This duplicates work but avoids storing vertices in memory
    
    Point poly1[15];
    Point poly2[15];
    
    // Transform Tree 1
    float rad1 = t1.angle * 3.14159265 / 180.0;
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
    
    // Transform Tree 2
    float rad2 = t2.angle * 3.14159265 / 180.0;
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
    
    // AABB Check
    if (max_x1 < min_x2 || max_x2 < min_x1 || max_y1 < min_y2 || max_y2 < min_y1) return;
    
    // Polygon Check
    // 1. Edges
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
    
    // 2. Point in Poly
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
    // Shared memory storage
    // Max 240 trees supported in this kernel (fits in 32KB threadgroup memory)
    threadgroup SharedPoly shared_polys[240];

    if (tid >= block_size) return;

    // 1. Transform and store in shared memory
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

    // 2. Check overlaps
    // Each thread checks itself against all j > tid
    
    // Quick exit if result found
    if (atomic_load_explicit(result, memory_order_relaxed) > 0) return;
    
    // Cache my poly in registers to avoid repeated shared mem reads?
    // Actually, reading from shared mem is fast.
    // Reading my own poly from shared mem is fine.
    
    float my_min_x = shared_polys[tid].min_x;
    float my_min_y = shared_polys[tid].min_y;
    float my_max_x = shared_polys[tid].max_x;
    float my_max_y = shared_polys[tid].max_y;
    
    // Loop
    for (uint j = tid + 1; j < block_size; ++j) {
        // AABB Check
        float other_min_x = shared_polys[j].min_x;
        float other_min_y = shared_polys[j].min_y;
        float other_max_x = shared_polys[j].max_x;
        float other_max_y = shared_polys[j].max_y;
        
        if (my_max_x < other_min_x || other_max_x < my_min_x || 
            my_max_y < other_min_y || other_max_y < my_min_y) continue;
            
        // Strict Polygon Check
        // Access shared memory arrays
        // Note: passing array from shared mem to function might be tricky in Metal
        // Better to inline or modify helper to take pointers
        
        // Edge check
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
        
        // Point in poly check
        // Inline simple check to avoid pointer issues
        
        // Check j points in i
        for (int k = 0; k < 15; ++k) {
            Point p = shared_polys[j].points[k];
            // Point in Poly I
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
                         wn = 0; break; // Boundary
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
        
        // Check i points in j
        for (int k = 0; k < 15; ++k) {
            Point p = shared_polys[tid].points[k];
            // Point in Poly J
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
                         wn = 0; break; // Boundary
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
