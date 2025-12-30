#pragma once
#include <vector>
#include <cmath>
#include <omp.h>
#include <Accelerate/Accelerate.h>
#include "tree.hpp"

namespace grid {

// Thread-local buffers to reuse memory
struct Workspace {
    std::vector<double> cx;
    std::vector<double> cy;
    std::vector<double> angle;
    std::vector<double> angle_rad;
    std::vector<double> sin_val;
    std::vector<double> cos_val;
    std::vector<double> base_x;
    std::vector<double> base_y;
    std::vector<double> rot_x;
    std::vector<double> rot_y;
    std::vector<double> final_x;
    std::vector<double> final_y;
};

inline std::vector<ChristmasTree> create_grid_trees(
    const std::vector<long double>& seed_xs,
    const std::vector<long double>& seed_ys,
    const std::vector<long double>& seed_degs,
    long double a, long double b,
    int ncols, int nrows,
    bool append_x, bool append_y,
    long double row_phase_x, long double col_phase_y,
    long double shear_x, long double shear_y,
    long double parity_row_deg, long double parity_col_deg
) {
    size_t n_seeds = seed_xs.size();
    
    // Calculate total number of trees
    size_t base_size = n_seeds * ncols * nrows;
    size_t append_x_size = (append_x && n_seeds > 1) ? nrows : 0;
    size_t append_y_size = (append_y && n_seeds > 1) ? ncols : 0;
    size_t total_size = base_size + append_x_size + append_y_size;

    std::vector<ChristmasTree> trees(total_size);
    static thread_local Workspace ws;

    // Resize workspace if needed
    // 15 points per tree
    size_t n_pts = 15;
    size_t total_pts = total_size * n_pts;
    
    if (ws.cx.size() < total_size) {
        ws.cx.resize(total_size);
        ws.cy.resize(total_size);
        ws.angle.resize(total_size);
        ws.angle_rad.resize(total_size);
        ws.sin_val.resize(total_size);
        ws.cos_val.resize(total_size);
    }
    
    // Fill centers and angles
    // This loop is fast enough in scalar, but could be vectorized.
    // For now, simple loop to fill vectors.
    
    // Base grid
    #pragma omp parallel for
    for (size_t i = 0; i < base_size; ++i) {
        size_t s = i / (ncols * nrows);
        size_t rem = i % (ncols * nrows);
        int col = rem / nrows;
        int row = rem % nrows;

        ws.cx[i] = (double)(seed_xs[s] + col * a + (row % 2) * row_phase_x + shear_x * row);
        ws.cy[i] = (double)(seed_ys[s] + row * b + (col % 2) * col_phase_y + shear_y * col);
        ws.angle[i] = (double)(seed_degs[s] + (row % 2) * parity_row_deg + (col % 2) * parity_col_deg);
    }

    // Append X
    if (append_x_size > 0) {
        #pragma omp parallel for
        for (int row = 0; row < nrows; ++row) {
            size_t idx = base_size + row;
            ws.cx[idx] = (double)(seed_xs[1] + ncols * a + (row % 2) * row_phase_x + shear_x * row);
            ws.cy[idx] = (double)(seed_ys[1] + row * b + (ncols % 2) * col_phase_y + shear_y * ncols);
            ws.angle[idx] = (double)(seed_degs[1] + (row % 2) * parity_row_deg + (ncols % 2) * parity_col_deg);
        }
    }

    // Append Y
    if (append_y_size > 0) {
        #pragma omp parallel for
        for (int col = 0; col < ncols; ++col) {
            size_t idx = base_size + append_x_size + col;
            ws.cx[idx] = (double)(seed_xs[1] + col * a + (nrows % 2) * row_phase_x + shear_x * nrows);
            ws.cy[idx] = (double)(seed_ys[1] + nrows * b + (col % 2) * col_phase_y + shear_y * col);
            ws.angle[idx] = (double)(seed_degs[1] + (nrows % 2) * parity_row_deg + (col % 2) * parity_col_deg);
        }
    }

    // Convert angles to radians
    double deg2rad = M_PI / 180.0;
    vDSP_vsmulD(ws.angle.data(), 1, &deg2rad, ws.angle_rad.data(), 1, total_size);

    // Compute sin/cos
    int n = (int)total_size;
    vvsin(ws.sin_val.data(), ws.angle_rad.data(), &n);
    vvcos(ws.cos_val.data(), ws.angle_rad.data(), &n);

    // Now construct polygons
    // We can't easily vectorize across different polygons because they are separate objects.
    // But we can do it tree by tree or batch it.
    // Batching requires expanding the vectors to (total_size * 15).
    
    // Let's try batching for vDSP efficiency.
    if (ws.base_x.size() < total_pts) {
        ws.base_x.resize(total_pts);
        ws.base_y.resize(total_pts);
        ws.rot_x.resize(total_pts);
        ws.rot_y.resize(total_pts);
        ws.final_x.resize(total_pts);
        ws.final_y.resize(total_pts);
    }

    // Get base polygon
    auto base_poly = ChristmasTree::get_initial_polygon();
    // Fill base_x/y repeated
    // We can use vDSP to replicate? No, just loop or memcpy.
    // Or simpler: construct polygon per tree using the sin/cos we computed.
    // Batching 15 points per tree is small for vDSP but might be okay.
    // Let's do it per tree but manually vectorizing the 15 points? 
    // No, 15 is too small for vDSP overhead.
    
    // BUT we can use the `ws.sin_val` and `ws.cos_val` we computed.
    
    double sf = (double)ChristmasTree::scale_factor;

    // Use OpenMP to construct trees
    #pragma omp parallel for
    for (size_t i = 0; i < total_size; ++i) {
        double c = ws.cos_val[i];
        double s = ws.sin_val[i];
        double cx = ws.cx[i] * sf;
        double cy = ws.cy[i] * sf;
        
        std::vector<TreePoint> poly(15);
        for (int k = 0; k < 15; ++k) {
            double px = (double)base_poly[k].x;
            double py = (double)base_poly[k].y;
            
            // Rotate
            double rx = px * c - py * s;
            double ry = px * s + py * c;
            
            // Translate
            poly[k].x = (long double)(rx + cx);
            poly[k].y = (long double)(ry + cy);
        }
        
        trees[i] = ChristmasTree(poly, (long double)ws.cx[i], (long double)ws.cy[i], (long double)ws.angle[i]);
    }

    return trees;
}

} // namespace grid
