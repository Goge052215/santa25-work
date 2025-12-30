#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
#include "tree.hpp"

namespace overlap {

struct DPoint {
    double x;
    double y;
};

// Convert tree polygon to unit-scale doubles for geometric checks
static inline std::vector<DPoint> to_unit_poly(const std::vector<Point>& poly) {
    std::vector<DPoint> out;
    out.reserve(poly.size());
    double sf = static_cast<double>(ChristmasTree::scale_factor);
    for (const auto& p : poly) {
        out.push_back({static_cast<double>(p.x) / sf, static_cast<double>(p.y) / sf});
    }
    return out;
}

static inline double cross(const DPoint& a, const DPoint& b) {
    return a.x * b.y - a.y * b.x;
}

static inline DPoint sub(const DPoint& a, const DPoint& b) {
    return {a.x - b.x, a.y - b.y};
}

// Check if point p lies on segment ab (with epsilon tolerance)
static inline bool on_segment(const DPoint& a, const DPoint& b, const DPoint& p, double eps) {
    double minx = std::min(a.x, b.x) - eps;
    double maxx = std::max(a.x, b.x) + eps;
    double miny = std::min(a.y, b.y) - eps;
    double maxy = std::max(a.y, b.y) + eps;
    double c = std::fabs(cross(sub(b, a), sub(p, a)));
    return c <= eps && p.x >= minx && p.x <= maxx && p.y >= miny && p.y <= maxy;
}

// Strict segment intersection: returns true only if segments cross strictly (excluding endpoints)
static inline bool segments_strict_intersect(const DPoint& p1, const DPoint& p2, const DPoint& q1, const DPoint& q2, double eps) {
    DPoint r = sub(p2, p1);
    DPoint s = sub(q2, q1);
    double d = cross(r, s);
    double o1 = cross(sub(p2, p1), sub(q1, p1));
    double o2 = cross(sub(p2, p1), sub(q2, p1));

    if (std::fabs(d) < eps) {
        if (std::fabs(o1) > eps || std::fabs(o2) > eps) return false;
        double rr = r.x * r.x + r.y * r.y;
        if (rr < eps) return false;
        double t0 = ((q1.x - p1.x) * r.x + (q1.y - p1.y) * r.y) / rr;
        double t1 = ((q2.x - p1.x) * r.x + (q2.y - p1.y) * r.y) / rr;
        double smin = std::min(t0, t1);
        double smax = std::max(t0, t1);
        double overlap_len = std::min(1.0, smax) - std::max(0.0, smin);
        return overlap_len > eps;
    }

    double t = cross(sub(q1, p1), s) / d;
    double u = cross(sub(q1, p1), r) / d;
    return t > eps && t < 1.0 - eps && u > eps && u < 1.0 - eps;
}

// Strict point-in-polygon: returns true if point is strictly inside
static inline bool point_in_polygon_strict(const std::vector<DPoint>& poly, const DPoint& p, double eps) {
    int wn = 0;
    size_t n = poly.size();
    for (size_t i = 0; i < n; ++i) {
        const DPoint& a = poly[i];
        const DPoint& b = poly[(i + 1) % n];
        // If on boundary, not strictly inside
        if (on_segment(a, b, p, eps)) return false;
        
        bool cond = ((a.y <= p.y) && (b.y > p.y)) || ((a.y > p.y) && (b.y <= p.y));
        if (cond) {
            double x_intersect = a.x + (p.y - a.y) * (b.x - a.x) / (b.y - a.y);
            if (x_intersect > p.x) wn += (b.y > a.y) ? 1 : -1;
        }
    }
    return wn != 0;
}

// Main overlap check: mirrors validate_overlap.py logic (strict intersection + point-in-poly)
static inline bool polygons_strict_overlap(const std::vector<Point>& A, const std::vector<Point>& B) {
    auto Ad = to_unit_poly(A);
    auto Bd = to_unit_poly(B);
    double eps = 1e-12;

    size_t na = Ad.size(), nb = Bd.size();
    
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

static inline bool boxes_overlap(const std::pair<Point, Point>& a, const std::pair<Point, Point>& b) {
    return !(
        a.second.x < b.first.x ||
        b.second.x < a.first.x ||
        a.second.y < b.first.y ||
        b.second.y < a.first.y
    );
}

} // namespace overlap
