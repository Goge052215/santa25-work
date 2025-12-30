#pragma once
#include <vector>
#include <cmath>
#include <algorithm>

struct TreePoint {
    long double x;
    long double y;
};

class ChristmasTree {
public:
    static constexpr long double scale_factor = 1e15L;
    long double center_x;
    long double center_y;
    long double angle_deg;
    std::vector<TreePoint> polygon;

    ChristmasTree(long double cx = 0.0L, long double cy = 0.0L, long double angle = 0.0L)
        : center_x(cx), center_y(cy), angle_deg(angle) {
        auto p = initial_polygon();
        auto r = rotate(p, angle_deg);
        polygon = translate(r, center_x * scale_factor, center_y * scale_factor);
    }

    std::pair<TreePoint, TreePoint> aabb() const {
        if (polygon.empty()) return {{0, 0}, {0, 0}};
        long double min_x = polygon[0].x;
        long double min_y = polygon[0].y;
        long double max_x = polygon[0].x;
        long double max_y = polygon[0].y;

        for (const auto& p : polygon) {
            min_x = std::min(min_x, p.x);
            min_y = std::min(min_y, p.y);
            max_x = std::max(max_x, p.x);
            max_y = std::max(max_y, p.y);
        }
        return {{min_x, min_y}, {max_x, max_y}};
    }

    ChristmasTree(const std::vector<TreePoint>& p, long double cx, long double cy, long double angle)
        : center_x(cx), center_y(cy), angle_deg(angle), polygon(p) {}

    static std::vector<TreePoint> get_initial_polygon() {
        return initial_polygon();
    }

private:
    static std::vector<TreePoint> rotate(const std::vector<TreePoint>& poly, long double angle_deg) {
        long double rad = angle_deg * (acosl(-1.0L) / 180.0L);
        long double c = std::cos(rad);
        long double s = std::sin(rad);
        std::vector<TreePoint> out;
        out.reserve(poly.size());
        for (const auto& pt : poly) {
            long double x = pt.x * c - pt.y * s;
            long double y = pt.x * s + pt.y * c;
            out.push_back({x, y});
        }
        return out;
    }

    static std::vector<TreePoint> translate(const std::vector<TreePoint>& poly, long double xoff, long double yoff) {
        std::vector<TreePoint> out;
        out.reserve(poly.size());
        for (const auto& pt : poly) {
            out.push_back({pt.x + xoff, pt.y + yoff});
        }
        return out;
    }

    static std::vector<TreePoint> initial_polygon() {
        long double trunk_w = 0.15L;
        long double trunk_h = 0.2L;
        long double base_w = 0.7L;
        long double mid_w = 0.4L;
        long double top_w = 0.25L;
        long double tip_y = 0.8L;
        long double tier_1_y = 0.5L;
        long double tier_2_y = 0.25L;
        long double base_y = 0.0L;
        long double trunk_bottom_y = -trunk_h;

        auto sf = scale_factor;
        // Vertices ordered to form the polygon
        std::vector<TreePoint> pts = {
            {0.0L * sf, tip_y * sf},
            {(top_w / 2.0L) * sf, tier_1_y * sf},
            {(top_w / 4.0L) * sf, tier_1_y * sf},
            {(mid_w / 2.0L) * sf, tier_2_y * sf},
            {(mid_w / 4.0L) * sf, tier_2_y * sf},
            {(base_w / 2.0L) * sf, base_y * sf},
            {(trunk_w / 2.0L) * sf, base_y * sf},
            {(trunk_w / 2.0L) * sf, trunk_bottom_y * sf},
            {-(trunk_w / 2.0L) * sf, trunk_bottom_y * sf},
            {-(trunk_w / 2.0L) * sf, base_y * sf},
            {-(base_w / 2.0L) * sf, base_y * sf},
            {-(mid_w / 4.0L) * sf, tier_2_y * sf},
            {-(mid_w / 2.0L) * sf, tier_2_y * sf},
            {-(top_w / 4.0L) * sf, tier_1_y * sf},
            {-(top_w / 2.0L) * sf, tier_1_y * sf}
        };

        return pts;
    }
};
