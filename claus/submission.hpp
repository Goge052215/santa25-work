#pragma once
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <limits>
#include <cstdio>
#include "tree.hpp"

namespace submission {

static inline std::string to_str_xy(long double v) {
    if (v < -100.0L) v = -100.0L;
    if (v > 100.0L) v = 100.0L;
    std::ostringstream oss;
    oss << "s" << std::setprecision(std::numeric_limits<long double>::max_digits10) << v;
    return oss.str();
}

static inline std::string to_str_deg(long double v) {
    std::ostringstream oss;
    oss << "s" << std::setprecision(std::numeric_limits<long double>::max_digits10) << v;
    return oss.str();
}

static inline void write_csv(const std::vector<std::pair<long double, std::vector<ChristmasTree>>>& solutions,
                             const std::string& path) {
    std::ofstream out(path);
    out << "id,x,y,deg\n";
    for (size_t idx = 0; idx < solutions.size(); ++idx) {
        const auto& trees = solutions[idx].second;
        size_t n = idx + 1;
        for (size_t i = 0; i < trees.size(); ++i) {
            char idbuf[32];
            std::snprintf(idbuf, sizeof(idbuf), "%03zu_%zu", n, i);
            out << idbuf << ","
                << to_str_xy(trees[i].center_x) << ","
                << to_str_xy(trees[i].center_y) << ","
                << to_str_deg(trees[i].angle_deg) << "\n";
        }
    }
}

} // namespace submission
