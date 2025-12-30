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

static inline std::vector<std::pair<long double, std::vector<ChristmasTree>>> read_csv(const std::string& path) {
    std::vector<std::pair<long double, std::vector<ChristmasTree>>> solutions(200);
    std::ifstream in(path);
    std::string line;
    std::getline(in, line); // Skip header

    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string segment;
        std::vector<std::string> parts;
        while (std::getline(ss, segment, ',')) {
            parts.push_back(segment);
        }
        if (parts.size() < 4) continue;

        // id: NNN_idx
        std::string id = parts[0];
        size_t us = id.find('_');
        int n = std::stoi(id.substr(0, us));
        int idx = std::stoi(id.substr(us + 1));

        // x, y, deg: sVALUE
        long double x = std::stold(parts[1].substr(1));
        long double y = std::stold(parts[2].substr(1));
        long double deg = std::stold(parts[3].substr(1));

        if (n >= 1 && n <= 200) {
            if (solutions[n-1].second.empty()) {
                solutions[n-1].second.resize(n);
            }
            if (idx < n) {
                solutions[n-1].second[idx] = ChristmasTree(x, y, deg);
            }
        }
    }
    return solutions;
}

} // namespace submission
