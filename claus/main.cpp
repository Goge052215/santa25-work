#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <sys/stat.h>
#include <sys/types.h>
// #include <omp.h>
#include "tree.hpp"
#include "ga.hpp"
#include "submission.hpp"

static inline std::pair<long double, std::vector<ChristmasTree>> find_best_trees(int n) {
    ga::GAParams params;
    params.pop_size = 48;
    params.generations = 120;
    params.position_delta = 0.04L;
    params.angle_delta = 3.0L;
    params.angle_delta2 = 20.0L;
    params.penalty_overlap = 1e6L;
    params.initial_spacing = 0.95L;
    params.seed = 123 + n;  // vary seed by group size
    ga::GeneticAlgorithm ga(params);
    return ga.optimize_group(n);
}

int main() {
    auto best_200 = find_best_trees(200);
    auto annealed = ga::GeneticAlgorithm::sa_refine_layout(best_200.second, 0.02L, 2.0L, 6000, 0.05L, 0.0005L, 777);
    auto compacted = ga::GeneticAlgorithm::compact_layout(annealed, 0.03L, 3.0L, 40);
    auto solutions = ga::GeneticAlgorithm::deletion_cascade_solutions(compacted.second);

    for (int n = 1; n <= 200; ++n) {
        std::cout << "Derived group " << n << " Side: " << solutions[n - 1].first << std::endl;
    }

    long double overall_score = 0.0L;
    for (int n = 1; n <= 200; ++n) {
        long double s = ga::bounding_square_side(solutions[n - 1].second);
        overall_score += (s * s) / n;
    }

    struct stat st;
    if (stat("data", &st) != 0) {
        mkdir("data", 0755);
    }
    submission::write_csv(solutions, "data/submission.csv");
    std::cout << "Overall Score: " << std::setprecision(12) << overall_score << std::endl;
    return 0;
}
