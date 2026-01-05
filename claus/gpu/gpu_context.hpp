#pragma once
#include <vector>
#include "../tree.hpp"

struct SAParamsGPU {
    float Tmax;
    float Tmin;
    float cooling_factor;
    int nsteps;
    float position_delta;
    float angle_delta;
};

class GpuContext {
public:
    static GpuContext& getInstance();
    bool is_valid();
    bool has_overlap(const std::vector<ChristmasTree>& trees, double buffer = 0.0);
    std::vector<ChristmasTree> physics_polish(const std::vector<ChristmasTree>& trees, int steps = 1000, double initial_lr = 0.01);
    
    // Batch SA optimization
    // Input: Vector of solutions (each solution is a vector of trees)
    // Returns: Optimized solutions
    std::vector<std::vector<ChristmasTree>> batch_sa_optimize(
        const std::vector<std::vector<ChristmasTree>>& solutions,
        const SAParamsGPU& params
    );

    // Check overlaps for a batch of candidates against a fixed set of trees
    // Returns: vector of booleans (true = overlap/invalid, false = valid)
    std::vector<int> check_candidates_overlap(
        const std::vector<ChristmasTree>& fixed_trees,
        const std::vector<ChristmasTree>& candidates,
        float buffer = 0.0f
    );

private:
    GpuContext();
    ~GpuContext();
    void* impl; // Pimpl idiom to hide Objective-C types
};
