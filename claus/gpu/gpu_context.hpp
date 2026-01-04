#pragma once
#include <vector>
#include "../tree.hpp"

class GpuContext {
public:
    static GpuContext& getInstance();
    bool is_valid();
    bool has_overlap(const std::vector<ChristmasTree>& trees, double buffer = 0.0);
    std::vector<ChristmasTree> physics_polish(const std::vector<ChristmasTree>& trees, int steps = 1000, double initial_lr = 0.01);

private:
    GpuContext();
    ~GpuContext();
    void* impl; // Pimpl idiom to hide Objective-C types
};
