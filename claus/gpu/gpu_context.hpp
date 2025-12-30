#pragma once
#include <vector>
#include "../tree.hpp"

class GpuContext {
public:
    static GpuContext& getInstance();
    bool is_valid();
    bool has_overlap(const std::vector<ChristmasTree>& trees);

private:
    GpuContext();
    ~GpuContext();
    void* impl; // Pimpl idiom to hide Objective-C types
};
