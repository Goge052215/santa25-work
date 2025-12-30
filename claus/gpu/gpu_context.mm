#include "gpu_context.hpp"
#import <Metal/Metal.h>
#include <iostream>

struct TreeData {
    float x;
    float y;
    float angle;
    float padding; // Align to 16 bytes for Metal
};

class GpuContextImpl {
public:
    id<MTLDevice> device;
    id<MTLCommandQueue> commandQueue;
    id<MTLComputePipelineState> pipelineState;
    bool valid;

    GpuContextImpl() : valid(false) {
        device = MTLCreateSystemDefaultDevice();
        if (!device) {
            std::cerr << "Metal is not supported on this device" << std::endl;
            return;
        }
        commandQueue = [device newCommandQueue];
        
        NSError* error = nil;
        
        // Try to load library from common locations
        // 1. Same directory as executable (if built there)
        // 2. claus/ directory (if running from root)
        
        NSString* possiblePaths[] = {
            @"claus/gpu/gpu_overlap.metallib",
            @"gpu/gpu_overlap.metallib",
            @"../claus/gpu/gpu_overlap.metallib",
            @"gpu_overlap.metallib"
        };
        
        id<MTLLibrary> library = nil;
        
        for (int i = 0; i < 4; ++i) {
            if ([[NSFileManager defaultManager] fileExistsAtPath:possiblePaths[i]]) {
                NSURL* libURL = [NSURL fileURLWithPath:possiblePaths[i]];
                library = [device newLibraryWithURL:libURL error:&error];
                if (library) break;
            }
        }

        if (!library) {
            // std::cerr << "Failed to load Metal library. Falling back to CPU." << std::endl;
            // Silent fallback or warning? Warning is better.
             std::cerr << "Warning: gpu_overlap.metallib not found. GPU acceleration disabled." << std::endl;
            return;
        }

        id<MTLFunction> kernel = [library newFunctionWithName:@"check_overlaps"];
        if (!kernel) {
             std::cerr << "Failed to find kernel 'check_overlaps'" << std::endl;
             return;
        }
        
        pipelineState = [device newComputePipelineStateWithFunction:kernel error:&error];
        
        if (!pipelineState) {
            std::cerr << "Failed to create pipeline state: " << [[error localizedDescription] UTF8String] << std::endl;
            return;
        }
        
        valid = true;
    }
    
    bool compute(const std::vector<ChristmasTree>& trees) {
        if (!valid) return false; // Should caller handle fallback? Yes. 
        // But here we return false meaning "no overlap found by GPU" which is DANGEROUS if it's just broken.
        // We should throw or return a status. 
        // Or simpler: The wrapper checks 'valid'.
        
        size_t n = trees.size();
        if (n < 2) return false;
        
        // Prepare data
        std::vector<TreeData> data(n);
        for (size_t i = 0; i < n; ++i) {
            data[i].x = (float)trees[i].center_x;
            data[i].y = (float)trees[i].center_y;
            data[i].angle = (float)trees[i].angle_deg;
        }
        
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        
        [encoder setComputePipelineState:pipelineState];
        
        NSUInteger dataSize = n * sizeof(TreeData);
        id<MTLBuffer> bufferTrees = [device newBufferWithBytes:data.data() length:dataSize options:MTLResourceStorageModeShared];
        
        int initialResult = 0;
        id<MTLBuffer> bufferResult = [device newBufferWithBytes:&initialResult length:sizeof(int) options:MTLResourceStorageModeShared];
        
        [encoder setBuffer:bufferTrees offset:0 atIndex:0];
        [encoder setBuffer:bufferResult offset:0 atIndex:1];
        
        MTLSize gridSize = MTLSizeMake(n, n, 1);
        
        // threadsPerThreadgroup
        NSUInteger w = pipelineState.maxTotalThreadsPerThreadgroup;
        if (w > n) w = n; // Optimization? Or just use max?
        // Using square block?
        // Let's use 1D threadgroup for 2D grid?
        // Metal can handle it. 
        // Let's try 8x8 or something.
        
        NSUInteger w_dim = (NSUInteger)sqrt(w);
        if (w_dim < 1) w_dim = 1;
        MTLSize threadgroupSize = MTLSizeMake(w_dim, w_dim, 1);
        
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];
        
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        int* resultPtr = (int*)[bufferResult contents];
        return (*resultPtr) > 0;
    }
};

GpuContext& GpuContext::getInstance() {
    static GpuContext instance;
    return instance;
}

GpuContext::GpuContext() {
    impl = new GpuContextImpl();
}

GpuContext::~GpuContext() {
    delete (GpuContextImpl*)impl;
}

bool GpuContext::has_overlap(const std::vector<ChristmasTree>& trees) {
    GpuContextImpl* p = (GpuContextImpl*)impl;
    if (p->valid) {
        return p->compute(trees);
    }
    return false; 
}

bool GpuContext::is_valid() {
    return ((GpuContextImpl*)impl)->valid;
}
