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
    id<MTLComputePipelineState> pipelineStateShared; // New optimized kernel
    bool valid;
    
    // Cached resources to avoid reallocation
    id<MTLBuffer> cachedTreeBuffer;
    id<MTLBuffer> cachedResultBuffer;
    id<MTLBuffer> cachedBufferValBuffer; // For buffer value
    NSUInteger cachedTreeCapacity;

    GpuContextImpl() : valid(false), cachedTreeCapacity(0) {
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

        id<MTLFunction> kernelShared = [library newFunctionWithName:@"check_overlaps_shared"];
        if (kernelShared) {
            pipelineStateShared = [device newComputePipelineStateWithFunction:kernelShared error:&error];
            if (!pipelineStateShared) {
                std::cerr << "Failed to create shared pipeline state: " << [[error localizedDescription] UTF8String] << std::endl;
                // It's okay, we can fallback to standard pipeline
            }
        } else {
             std::cerr << "Warning: 'check_overlaps_shared' kernel not found." << std::endl;
        }
        
        valid = true;
    }
    
    bool compute(const std::vector<ChristmasTree>& trees, float buffer_val) {
        if (!valid) return false;
        
        size_t n = trees.size();
        if (n < 2) return false;
        
        // Decide which kernel to use
        bool useShared = (pipelineStateShared != nil && n <= 240);

        NSUInteger dataSize = n * sizeof(TreeData);
        
        // Resize buffers if needed
        if (n > cachedTreeCapacity || cachedTreeBuffer == nil) {
            cachedTreeCapacity = n * 2; // Growth factor
            if (cachedTreeCapacity < 256) cachedTreeCapacity = 256; // Min size
            
            cachedTreeBuffer = [
                device newBufferWithLength:cachedTreeCapacity * 
                sizeof(TreeData) options:MTLResourceStorageModeShared
            ];
            cachedResultBuffer = [
                device newBufferWithLength:sizeof(int) options:MTLResourceStorageModeShared
            ];
        }
        
        // Copy data to shared buffer
        // Prepare data vector
        std::vector<TreeData> data(n);
        for (size_t i = 0; i < n; ++i) {
            data[i].x = (float)trees[i].center_x;
            data[i].y = (float)trees[i].center_y;
            data[i].angle = (float)trees[i].angle_deg;
        }
        memcpy(cachedTreeBuffer.contents, data.data(), dataSize);
        
        // Reset result
        int initialResult = 0;
        memcpy(cachedResultBuffer.contents, &initialResult, sizeof(int));

        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        
        if (useShared) {
            [encoder setComputePipelineState:pipelineStateShared];
            [encoder setBuffer:cachedTreeBuffer offset:0 atIndex:0];
            [encoder setBuffer:cachedResultBuffer offset:0 atIndex:1];
            [encoder setBytes:&buffer_val length:sizeof(float) atIndex:2];
            
            // 1D grid, 1 threadgroup
            MTLSize gridSize = MTLSizeMake(n, 1, 1);
            MTLSize threadgroupSize = MTLSizeMake(n, 1, 1);
            
            // Metal requires threadgroup size to be valid
            NSUInteger maxThreads = pipelineStateShared.maxTotalThreadsPerThreadgroup;
            if (n > maxThreads) {
                // Fallback to standard if n exceeds hardware threadgroup limit
                useShared = false;
            } else {
                 [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
            }
        }
        
        if (!useShared) {
            [encoder setComputePipelineState:pipelineState];
            [encoder setBuffer:cachedTreeBuffer offset:0 atIndex:0];
            [encoder setBuffer:cachedResultBuffer offset:0 atIndex:1];
            [encoder setBytes:&buffer_val length:sizeof(float) atIndex:2];
            
            MTLSize gridSize = MTLSizeMake(n, n, 1);
            
            NSUInteger w = pipelineState.maxTotalThreadsPerThreadgroup;
            if (w > n) w = n;
            NSUInteger w_dim = (NSUInteger)sqrt(w);
            if (w_dim < 1) w_dim = 1;
            MTLSize threadgroupSize = MTLSizeMake(w_dim, w_dim, 1);
            
            [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
        }
        
        [encoder endEncoding];
        
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        int* resultPtr = (int*)[cachedResultBuffer contents];
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

bool GpuContext::has_overlap(const std::vector<ChristmasTree>& trees, double buffer) {
    GpuContextImpl* p = (GpuContextImpl*)impl;
    if (p->valid) {
        return p->compute(trees, (float)buffer);
    }
    return false; 
}

bool GpuContext::is_valid() {
    return ((GpuContextImpl*)impl)->valid;
}
