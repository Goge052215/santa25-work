#include "gpu_context.hpp"
#import <Metal/Metal.h>
#include <iostream>

struct TreeData {
    float x;
    float y;
    float angle;
    float padding; // Align to 16 bytes for Metal
};

struct PhysicsParams {
    float repulsion_strength;
    float gravity_strength;
    float learning_rate;
    float buffer_val;
};

class GpuContextImpl {
public:
    id<MTLDevice> device;
    id<MTLCommandQueue> commandQueue;
    id<MTLComputePipelineState> pipelineState;
    id<MTLComputePipelineState> pipelineStateShared;
    id<MTLComputePipelineState> pipelineStatePhysics; // New physics kernel
    bool valid;
    
    // Cached resources to avoid reallocation
    id<MTLBuffer> cachedTreeBuffer;
    id<MTLBuffer> cachedTreeBufferOut; // Double buffering
    id<MTLBuffer> cachedResultBuffer;
    id<MTLBuffer> cachedBufferValBuffer; 
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
        } else {
             std::cerr << "Warning: 'check_overlaps_shared' kernel not found." << std::endl;
        }
        
        id<MTLFunction> kernelPhysics = [library newFunctionWithName:@"physics_step"];
        if (kernelPhysics) {
            pipelineStatePhysics = [device newComputePipelineStateWithFunction:kernelPhysics error:&error];
            if (!pipelineStatePhysics) {
                std::cerr << "Failed to create physics pipeline state: " << [[error localizedDescription] UTF8String] << std::endl;
            }
        } else {
             std::cerr << "Warning: 'physics_step' kernel not found." << std::endl;
        }
        
        valid = true;
    }
    
    bool compute(const std::vector<ChristmasTree>& trees, float buffer_val) {
        if (!valid) return false;
        @autoreleasepool {
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
                cachedTreeBufferOut = [
                    device newBufferWithLength:cachedTreeCapacity * 
                    sizeof(TreeData) options:MTLResourceStorageModeShared
                ];
                cachedResultBuffer = [
                    device newBufferWithLength:sizeof(int) options:MTLResourceStorageModeShared
                ];
            }
            
            // Copy data to shared buffer
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
                
                MTLSize gridSize = MTLSizeMake(n, 1, 1);
                MTLSize threadgroupSize = MTLSizeMake(n, 1, 1);
                
                NSUInteger maxThreads = pipelineStateShared.maxTotalThreadsPerThreadgroup;
                if (n > maxThreads) {
                    useShared = false;
                } else {
                     [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
                }
            }
            
            if (!useShared) {
                [encoder setComputePipelineState:pipelineState];
                [encoder setBuffer:cachedTreeBuffer offset:0 atIndex:0];
                [encoder setBuffer:cachedResultBuffer offset:0 atIndex:1];
                int n_val = (int)n;
                [encoder setBytes:&n_val length:sizeof(int) atIndex:2];
                [encoder setBytes:&buffer_val length:sizeof(float) atIndex:3];
                
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
    }
    
    std::vector<ChristmasTree> physics_polish(const std::vector<ChristmasTree>& trees, int steps, double initial_lr) {
        if (!valid || !pipelineStatePhysics) return trees;
        @autoreleasepool {
            size_t n = trees.size();
            if (n < 2) return trees;
            
            NSUInteger dataSize = n * sizeof(TreeData);
            
            // Resize buffers
            if (n > cachedTreeCapacity || cachedTreeBuffer == nil) {
                cachedTreeCapacity = n * 2;
                if (cachedTreeCapacity < 256) cachedTreeCapacity = 256;
                
                cachedTreeBuffer = [device newBufferWithLength:cachedTreeCapacity * sizeof(TreeData) options:MTLResourceStorageModeShared];
                cachedTreeBufferOut = [device newBufferWithLength:cachedTreeCapacity * sizeof(TreeData) options:MTLResourceStorageModeShared];
            }
            
            // Upload
            std::vector<TreeData> data(n);
            for (size_t i = 0; i < n; ++i) {
                data[i].x = (float)trees[i].center_x;
                data[i].y = (float)trees[i].center_y;
                data[i].angle = (float)trees[i].angle_deg;
            }
            memcpy(cachedTreeBuffer.contents, data.data(), dataSize);
            
            // Parameters
            PhysicsParams params;
            params.repulsion_strength = 1.0f;
            params.gravity_strength = 0.001f;
            params.learning_rate = (float)initial_lr;
            params.buffer_val = 0.0f;
            
            float decay = 0.999f;
            int n_val = (int)n;
            
            id<MTLBuffer> bufferIn = cachedTreeBuffer;
            id<MTLBuffer> bufferOut = cachedTreeBufferOut;
            
            id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
            [encoder setComputePipelineState:pipelineStatePhysics];
            
            MTLSize gridSize = MTLSizeMake(n, 1, 1);
            NSUInteger w = pipelineStatePhysics.maxTotalThreadsPerThreadgroup;
            if (w > n) w = n;
            MTLSize threadgroupSize = MTLSizeMake(w, 1, 1);
            
            for (int s = 0; s < steps; ++s) {
                [encoder setBuffer:bufferIn offset:0 atIndex:0];
                [encoder setBuffer:bufferOut offset:0 atIndex:1];
                [encoder setBytes:&n_val length:sizeof(int) atIndex:2];
                [encoder setBytes:&params length:sizeof(PhysicsParams) atIndex:3];
                
                [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
                
                // Swap
                id<MTLBuffer> tmp = bufferIn;
                bufferIn = bufferOut;
                bufferOut = tmp;
                
                // Decay
                params.learning_rate *= decay;
            }
            
            [encoder endEncoding];
            [commandBuffer commit];
            [commandBuffer waitUntilCompleted];
            
            TreeData* resultData = (TreeData*)bufferIn.contents;
            std::vector<ChristmasTree> result = trees;
            for (size_t i = 0; i < n; ++i) {
                result[i].center_x = resultData[i].x;
                result[i].center_y = resultData[i].y;
                // Angle is constant in this physics model
            }
            
            return result;
        }
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

std::vector<ChristmasTree> GpuContext::physics_polish(const std::vector<ChristmasTree>& trees, int steps, double initial_lr) {
    GpuContextImpl* p = (GpuContextImpl*)impl;
    if (p->valid) {
        return p->physics_polish(trees, steps, initial_lr);
    }
    return trees;
}

bool GpuContext::is_valid() {
    return ((GpuContextImpl*)impl)->valid;
}
