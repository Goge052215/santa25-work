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
    id<MTLComputePipelineState> pipelineStatePhysics; 
    id<MTLComputePipelineState> pipelineStateSA; // New SA kernel
    id<MTLComputePipelineState> pipelineStateCandidates;
    bool valid;
    
    // Cached resources to avoid reallocation
    id<MTLBuffer> cachedTreeBuffer;
    id<MTLBuffer> cachedTreeBufferOut; 
    id<MTLBuffer> cachedResultBuffer;
    NSUInteger cachedTreeCapacity;

    GpuContextImpl() : valid(false), cachedTreeCapacity(0) {
        device = MTLCreateSystemDefaultDevice();
        if (!device) {
            std::cerr << "Metal is not supported on this device" << std::endl;
            return;
        }
        commandQueue = [device newCommandQueue];
        
        NSError* error = nil;
        
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
        if (kernel) {
             pipelineState = [device newComputePipelineStateWithFunction:kernel error:&error];
        } else {
             std::cerr << "Failed to find kernel 'check_overlaps'" << std::endl;
             return;
        }

        id<MTLFunction> kernelShared = [library newFunctionWithName:@"check_overlaps_shared"];
        if (kernelShared) {
            pipelineStateShared = [device newComputePipelineStateWithFunction:kernelShared error:&error];
        }
        
        id<MTLFunction> kernelPhysics = [library newFunctionWithName:@"physics_step"];
        if (kernelPhysics) {
            pipelineStatePhysics = [device newComputePipelineStateWithFunction:kernelPhysics error:&error];
        }

        id<MTLFunction> kernelSA = [library newFunctionWithName:@"batch_sa_optimize"];
        if (kernelSA) {
            pipelineStateSA = [device newComputePipelineStateWithFunction:kernelSA error:&error];
        } else {
             std::cerr << "Warning: 'batch_sa_optimize' kernel not found." << std::endl;
        }

        id<MTLFunction> kernelCandidates = [library newFunctionWithName:@"check_candidate_overlaps"];
        if (kernelCandidates) {
            pipelineStateCandidates = [device newComputePipelineStateWithFunction:kernelCandidates error:&error];
        } else {
             std::cerr << "Warning: 'check_candidate_overlaps' kernel not found." << std::endl;
        }
        
        valid = true;
    }
    
    bool compute(const std::vector<ChristmasTree>& trees, float buffer_val) {
        if (!valid || !pipelineState) return false;
        @autoreleasepool {
            size_t n = trees.size();
            if (n < 2) return false;
            
            bool useShared = (pipelineStateShared != nil && n <= 240);
            NSUInteger dataSize = n * sizeof(TreeData);
            
            if (n > cachedTreeCapacity || cachedTreeBuffer == nil) {
                cachedTreeCapacity = n * 2;
                if (cachedTreeCapacity < 256) cachedTreeCapacity = 256;
                
                cachedTreeBuffer = [device newBufferWithLength:cachedTreeCapacity * sizeof(TreeData) options:MTLResourceStorageModeShared];
                cachedTreeBufferOut = [device newBufferWithLength:cachedTreeCapacity * sizeof(TreeData) options:MTLResourceStorageModeShared];
                cachedResultBuffer = [device newBufferWithLength:sizeof(int) options:MTLResourceStorageModeShared];
            }
            
            std::vector<TreeData> data(n);
            for (size_t i = 0; i < n; ++i) {
                data[i].x = (float)trees[i].center_x;
                data[i].y = (float)trees[i].center_y;
                data[i].angle = (float)trees[i].angle_deg;
            }
            memcpy(cachedTreeBuffer.contents, data.data(), dataSize);
            
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
            if (n > cachedTreeCapacity || cachedTreeBuffer == nil) {
                cachedTreeCapacity = n * 2;
                if (cachedTreeCapacity < 256) cachedTreeCapacity = 256;
                cachedTreeBuffer = [device newBufferWithLength:cachedTreeCapacity * sizeof(TreeData) options:MTLResourceStorageModeShared];
                cachedTreeBufferOut = [device newBufferWithLength:cachedTreeCapacity * sizeof(TreeData) options:MTLResourceStorageModeShared];
            }
            
            std::vector<TreeData> data(n);
            for (size_t i = 0; i < n; ++i) {
                data[i].x = (float)trees[i].center_x;
                data[i].y = (float)trees[i].center_y;
                data[i].angle = (float)trees[i].angle_deg;
            }
            memcpy(cachedTreeBuffer.contents, data.data(), dataSize);
            
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
                
                [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];

                id<MTLBuffer> tmp = bufferIn;
                bufferIn = bufferOut;
                bufferOut = tmp;
                
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
            }
            return result;
        }
    }
    
    std::vector<std::vector<ChristmasTree>> batch_sa_optimize(
        const std::vector<std::vector<ChristmasTree>>& solutions,
        const SAParamsGPU& params
    ) {
        if (!valid || !pipelineStateSA) return solutions;
        @autoreleasepool {
            // Flatten solutions
            std::vector<TreeData> all_trees;
            std::vector<int> offsets;
            std::vector<int> sizes;
            std::vector<uint> seeds;
            
            int current_offset = 0;
            int group_count = 0;
            
            for(size_t i=0; i<solutions.size(); ++i) {
                if (solutions[i].empty()) continue;
                
                int sz = (int)solutions[i].size();
                offsets.push_back(current_offset);
                sizes.push_back(sz);
                seeds.push_back(12345 + (uint)i * 100);
                group_count++;
                
                for(const auto& t : solutions[i]) {
                    TreeData d;
                    d.x = (float)t.center_x;
                    d.y = (float)t.center_y;
                    d.angle = (float)t.angle_deg;
                    all_trees.push_back(d);
                }
                current_offset += sz;
            }
            
            if (all_trees.empty()) return solutions;
            
            NSUInteger totalTrees = all_trees.size();
            
            id<MTLBuffer> treesBufferIn = [device newBufferWithBytes:all_trees.data() length:totalTrees * sizeof(TreeData) options:MTLResourceStorageModeShared];
            id<MTLBuffer> treesBufferOut = [device newBufferWithLength:totalTrees * sizeof(TreeData) options:MTLResourceStorageModeShared];
            id<MTLBuffer> offsetsBuffer = [device newBufferWithBytes:offsets.data() length:offsets.size() * sizeof(int) options:MTLResourceStorageModeShared];
            id<MTLBuffer> sizesBuffer = [device newBufferWithBytes:sizes.data() length:sizes.size() * sizeof(int) options:MTLResourceStorageModeShared];
            id<MTLBuffer> seedsBuffer = [device newBufferWithBytes:seeds.data() length:seeds.size() * sizeof(uint) options:MTLResourceStorageModeShared];
            
            id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
            [encoder setComputePipelineState:pipelineStateSA];
            
            [encoder setBuffer:treesBufferIn offset:0 atIndex:0];
            [encoder setBuffer:treesBufferOut offset:0 atIndex:1];
            [encoder setBuffer:offsetsBuffer offset:0 atIndex:2];
            [encoder setBuffer:sizesBuffer offset:0 atIndex:3];
            [encoder setBytes:&params length:sizeof(SAParamsGPU) atIndex:4];
            [encoder setBuffer:seedsBuffer offset:0 atIndex:5];
            
            // Dispatch one threadgroup per system
            // Threadgroup size must be >= max N (200). Use 256.
            MTLSize threadgroupSize = MTLSizeMake(256, 1, 1);
            MTLSize gridSize = MTLSizeMake(group_count * 256, 1, 1); // Grid size is total threads
            
            // Wait, dispatchThreads vs dispatchThreadgroups
            // [encoder dispatchThreadgroups:numGroups threadsPerThreadgroup:threadgroupSize];
            // Metal compute uses grids.
            // If I use dispatchThreadgroups:
            MTLSize groups = MTLSizeMake(group_count, 1, 1);
            [encoder dispatchThreadgroups:groups threadsPerThreadgroup:threadgroupSize];
            
            [encoder endEncoding];
            [commandBuffer commit];
            [commandBuffer waitUntilCompleted];
            
            // Read back
            TreeData* resData = (TreeData*)treesBufferOut.contents;
            std::vector<std::vector<ChristmasTree>> new_solutions = solutions;
            
            int sol_idx = 0;
            for(size_t i=0; i<solutions.size(); ++i) {
                if (solutions[i].empty()) continue;
                int sz = sizes[sol_idx];
                int off = offsets[sol_idx];
                
                for(int k=0; k<sz; ++k) {
                    TreeData d = resData[off + k];
                    new_solutions[i][k].center_x = d.x;
                    new_solutions[i][k].center_y = d.y;
                    new_solutions[i][k].angle_deg = d.angle;
                }
                sol_idx++;
            }
            
            return new_solutions;
        }
    }

    std::vector<int> check_candidates_overlap(
        const std::vector<ChristmasTree>& fixed_trees,
        const std::vector<ChristmasTree>& candidates,
        float buffer
    ) {
        if (!valid || !pipelineStateCandidates || candidates.empty()) return std::vector<int>(candidates.size(), 0);
        
        @autoreleasepool {
            size_t n_fixed = fixed_trees.size();
            size_t n_cand = candidates.size();
            
            // Prepare buffers
            // Fixed Trees
            std::vector<TreeData> fixedData(n_fixed);
            for(size_t i=0; i<n_fixed; ++i) {
                fixedData[i].x = (float)fixed_trees[i].center_x;
                fixedData[i].y = (float)fixed_trees[i].center_y;
                fixedData[i].angle = (float)fixed_trees[i].angle_deg;
            }
            
            // Candidates
            std::vector<TreeData> candData(n_cand);
            for(size_t i=0; i<n_cand; ++i) {
                candData[i].x = (float)candidates[i].center_x;
                candData[i].y = (float)candidates[i].center_y;
                candData[i].angle = (float)candidates[i].angle_deg;
            }
            
            id<MTLBuffer> fixedBuffer = [device newBufferWithBytes:fixedData.data() length:n_fixed * sizeof(TreeData) options:MTLResourceStorageModeShared];
            id<MTLBuffer> candBuffer = [device newBufferWithBytes:candData.data() length:n_cand * sizeof(TreeData) options:MTLResourceStorageModeShared];
            id<MTLBuffer> resultsBuffer = [device newBufferWithLength:n_cand * sizeof(int) options:MTLResourceStorageModeShared];
            
            // Encode
            id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
            [encoder setComputePipelineState:pipelineStateCandidates];
            
            int n_fixed_val = (int)n_fixed;
            int n_cand_val = (int)n_cand;
            
            [encoder setBuffer:fixedBuffer offset:0 atIndex:0];
            [encoder setBuffer:candBuffer offset:0 atIndex:1];
            [encoder setBuffer:resultsBuffer offset:0 atIndex:2];
            [encoder setBytes:&n_fixed_val length:sizeof(int) atIndex:3];
            [encoder setBytes:&n_cand_val length:sizeof(int) atIndex:4];
            [encoder setBytes:&buffer length:sizeof(float) atIndex:5];
            
            MTLSize gridSize = MTLSizeMake(n_cand, 1, 1);
            NSUInteger w = pipelineStateCandidates.maxTotalThreadsPerThreadgroup;
            if (w > n_cand) w = n_cand;
            MTLSize threadgroupSize = MTLSizeMake(w, 1, 1);
            
            [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
            
            [encoder endEncoding];
            [commandBuffer commit];
            [commandBuffer waitUntilCompleted];
            
            // Read results
            int* rawResults = (int*)resultsBuffer.contents;
            std::vector<int> results(n_cand);
            memcpy(results.data(), rawResults, n_cand * sizeof(int));
            
            return results;
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

std::vector<std::vector<ChristmasTree>> GpuContext::batch_sa_optimize(
    const std::vector<std::vector<ChristmasTree>>& solutions,
    const SAParamsGPU& params
) {
    GpuContextImpl* p = (GpuContextImpl*)impl;
    if (p->valid) {
        return p->batch_sa_optimize(solutions, params);
    }
    return solutions;
}

std::vector<int> GpuContext::check_candidates_overlap(
    const std::vector<ChristmasTree>& fixed_trees,
    const std::vector<ChristmasTree>& candidates,
    float buffer
) {
    GpuContextImpl* p = (GpuContextImpl*)impl;
    if (p->valid) {
        return p->check_candidates_overlap(fixed_trees, candidates, buffer);
    }
    return std::vector<int>(candidates.size(), 0);
}

bool GpuContext::is_valid() {
    return ((GpuContextImpl*)impl)->valid;
}
