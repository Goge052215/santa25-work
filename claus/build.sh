#!/bin/bash
# Compile Metal shader
echo "Compiling Metal shader..."
xcrun -sdk macosx metal -c gpu/gpu_overlap.metal -o gpu/gpu_overlap.air
xcrun -sdk macosx metallib gpu/gpu_overlap.air -o gpu/gpu_overlap.metallib
rm gpu/gpu_overlap.air

# Compile C++ code
echo "Compiling C++ code..."
clang++ -std=gnu++17 -O3 -Xpreprocessor -fopenmp \
    -fobjc-arc \
    -I/opt/homebrew/opt/libomp/include \
    -L/opt/homebrew/opt/libomp/lib -lomp \
    -framework Metal -framework Foundation \
    main.cpp gpu/gpu_context.mm -o main

