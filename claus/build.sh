#!/bin/bash
clang++ -std=gnu++17 -O3 -Xpreprocessor -fopenmp \
    -I/opt/homebrew/opt/libomp/include \
    -L/opt/homebrew/opt/libomp/lib -lomp \
    -framework Metal -framework Foundation \
    main.cpp gpu/gpu_context.mm -o main
