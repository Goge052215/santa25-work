BREW_LIBOMP ?= $(shell brew --prefix libomp 2>/dev/null)
SRC := claus/main.cpp
OBJ_CLANG := claus/main_clang.o
OBJ_GCC := claus/main_gcc.o
BIN_CLANG := claus/main
BIN_GCC := claus/main_gcc
STD ?= gnu++17

# GPU Support
METAL_SRC := claus/gpu/gpu_overlap.metal
METAL_AIR := claus/gpu/gpu_overlap.air
METAL_LIB := claus/gpu/gpu_overlap.metallib
OBJ_GPU := claus/gpu/gpu_context.o

.PHONY: clang-obj clang gcc-obj gcc clean

$(METAL_LIB): $(METAL_SRC)
	xcrun -sdk macosx metal -c $(METAL_SRC) -o $(METAL_AIR)
	xcrun -sdk macosx metallib $(METAL_AIR) -o $(METAL_LIB)

$(OBJ_GPU): claus/gpu/gpu_context.mm claus/gpu/gpu_context.hpp
	clang++ -std=$(STD) -x objective-c++ -fobjc-arc -c claus/gpu/gpu_context.mm -o $(OBJ_GPU)

clang-obj:
	clang++ -std=$(STD) -g -Xpreprocessor -fopenmp -I$(BREW_LIBOMP)/include -c $(SRC) -o $(OBJ_CLANG)

clang: $(METAL_LIB) $(OBJ_GPU)
	clang++ -std=$(STD) -g -Xpreprocessor -fopenmp -I$(BREW_LIBOMP)/include $(SRC) $(OBJ_GPU) -L$(BREW_LIBOMP)/lib -lomp -Wl,-rpath,$(BREW_LIBOMP)/lib -framework Metal -framework Foundation -framework Accelerate -o $(BIN_CLANG)

gcc-obj:
	g++-14 -std=$(STD) -g -O3 -fopenmp -c $(SRC) -o $(OBJ_GCC)

gcc:
	g++-14 -std=$(STD) -g -O3 -fopenmp $(SRC) -o $(BIN_GCC)

clean:
	rm -f $(OBJ_CLANG) $(OBJ_GCC) $(BIN_CLANG) $(BIN_GCC) $(METAL_AIR) $(METAL_LIB) $(OBJ_GPU)
