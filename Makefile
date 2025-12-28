BREW_LIBOMP ?= $(shell brew --prefix libomp 2>/dev/null)
SRC := claude/a.cpp
OBJ_CLANG := claude/a_clang.o
OBJ_GCC := claude/a_gcc.o
BIN_CLANG := claude/a_clang
BIN_GCC := claude/a_gcc
STD ?= gnu++14

.PHONY: clang-obj clang gcc-obj gcc clean

clang-obj:
	clang++ -std=$(STD) -g -Xpreprocessor -fopenmp -I$(BREW_LIBOMP)/include -c $(SRC) -o $(OBJ_CLANG)

clang:
	clang++ -std=$(STD) -g -Xpreprocessor -fopenmp -I$(BREW_LIBOMP)/include $(SRC) -L$(BREW_LIBOMP)/lib -lomp -Wl,-rpath,$(BREW_LIBOMP)/lib -o $(BIN_CLANG)

gcc-obj:
	g++-14 -std=$(STD) -g -O3 -fopenmp -c $(SRC) -o $(OBJ_GCC)

gcc:
	g++-14 -std=$(STD) -g -O3 -fopenmp $(SRC) -o $(BIN_GCC)

clean:
	rm -f $(OBJ_CLANG) $(OBJ_GCC) $(BIN_CLANG) $(BIN_GCC)
