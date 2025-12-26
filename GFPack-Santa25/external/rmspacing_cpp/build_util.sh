#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

UNAME="$(uname -s)"
if [ "$UNAME" = "Darwin" ]; then
  CXX="${CXX:-clang++}"
else
  CXX="${CXX:-g++}"
fi
PY_INCLUDES="$(python3 -m pybind11 --includes)"
PY_SUFFIX="$(python3-config --extension-suffix)"
PY_LDFLAGS="$(python3-config --ldflags)"

COMMON_FLAGS=(
  -std=c++17
  -O3
  -march=native
  -finline-functions
  -funroll-loops
  -Wall
  -fPIC
)

EXTRA_LIBS=""
EXTRA_FLAGS=""
if [ "$UNAME" = "Darwin" ]; then
  BREW_PREFIX="$(brew --prefix)"
  OMP_PREFIX="${BREW_PREFIX}/opt/libomp"
  EXTRA_FLAGS="-Xpreprocessor -fopenmp -I${BREW_PREFIX}/include -I${OMP_PREFIX}/include"
  EXTRA_LIBS="-L${BREW_PREFIX}/lib -L${OMP_PREFIX}/lib -lomp ${PY_LDFLAGS}"
  EXTRA_FLAGS="${EXTRA_FLAGS} -undefined dynamic_lookup"
else
  COMMON_FLAGS+=(-fopenmp)
fi

"${CXX}" "${COMMON_FLAGS[@]}" ${EXTRA_FLAGS} ${PY_INCLUDES} \
  util.cpp clipper.engine.cpp clipper.offset.cpp clipper.rectclip.cpp \
  -shared -o "calutil${PY_SUFFIX}" ${EXTRA_LIBS}
