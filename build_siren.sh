#!/bin/bash
# =============================================================================
# build_siren.sh -- compile the SIREN C++ (including GammaDISFromSpline) and
# (re)install the python module on FASRC.
#
# "pip install ." here is a FULL C++ BUILD: scikit-build-core runs CMake + the
# compiler and builds every .cxx, then installs the wheel. It is not just a
# dependency install.
#
# Usage (run on a COMPUTE NODE, not the login node -- the login node OOM-kills
# the compile):
#     salloc -p sapphire --mem=24G -c 4 -t 0-02:00
#     conda activate siren-dev
#     ./build_siren.sh            # incremental: recompiles only changed files (fast)
#     ./build_siren.sh clean      # full rebuild: wipes build-pip first (use when
#                                 # CMakeLists / new source files changed)
# =============================================================================
set -euo pipefail

SIREN_ROOT="${SIREN_ROOT:-$HOME/SIREN}"
MODE="${1:-incremental}"
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-4}"

cd "$SIREN_ROOT"

# --- toolchain fixes (from geant4_shower/Notes.txt) --------------------------
# The helmod gcc module breaks on libmpfr.so.6; use the conda toolchain instead.
if command -v module >/dev/null 2>&1; then
    module unload gcc 2>/dev/null || true
fi
if [ -n "${CONDA_PREFIX:-}" ]; then
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
else
    echo "WARNING: no CONDA_PREFIX -- did you 'conda activate siren-dev'?" >&2
fi

echo "== compiler =="
echo "  g++: $(command -v g++)"
g++ --version | head -1
echo "  build parallelism: $CMAKE_BUILD_PARALLEL_LEVEL"
echo

# --- prerequisite: cnpy must be vendored (GammaDISFromSpline #include "cnpy.h") --
if [ ! -f third_party/cnpy/cnpy.h ] || [ ! -f third_party/cnpy/cnpy.cpp ]; then
    echo "ERROR: cnpy is not vendored." >&2
    echo "  Put cnpy.h and cnpy.cpp in $SIREN_ROOT/third_party/cnpy/" >&2
    echo "  (from https://github.com/rogersce/cnpy -- MIT license)" >&2
    exit 1
fi

# --- clean mode wipes the build dir so CMake reconfigures from scratch --------
if [ "$MODE" = "clean" ]; then
    echo "== clean build: removing build-pip =="
    rm -rf build-pip
fi

# --- compile + (re)install ----------------------------------------------------
# --force-reinstall --no-deps: replace the installed module even if the version
# is unchanged, without re-resolving dependencies (they're already in siren-dev).
echo "== building (this compiles all C++; first clean build takes a few minutes) =="
python -m pip install . \
    --config-settings='build-dir=build-pip' \
    --force-reinstall --no-deps

# --- sanity check -------------------------------------------------------------
echo
echo "== verifying the module imports and the class registered =="
python -c "import siren; print('OK:', siren.interactions.GammaDISFromSpline)"
echo
echo "build complete."
