#!/usr/bin/env bash
# rebuild.sh — rebuild SIREN and deploy to the active Python environment
# Usage: bash rebuild.sh [--jobs N]
#   --jobs N   parallel build jobs (default: all cores)

set -euo pipefail

SIREN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SIREN_ROOT/build"

# ── Parse args ────────────────────────────────────────────────────────────────
JOBS=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --jobs|-j) JOBS="-j $2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ── Locate the active Python's site-packages ──────────────────────────────────
SITE=$(python -c "import site; print(site.getsitepackages()[0])")
SIREN_PKG="$SITE/siren"
SIREN_LIBS="$SITE/siren.libs"

echo "==> SIREN root  : $SIREN_ROOT"
echo "==> Build dir   : $BUILD_DIR"
echo "==> site-packages: $SITE"
echo

# ── Build ─────────────────────────────────────────────────────────────────────
if [[ ! -f "$BUILD_DIR/build.ninja" && ! -f "$BUILD_DIR/Makefile" ]]; then
    echo "ERROR: No build system found in $BUILD_DIR"
    echo "       Run cmake first:  cmake -B $BUILD_DIR -S $SIREN_ROOT"
    exit 1
fi

cd "$BUILD_DIR"
echo "==> Building..."
if [[ -f build.ninja ]]; then
    ninja $JOBS
else
    make $JOBS
fi

# ── Deploy ────────────────────────────────────────────────────────────────────
echo
echo "==> Deploying to $SIREN_PKG ..."

# Main shared library — two locations the extension modules may load from
cp "$BUILD_DIR/libSIREN.so" "$SIREN_PKG/libSIREN.so"
if [[ -d "$SIREN_LIBS" ]]; then
    cp "$BUILD_DIR/libSIREN.so" "$SIREN_LIBS/libSIREN.so"
    echo "    copied libSIREN.so -> siren.libs/"
fi

# Python extension modules
find "$BUILD_DIR" -name "*.cpython-*.so" | while read -r so; do
    cp "$so" "$SIREN_PKG/"
    echo "    copied $(basename "$so")"
done

# ── Verify ────────────────────────────────────────────────────────────────────
echo
echo "==> Verifying subshower symbols..."
if strings "$SIREN_PKG/libSIREN.so" | grep -q "subshower_N"; then
    echo "    OK — subshower code is present in deployed libSIREN.so"
else
    echo "    WARNING — subshower_N not found in deployed libSIREN.so"
fi

echo
echo "Done. Run your simulation and check output with:"
echo "  python -c \"import awkward as ak, h5py, numpy as np; f=h5py.File('output/IceCube.hdf5','r'); grp=f['Events']; arr=ak.from_buffers(ak.forms.from_json(grp.attrs['form']),grp.attrs['length'],{k:np.asarray(v) for k,v in grp.items()}); sN=ak.to_numpy(ak.flatten(arr['subshower_N'])); print('non-nan:', np.any(~np.isnan(sN)), 'min/max:', np.nanmin(sN), np.nanmax(sN))\""
