#!/usr/bin/env bash
# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
#
# Build and install the triton-cpu backend into the currently active Python environment.
#
# This is a post-install step for AOT SpD testing.  Run it after install.sh aot so that
# the rejection-sampler Triton kernels can execute on CPU (no CUDA / QAIC hardware).
#
# Usage (standalone):
#   ./scripts/install_triton_cpu.sh
#
# Usage (via install.sh):
#   TRITON_CPU=1 ./scripts/install.sh aot
#
# Environment (all have defaults in utility.sh):
#   TRITON_CPU_SRC      Clone destination (default: $HOME/triton-cpu)
#   TRITON_CPU_COMMIT   Pinned commit hash
#   TRITON_CPU_COMPILE_MAX_JOBS  Parallel build jobs (default: 4)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/utility.sh"

# ── Clone / checkout triton-cpu ───────────────────────────────────────────────
if [ ! -f "${TRITON_CPU_SRC}/python/setup.py" ]; then
    echo "INFO: triton-cpu not found at ${TRITON_CPU_SRC} — cloning..."
    git clone https://github.com/triton-lang/triton-cpu "${TRITON_CPU_SRC}"
fi
echo "INFO: checking out commit ${TRITON_CPU_COMMIT}"
git -C "${TRITON_CPU_SRC}" checkout "${TRITON_CPU_COMMIT}"
git -C "${TRITON_CPU_SRC}" submodule update --init --recursive

# ── Resolve the active Python / pip ────────────────────────────────────────
# Prefer ${VIRTUAL_ENV}/bin/python when a venv is active — command -v python can
# resolve to a shadowing conda python when conda's bin appears in PATH before the
# venv's bin (e.g. PATH=miniconda/bin:$PATH prepended after source activate).
if [ -n "${VIRTUAL_ENV:-}" ] && [ -x "${VIRTUAL_ENV}/bin/python" ]; then
    PYTHON="${VIRTUAL_ENV}/bin/python"
else
    PYTHON=$(command -v python 2>/dev/null || true)
fi
if [ -z "${PYTHON}" ]; then
    echo "ERROR: no python found. Activate a conda env or venv first." >&2
    exit 1
fi

# Use uv pip if available and the active venv was created by uv (faster installs).
# uv venvs are identified by a 'uv = ' line in pyvenv.cfg.
UV=$(command -v uv 2>/dev/null || true)
PIP="${PYTHON} -m pip"
USE_UV=0
if [ -n "${UV}" ] && [ -n "${VIRTUAL_ENV:-}" ] && \
   grep -q "^uv = " "${VIRTUAL_ENV}/pyvenv.cfg" 2>/dev/null; then
    PIP="${UV} pip"
    USE_UV=1
    echo "INFO: uv venv detected — using uv pip for faster installs"
fi

echo ""
echo "========================================================"
echo "  triton-cpu installer"
echo "  src      : ${TRITON_CPU_SRC}"
echo "  python   : $(${PYTHON} --version)"
echo "  env      : $(${PYTHON} -c 'import sys; print(sys.prefix)')"
echo "========================================================"
echo ""

# ── Uninstall any existing PyPI triton (it has no CPU backend) ─────────────
if ${PYTHON} -c "import importlib.metadata; importlib.metadata.version('triton')" 2>/dev/null; then
    echo "INFO: uninstalling existing triton package..."
    if [ "${USE_UV}" = "1" ]; then
        ${PIP} uninstall triton
    else
        ${PIP} uninstall -y triton
    fi
fi

# ── Build environment ───────────────────────────────────────────────────────
# Keep build artifacts out of the default ~/.triton to avoid per-user quota issues.
export TRITON_HOME="${TRITON_CPU_SRC}/.triton_home"

# Skip Proton (requires newer CUPTI headers) and C++ unit tests.
export TRITON_BUILD_PROTON=OFF
export TRITON_BUILD_UT=OFF

# Use system-installed NVIDIA tools/headers rather than downloading large bundles.
# These are harmless no-ops if the tools are not present; triton-cpu does not need them.
export TRITON_PTXAS_PATH="${TRITON_PTXAS_PATH:-/usr/bin/ptxas}"
export TRITON_CUOBJDUMP_PATH="${TRITON_CUOBJDUMP_PATH:-/usr/bin/cuobjdump}"
export TRITON_NVDISASM_PATH="${TRITON_NVDISASM_PATH:-/usr/bin/nvdisasm}"
export TRITON_CUDACRT_PATH="${TRITON_CUDACRT_PATH:-/usr/include}"
export TRITON_CUDART_PATH="${TRITON_CUDART_PATH:-/usr/include}"
export TRITON_CUPTI_INCLUDE_PATH="${TRITON_CUPTI_INCLUDE_PATH:-/usr/include}"
export TRITON_CUPTI_LIB_PATH="${TRITON_CUPTI_LIB_PATH:-/usr/lib/x86_64-linux-gnu}"

# ── Build dependencies (needed because --no-build-isolation skips pyproject.toml) ──
# Versions match triton-cpu's pyproject.toml [build-system].requires
${PIP} install "pybind11>=2.13.1" "ninja>=1.11.1"

# ── Install (editable, no build isolation so it reuses existing torch) ──────
echo "=== Building and installing triton-cpu (editable) ==="
MAX_JOBS="${TRITON_CPU_COMPILE_MAX_JOBS:-4}" ${PIP} install -e "${TRITON_CPU_SRC}/python" --no-build-isolation

# ── Sanity check ────────────────────────────────────────────────────────────
echo ""
echo "=== Verifying install ==="
${PYTHON} -c "import triton; print('triton version:', triton.__version__)"
${PYTHON} -c "
import os
os.environ['TRITON_CPU_BACKEND'] = '1'
from triton.backends import backends
active = [k for k, v in backends.items() if v.driver and v.driver.is_active()]
print('active triton backends:', active)
assert 'cpu' in active, 'cpu backend not active — something went wrong'
print('OK: cpu backend is active')
"

echo ""
echo "========================================================"
echo "  triton-cpu installed successfully"
echo ""
echo "  When running SpD tests in AOT mode, set:"
echo "    TRITON_CPU_BACKEND=1 DISABLE_QAIC_EDITABLE_INSTALL=1 pytest ..."
echo "========================================================"
