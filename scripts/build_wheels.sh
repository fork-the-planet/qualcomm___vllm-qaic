#!/usr/bin/env bash
# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
#
# Build vllm-qaic wheels for AOT and/or PYT modes.
#
# Usage:
#   ./scripts/build_wheels.sh [aot|pyt|both] [--pyver 3.12] [--outdir <dir>]
#
# The script installs all build prerequisites automatically into the active environment.
# Activate a venv first, then run:
#   source /path/to/venv/bin/activate
#   ./scripts/build_wheels.sh pyt

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"

source "${SCRIPT_DIR}/utility.sh"

# Defaults
BUILD_TARGET="${1:-both}"
PYTHON_VERSION="3.12"
OUT_DIR="${REPO_ROOT}/dist"

# Parse remaining args
shift || true
while [[ $# -gt 0 ]]; do
    case "$1" in
        --pyver) PYTHON_VERSION="$2"; shift 2 ;;
        --outdir) OUT_DIR="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Prefer the active venv's python so that packages installed by PIP (e.g. 'build')
# are importable when running 'python -m build'. Fall back to the system pythonX.Y.
if [ -n "${VIRTUAL_ENV:-}" ] && [ -x "${VIRTUAL_ENV}/bin/python" ]; then
    PYTHON="${VIRTUAL_ENV}/bin/python"
else
    PYTHON="python${PYTHON_VERSION}"
    if ! command -v "${PYTHON}" &>/dev/null; then
        echo "ERROR: ${PYTHON} not found in PATH. Install Python ${PYTHON_VERSION} first." >&2
        exit 1
    fi
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

echo "========================================================"
echo "  vllm-qaic wheel build"
echo "  target   : ${BUILD_TARGET}"
echo "  python   : ${PYTHON_VERSION}"
echo "  outdir   : ${OUT_DIR}"
echo "========================================================"

# Install build system deps needed for --no-isolation builds.
# Mirrors the build-deps step in install.sh.
setup_build_tools() {
    ${PIP} install "setuptools>=77.0.3,<80.0.0" setuptools-scm wheel build "cmake>=3.26"
}

# Install PYT build prerequisites: CPU torch first, then torch_qaic.
# Mirrors install.sh Step 1 (PYT) — CPU torch must precede torch_qaic because
# torch_qaic validates at import that torch is CPU-only.
# torch must use python -m pip because uv refuses +local version labels
# (e.g. torch==2.10.0+cpu) from remote indexes.
setup_pyt_build_deps() {
    local pyver_tag="py${PYTHON_VERSION/./}"
    echo "=== Installing PYT build deps ==="
    ${PIP} install pip
    ${PYTHON} -m pip install --quiet \
        --index-url https://download.pytorch.org/whl/cpu \
        "torch==${TORCH_VERSION_PYT}" \
        "torchvision==${TORCHVISION_VERSION_PYT}" \
        "torchaudio==${TORCHAUDIO_VERSION_PYT}"
    ${PIP} install "${TORCH_QAIC_BASE_PATH}/${pyver_tag}"/torch_qaic-*.whl
}

build_aot_wheel() {
    echo ""
    echo "=== Building AOT wheel (pure Python, py3-none-any — pyver-independent) ==="
    setup_build_tools
    local aot_out="${OUT_DIR}/aot"
    mkdir -p "${aot_out}"

    cd "${REPO_ROOT}"
    # Force AOT mode: TORCH_QAIC_INSTALLED=0 overrides auto-detection
    TORCH_QAIC_INSTALLED=0 \
        ${PYTHON} -m build --wheel --no-isolation \
        --outdir "${aot_out}"

    local whl
    whl=$(ls "${aot_out}"/vllm_qaic-*aot*.whl 2>/dev/null | head -1)
    if [ -z "${whl}" ]; then
        echo "ERROR: AOT wheel not found in ${aot_out}" >&2
        exit 1
    fi
    echo "AOT wheel: ${whl}"
}

build_pyt_wheel() {
    echo ""
    echo "=== Building PYT wheel (Hexagon compiled) ==="
    setup_build_tools

    # Install torch_qaic + torch if not already present
    if ! ${PYTHON} -c "import torch_qaic" 2>/dev/null; then
        setup_pyt_build_deps
    fi

    local pyver_tag="py${PYTHON_VERSION/./}"
    local pyt_out="${OUT_DIR}/pyt/${pyver_tag}"
    mkdir -p "${pyt_out}"

    cd "${REPO_ROOT}"
    # torch_qaic present → auto-detected as PYT mode
    ${PYTHON} -m build --wheel --no-isolation \
        --outdir "${pyt_out}"

    local whl
    whl=$(ls "${pyt_out}"/vllm_qaic-*pyt*.whl 2>/dev/null | head -1)
    if [ -z "${whl}" ]; then
        echo "ERROR: PYT wheel not found in ${pyt_out}" >&2
        exit 1
    fi
    echo "PYT wheel: ${whl}"
}

case "${BUILD_TARGET}" in
    aot)  build_aot_wheel ;;
    pyt)  build_pyt_wheel ;;
    both) build_aot_wheel; build_pyt_wheel ;;
    *)    echo "ERROR: unknown target '${BUILD_TARGET}'. Use aot|pyt|both" >&2; exit 1 ;;
esac

echo ""
echo "========================================================"
echo "  Build complete. Wheels in ${OUT_DIR}/"
echo "========================================================"
