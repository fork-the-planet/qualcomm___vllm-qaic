# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------

try:
    from vllm_qaic._version import __version__
except ImportError:
    __version__ = "unknown"


def register():
    return "vllm_qaic.platform.QaicPlatform"


def register_connector():
    from vllm_qaic.distributed.kv_transfer.kv_connector import register_connector

    register_connector()
