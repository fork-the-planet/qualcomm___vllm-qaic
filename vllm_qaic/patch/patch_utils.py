# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------

import torch
from vllm.utils import torch_utils

from vllm_qaic.utils import QAIC_KV_CACHE_DTYPE

torch_utils.STR_DTYPE_TO_TORCH_DTYPE[QAIC_KV_CACHE_DTYPE] = torch.int8
