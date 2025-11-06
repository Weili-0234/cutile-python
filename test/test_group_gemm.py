# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from kernels.group_gemm import grouped_matmul_kernel
from util import assert_close
from conftest import float_dtypes, dtype_id
import cuda.tile as ct


@pytest.mark.parametrize("mnk", [
    [(4, 4, 4)],
    [(2, 4, 4), (8, 6, 6), (12, 16, 8)]
])
@pytest.mark.parametrize("tile", [(2, 2, 2), (16, 8, 8)])
@pytest.mark.parametrize("dtype", float_dtypes, ids=dtype_id)
def test_group_gemm(mnk, tile, dtype):
    As = [torch.rand((m, k), dtype=dtype, device='cuda') for (m, _, k) in mnk]
    Bs = [torch.rand((k, n), dtype=dtype, device='cuda') for (_, n, k) in mnk]
    Cs = [torch.rand((m, n), dtype=dtype, device='cuda') for (m, n, _) in mnk]
    tm, tn, tk = tile
    num_sms = 4
    ct.launch(torch.cuda.default_stream(),
              (num_sms,),
              grouped_matmul_kernel,
              (As, Bs, Cs, num_sms, tm, tn, tk))
    atol = 1e-4 if dtype == torch.float32 else 1e-2
    rtol = 1e-4 if dtype == torch.float32 else 1e-2
    for (a, b, c) in zip(As, Bs, Cs):
        ref_result = a @ b
        assert_close(c, ref_result, atol=atol, rtol=rtol)
