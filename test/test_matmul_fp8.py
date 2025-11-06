# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from math import ceil
import cuda.tile as ct
from util import assert_close


def make_mma(acc_dtype):
    @ct.kernel
    def kernel(a, b, c, tm: ct.Constant[int], tn: ct.Constant[int], tk: ct.Constant[int]):
        bidx = ct.bid(0)
        bidy = ct.bid(1)
        num_tiles = ct.num_tiles(a, axis=1, shape=(tm, tk))
        acc = ct.full((tm, tn), 0, dtype=acc_dtype)

        for k in range(num_tiles):
            a_tile = ct.load(a, index=(bidx, k), shape=(tm, tk))
            b_tile = ct.load(b, index=(k, bidy), shape=(tk, tn))
            acc = ct.mma(a_tile, b_tile, acc)

        acc = ct.astype(acc, c.dtype)
        ct.store(c, index=(bidx, bidy), tile=acc)
    return kernel


@pytest.mark.parametrize("shape", [(128, 128, 128), (32, 32, 32)])
@pytest.mark.parametrize("tile", [(64, 64, 32), (16, 8, 8)])
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("acc_dtype", [torch.float32, torch.float16])
def test_mma(shape, tile, dtype, acc_dtype):
    m, n, k = shape
    a = torch.rand((m, k), dtype=torch.float32, device="cuda").to(dtype)
    b = torch.rand((k, n), dtype=torch.float32, device="cuda").to(dtype)
    c = torch.zeros((m, n), dtype=acc_dtype, device="cuda")
    tm, tn, tk = tile
    grid = (ceil(m / tm), ceil(n / tn), 1)
    ct.launch(torch.cuda.current_stream(), grid, make_mma(acc_dtype), (a, b, c, tm, tn, tk))

    # Only multiplication of row-major and column-major matrices is supported by cuBLASLt
    # Move to CPU for reference.
    a = a.to(device="cpu")
    b = b.to(device="cpu")
    inv_sa = torch.tensor(1.0, device=a.device, dtype=torch.float32)
    inv_sb = torch.tensor(1.0, device=b.device, dtype=torch.float32)
    ref_result = torch._scaled_mm(a, b, scale_a=inv_sa, scale_b=inv_sb, out_dtype=acc_dtype)
    atol, rtol = (1e-5, 1e-5) if acc_dtype == torch.float32 else (1e-2, 1e-2)
    assert_close(c, ref_result.to(device="cuda"), atol=atol, rtol=rtol)


@ct.kernel
def matmul_2d(a, b, c, tm: ct.Constant[int], tn: ct.Constant[int], tk: ct.Constant[int]):
    bidx = ct.bid(0)
    bidy = ct.bid(1)
    num_tiles = ct.num_tiles(a, axis=1, shape=(tm, tk))
    acc = ct.full((tm, tn), 0, dtype=np.float32)

    for k in range(num_tiles):
        a_tile = ct.load(a, index=(bidx, k), shape=(tm, tk))
        b_tile = ct.load(b, index=(k, bidy), shape=(tk, tn))
        c_tile = ct.matmul(a_tile, b_tile)
        acc = acc + ct.astype(c_tile, acc.dtype)

    acc = ct.astype(acc, c.dtype)
    ct.store(c, index=(bidx, bidy), tile=acc)


@pytest.mark.parametrize("shape", [(128, 128, 128), (32, 32, 32)])
@pytest.mark.parametrize("tile", [(64, 64, 32), (16, 8, 8)])
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_matmul_2d(shape, tile, dtype):
    m, n, k = shape
    a = torch.rand((m, k), dtype=torch.float32, device="cuda").to(dtype)
    b = torch.rand((k, n), dtype=torch.float32, device="cuda").to(dtype)
    c = torch.zeros((m, n), dtype=torch.float32, device="cuda")
    tm, tn, tk = tile
    grid = (ceil(m / tm), ceil(n / tn), 1)
    ct.launch(torch.cuda.current_stream(), grid, matmul_2d, (a, b, c, tm, tn, tk))
    a = a.to(device="cpu")
    b = b.to(device="cpu")
    inv_sa = torch.tensor(1.0, device=a.device, dtype=torch.float32)
    inv_sb = torch.tensor(1.0, device=b.device, dtype=torch.float32)
    ref_result = torch._scaled_mm(a, b, scale_a=inv_sa, scale_b=inv_sb, out_dtype=torch.float32)
    atol, rtol = 1e-5, 1e-5
    assert_close(c, ref_result.to(device="cuda"), atol=atol, rtol=rtol)
