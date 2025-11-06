# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from math import ceil
import cuda.tile as ct
from cuda.tile import tfloat32
from util import assert_close
from cuda.tile._exception import TileTypeError

torch.set_float32_matmul_precision("highest")
atol, rtol = 1e-5, 1e-5


@ct.kernel
def mma_kernel(a, b, c, tm: ct.Constant[int], tn: ct.Constant[int], tk: ct.Constant[int]):
    bidx = ct.bid(0)
    bidy = ct.bid(1)
    num_tiles = ct.num_tiles(a, axis=1, shape=(tm, tk))
    acc = ct.full((tm, tn), 0, dtype=np.float32)

    for k in range(num_tiles):
        a_tile = ct.load(a, index=(bidx, k), shape=(tm, tk))
        a_tile = ct.astype(a_tile, tfloat32)
        b_tile = ct.load(b, index=(k, bidy), shape=(tk, tn))
        b_tile = ct.astype(b_tile, tfloat32)
        acc = ct.mma(a_tile, b_tile, acc)

    acc = ct.astype(acc, c.dtype)
    ct.store(c, index=(bidx, bidy), tile=acc)


@pytest.mark.parametrize("shape", [(128, 128, 128), (32, 32, 32)])
@pytest.mark.parametrize("tile", [(64, 64, 32), (16, 8, 8)])
def test_mma(shape, tile):
    m, n, k = shape
    a = torch.rand((m, k), dtype=torch.float32, device="cuda")
    b = torch.rand((k, n), dtype=torch.float32, device="cuda")
    c = torch.zeros((m, n), dtype=torch.float32, device="cuda")
    tm, tn, tk = tile
    grid = (ceil(m / tm), ceil(n / tn), 1)
    ct.launch(torch.cuda.current_stream(), grid, mma_kernel, (a, b, c, tm, tn, tk))
    torch.set_float32_matmul_precision("high")
    ref_result = a @ b
    torch.set_float32_matmul_precision("highest")
    assert_close(c, ref_result, atol=atol, rtol=rtol)


@ct.kernel
def matmul_2d_kernel(a, b, c, tm: ct.Constant[int], tn: ct.Constant[int], tk: ct.Constant[int]):
    bidx = ct.bid(0)
    bidy = ct.bid(1)
    num_tiles = ct.num_tiles(a, axis=1, shape=(tm, tk))
    acc = ct.full((tm, tn), 0, dtype=np.float32)

    for k in range(num_tiles):
        a_tile = ct.load(a, index=(bidx, k), shape=(tm, tk))
        a_tile = ct.astype(a_tile, tfloat32)
        b_tile = ct.load(b, index=(k, bidy), shape=(tk, tn))
        b_tile = ct.astype(b_tile, tfloat32)
        c_tile = ct.matmul(a_tile, b_tile)
        acc = acc + ct.astype(c_tile, acc.dtype)

    acc = ct.astype(acc, c.dtype)
    ct.store(c, index=(bidx, bidy), tile=acc)


@pytest.mark.parametrize("shape", [(128, 128, 128), (32, 32, 32)])
@pytest.mark.parametrize("tile", [(64, 64, 32), (16, 8, 8)])
def test_matmul_2d(shape, tile):
    m, n, k = shape
    a = torch.rand((m, k), dtype=torch.float32, device="cuda")
    b = torch.rand((k, n), dtype=torch.float32, device="cuda")
    c = torch.zeros((m, n), dtype=torch.float32, device="cuda")
    tm, tn, tk = tile
    grid = (ceil(m / tm), ceil(n / tn), 1)
    ct.launch(torch.cuda.current_stream(), grid, matmul_2d_kernel, (a, b, c, tm, tn, tk))
    torch.set_float32_matmul_precision("high")
    ref_result = a @ b
    torch.set_float32_matmul_precision("highest")
    assert_close(c, ref_result, atol=atol, rtol=rtol)


@pytest.mark.parametrize("kernel", [matmul_2d_kernel, mma_kernel])
def test_matmul_unsupported_dtypes(kernel):
    m, n, k = 128, 128, 128
    a = torch.rand((m, k), dtype=torch.float32, device="cuda")
    b = torch.rand((k, n), dtype=torch.bfloat16, device="cuda")
    c = torch.zeros((m, n), dtype=torch.float32, device="cuda")
    tm, tn, tk = 64, 64, 32
    grid = (ceil(m / tm), ceil(n / tn), 1)
    with pytest.raises(TileTypeError):
        ct.launch(torch.cuda.current_stream(), grid, kernel, (a, b, c, tm, tn, tk))
