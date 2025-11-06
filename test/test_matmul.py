# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from math import ceil
import cuda.tile as ct
from cuda.tile._datatype import mma_supported_result_dtype
from cuda.tile._ir.typing_support import to_dtype
from util import assert_close, assert_equal
from conftest import float_dtypes, dtype_id
from cuda.tile._exception import TileTypeError


@ct.kernel
def mma_kernel(A, B, C,
               D,   # dummy array for passing acc dtype
               tm: ct.Constant[int],
               tn: ct.Constant[int],
               tk: ct.Constant[int]):
    bidx = ct.bid(0)
    bidy = ct.bid(1)
    num_tiles = ct.num_tiles(A, axis=1, shape=(tm, tk))
    acc_dtype = D.dtype
    sum = ct.full((tm, tn), 0, dtype=acc_dtype)

    for k in range(num_tiles):
        a = ct.load(A, index=(bidx, k), shape=(tm, tk))
        b = ct.load(B, index=(k, bidy), shape=(tk, tn))
        sum = ct.mma(a, b, sum)

    sum = ct.astype(sum, C.dtype)
    ct.store(C, index=(bidx, bidy), tile=sum)


@ct.kernel
def matmul_2d_kernel(A, B, C,
                     D,   # dummy array for passing dtype
                     tm: ct.Constant[int],
                     tn: ct.Constant[int],
                     tk: ct.Constant[int]):
    bidx = ct.bid(0)
    bidy = ct.bid(1)
    num_tiles = ct.num_tiles(A, axis=1, shape=(tm, tk))
    sum = ct.full((tm, tn), 0, dtype=D.dtype)

    for k in range(num_tiles):
        a = ct.load(A, index=(bidx, k), shape=(tm, tk))
        b = ct.load(B, index=(k, bidy), shape=(tk, tn))
        c = ct.matmul(a, b)
        sum = sum + ct.astype(c, sum.dtype)

    sum = ct.astype(sum, C.dtype)
    ct.store(C, index=(bidx, bidy), tile=sum)


@pytest.mark.parametrize("shape", [(4, 2, 8), (17, 13, 8)])
@pytest.mark.parametrize("tile", [(2, 2, 2), (16, 8, 8)])
@pytest.mark.parametrize("dtype", float_dtypes, ids=dtype_id)
@pytest.mark.parametrize("acc_dtype", float_dtypes, ids=dtype_id)
@pytest.mark.parametrize("kernel", [mma_kernel, matmul_2d_kernel], ids=['mma', 'matmul'])
def test_matmul_2d(shape, tile, dtype, acc_dtype, kernel):
    m, n, k = shape
    A = torch.rand((m, k), dtype=dtype, device="cuda")
    B = torch.rand((k, n), dtype=dtype, device=A.device)
    C = torch.zeros((m, n), dtype=dtype, device=A.device)
    D = torch.ones((1,), dtype=acc_dtype, device=A.device)
    tm, tn, tk = tile
    grid = (ceil(m / tm), ceil(n / tn), 1)

    is_acc_dtype_supported = to_dtype(acc_dtype) in mma_supported_result_dtype(to_dtype(dtype))
    if kernel == matmul_2d_kernel or is_acc_dtype_supported:
        ct.launch(torch.cuda.current_stream(), grid, kernel, (A, B, C, D, tm, tn, tk))
        if acc_dtype == ct.float32:
            ref_result = A @ B
            atol = 1e-4 if dtype == torch.float32 else 1e-2
            rtol = 1e-5 if dtype == torch.float32 else 1e-3
            assert_close(C, ref_result, atol=atol, rtol=rtol)
    else:
        with pytest.raises(TileTypeError, match=r"Expect acc dtype to be in .*, got .*"):
            ct.launch(torch.cuda.current_stream(), grid, kernel, (A, B, C, D, tm, tn, tk))


@pytest.mark.parametrize("shape", [(4, 2, 8), (17, 13, 8)])
@pytest.mark.parametrize("tile", [(2, 2, 2), (16, 8, 8)])
@pytest.mark.parametrize("dtype", [torch.int8], ids=dtype_id)
@pytest.mark.parametrize("acc_dtype", [torch.int32], ids=dtype_id)
@pytest.mark.parametrize("kernel", [mma_kernel, matmul_2d_kernel], ids=['mma', 'matmul'])
def test_matmul_2d_int8(shape, tile, dtype, acc_dtype, kernel):
    m, n, k = shape
    A = torch.randint(-128, 128, (m, k), dtype=dtype, device="cuda")
    B = torch.randint(-128, 128, (k, n), dtype=dtype, device=A.device)
    C = torch.zeros((m, n), dtype=acc_dtype, device=A.device)
    D = torch.ones((1,), dtype=acc_dtype, device=A.device)
    tm, tn, tk = tile
    grid = (ceil(m / tm), ceil(n / tn), 1)

    assert to_dtype(acc_dtype) in mma_supported_result_dtype(to_dtype(dtype))
    ct.launch(torch.cuda.current_stream(), grid, kernel, (A, B, C, D, tm, tn, tk))
    ref_result = (A.float() @ B.float()).to(acc_dtype)
    assert_equal(C, ref_result)


@ct.kernel
def matmul_x_1d_kernel(A, B, C,
                       tn: ct.Constant[int],
                       tk: ct.Constant[int]):
    bidx = ct.bid(0)
    num_tiles = ct.num_tiles(A, axis=0, shape=tk)
    sum = ct.full((tn,), 0, dtype=np.float32)

    for k in range(num_tiles):
        a = ct.load(A, index=(k,), shape=(tk,))
        b = ct.load(B, index=(k, bidx), shape=(tk, tn))
        c = ct.matmul(a, b)
        sum = sum + ct.astype(c, sum.dtype)

    sum = ct.astype(sum, C.dtype)
    ct.store(C, index=(bidx, ), tile=sum)


@pytest.mark.parametrize("shape", [(2, 8), (13, 8)])
@pytest.mark.parametrize("tile", [(2, 2), (8, 8)])
@pytest.mark.parametrize("dtype", float_dtypes, ids=dtype_id)
def test_matmul_x_1d(shape, tile, dtype):
    n, k = shape
    A = torch.rand((k,), dtype=dtype, device="cuda")
    B = torch.rand((k, n), dtype=dtype, device=A.device)
    C = torch.zeros((n,), dtype=dtype, device=A.device)
    tn, tk = tile
    grid = (ceil(n / tn), 1, 1)
    ct.launch(torch.cuda.current_stream(), grid, matmul_x_1d_kernel, (A, B, C, tn, tk))
    ref_result = A @ B
    atol = 1e-4 if dtype == torch.float32 else 1e-2
    rtol = 1e-5 if dtype == torch.float32 else 1e-3
    assert_close(C, ref_result, atol=atol, rtol=rtol)


@ct.kernel
def matmul_y_1d_kernel(A, B, C, tm: ct.Constant[int], tk: ct.Constant[int]):
    bidx = ct.bid(0)
    num_tiles = ct.num_tiles(A, axis=1, shape=(tm, tk))
    sum = ct.full((tm,), 0, dtype=np.float32)

    for k in range(num_tiles):
        a = ct.load(A, index=(bidx, k), shape=(tm, tk))
        b = ct.load(B, index=(k,), shape=(tk,))
        c = a @ b
        sum = sum + ct.astype(c, sum.dtype)

    sum = ct.astype(sum, C.dtype)
    ct.store(C, index=(bidx,), tile=sum)


@pytest.mark.parametrize("shape", [(4, 8), (17, 8)])
@pytest.mark.parametrize("tile", [(2, 2), (16, 8)])
@pytest.mark.parametrize("dtype", float_dtypes, ids=dtype_id)
def test_matmul_y_1d(shape, tile, dtype):
    m, k = shape
    A = torch.rand((m, k), dtype=dtype, device="cuda")
    B = torch.rand((k,), dtype=dtype, device=A.device)
    C = torch.zeros((m,), dtype=dtype, device=A.device)
    tm, tk = tile
    grid = (ceil(m / tm), 1, 1)
    ct.launch(torch.cuda.current_stream(), grid, matmul_y_1d_kernel, (A, B, C, tm, tk))
    ref_result = A @ B
    atol = 1e-4 if dtype == torch.float32 else 1e-2
    rtol = 1e-5 if dtype == torch.float32 else 1e-3
    assert_close(C, ref_result, atol=atol, rtol=rtol)


@pytest.mark.parametrize("kernel", [matmul_2d_kernel, mma_kernel])
def test_matmul_unsupported_dtypes(kernel):
    m, n, k = 4, 2, 8
    A = torch.rand((m, k), dtype=torch.float16, device="cuda")
    B = torch.rand((k, n), dtype=torch.bfloat16, device=A.device)
    C = torch.zeros((m, n), dtype=torch.float32, device=A.device)
    D = torch.zeros((m, n), dtype=torch.float32, device=A.device)
    tm, tn, tk = 2, 2, 2
    grid = (ceil(m / tm), ceil(n / tn), 1)
    with pytest.raises(TileTypeError,
                       match="Implicit promotion of float16 and bfloat16 is not supported."):
        ct.launch(torch.cuda.current_stream(), grid, kernel, (A, B, C, D, tm, tn, tk))


@ct.kernel
def batch_mma_kernel(A, B, C,
                     D,  # dummy array for passing acc dtype
                     tb: ct.Constant[int],
                     tm: ct.Constant[int],
                     tn: ct.Constant[int],
                     tk: ct.Constant[int]):
    bidx = ct.bid(0)
    bidy = ct.bid(1)
    acc_dtype = D.dtype
    NB = ct.num_tiles(A, axis=0, shape=(tb, tm, tk))
    NK = ct.num_tiles(A, axis=2, shape=(tb, tm, tk))
    for bi in range(NB):
        sum = ct.full((tb, tm, tn), 0, dtype=acc_dtype)
        for ki in range(NK):
            a = ct.load(A, index=(bi, bidx, ki), shape=(tb, tm, tk))
            b = ct.load(B, index=(bi, ki, bidy), shape=(tb, tk, tn))
            sum = ct.mma(a, b, sum)
        sum = ct.astype(sum, C.dtype)
        ct.store(C, index=(bi, bidx, bidy), tile=sum)


@pytest.mark.parametrize("shape", [(5, 4, 2, 8), (8, 17, 13, 8)])
@pytest.mark.parametrize("tile", [(2, 2, 2, 2), (8, 16, 8, 8)])
@pytest.mark.parametrize("dtype", float_dtypes, ids=dtype_id)
def test_batch_mma(shape, tile, dtype):
    b, m, n, k = shape
    A = torch.rand((b, m, k), dtype=dtype, device="cuda")
    B = torch.rand((b, k, n), dtype=dtype, device=A.device)
    C = torch.zeros((b, m, n), dtype=dtype, device=A.device)
    D = torch.ones((1,), dtype=torch.float32, device=A.device)
    tb, tm, tn, tk = tile
    grid = (ceil(m / tm), ceil(n / tn), 1)
    ct.launch(torch.cuda.current_stream(), grid, batch_mma_kernel, (A, B, C, D, tb, tm, tn, tk))
    ref_result = A @ B
    atol = 1e-4 if dtype == torch.float32 else 1e-2
    rtol = 1e-5 if dtype == torch.float32 else 1e-3
    assert_close(C, ref_result, atol=atol, rtol=rtol)


@pytest.mark.parametrize("shape", [(5, 4, 2, 8), (8, 17, 13, 8)])
@pytest.mark.parametrize("tile", [(2, 2, 2, 2), (8, 16, 8, 8)])
def test_batch_mma_int8(shape, tile):
    b, m, n, k = shape
    A = torch.randint(-128, 128, (b, m, k), dtype=torch.int8, device="cuda")
    B = torch.randint(-128, 128, (b, k, n), dtype=torch.int8, device=A.device)
    C = torch.zeros((b, m, n), dtype=torch.int32, device=A.device)
    D = torch.ones((1,), dtype=torch.int32, device=A.device)
    tb, tm, tn, tk = tile
    grid = (ceil(m / tm), ceil(n / tn), 1)
    ct.launch(torch.cuda.current_stream(), grid, batch_mma_kernel, (A, B, C, D, tb, tm, tn, tk))
    ref_result = (A.float() @ B.float()).to(torch.int32)
    assert_equal(C, ref_result)
