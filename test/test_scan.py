# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from torch.testing import make_tensor
from conftest import float_dtypes, int_dtypes, dtype_id
from util import filecheck, get_bytecode
import cuda.tile as ct
from cuda.tile._exception import TileTypeError
from cuda.tile._numeric_semantics import RoundingMode as RMd


@ct.kernel
def cumsum_axis1(input, output, reverse: ct.Constant[bool],
                 T: ct.Constant[int], N: ct.Constant[int]):
    px = ct.bid(0)
    tile = ct.load(input, index=(px, 0), shape=(T, N))
    out = ct.cumsum(tile, axis=1, reverse=reverse)
    ct.store(output, index=(px, 0), tile=out)


@pytest.mark.parametrize("shape", [(32, 32)])
@pytest.mark.parametrize("dtype", float_dtypes, ids=dtype_id)
@pytest.mark.parametrize("reverse", [False, True])
def test_cumsumf(shape, dtype, reverse):
    x = torch.rand(shape, dtype=dtype, device="cuda") * 2 - 1
    y = torch.zeros_like(x)
    grid = (1, 1, 1)
    ct.launch(torch.cuda.current_stream(), grid, cumsum_axis1, (x, y, reverse, shape[0], shape[1]))
    ref_result = torch.cumsum(x.flip(1), 1).flip(1) if reverse else torch.cumsum(x, 1)
    atol, rtol = (1e-5, 1e-6) if dtype is torch.float32 else (5e-1, 1e-1)
    torch.testing.assert_close(y, ref_result, atol=atol, rtol=rtol)


@pytest.mark.parametrize("shape", [(32, 32)])
@pytest.mark.parametrize("dtype", int_dtypes, ids=dtype_id)
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("low", [-100])
@pytest.mark.parametrize("high", [-20, 100])
def test_cumsumi(shape, dtype, reverse, low, high):
    x = torch.randint(low, high + 1, shape, dtype=dtype, device="cuda")
    y = torch.zeros_like(x)
    grid = (1, 1, 1)
    ct.launch(torch.cuda.current_stream(), grid, cumsum_axis1, (x, y, reverse, shape[0], shape[1]))
    ref_result = torch.cumsum(x.flip(1), 1).flip(1) if reverse else torch.cumsum(x, 1)
    ref_result = ref_result.to(dtype)
    torch.testing.assert_close(y, ref_result)


@pytest.mark.parametrize("shape", [(32, 32)])
@pytest.mark.parametrize("reverse", [False, True])
def test_cumsumb(shape, reverse):
    x = torch.randint(0, 2, shape, dtype=torch.bool, device="cuda")
    y = torch.zeros_like(x, dtype=torch.int32)
    grid = (1, 1, 1)
    ct.launch(torch.cuda.current_stream(), grid, cumsum_axis1, (x, y, reverse, shape[0], shape[1]))
    ref_result = torch.cumsum(x.flip(1), 1).flip(1) if reverse else torch.cumsum(x, 1)
    ref_result = ref_result.to(torch.int32)
    torch.testing.assert_close(y, ref_result)


@ct.kernel
def cumprod_axis0(input, output, reverse: ct.Constant[bool],
                  T: ct.Constant[int], N: ct.Constant[int]):
    px = ct.bid(0)
    tile = ct.load(input, index=(px, 0), shape=(T, N))
    out = ct.cumprod(tile, axis=0, reverse=reverse)
    ct.store(output, index=(px, 0), tile=out)


@pytest.mark.parametrize("shape", [(16, 32)])
@pytest.mark.parametrize("dtype", float_dtypes, ids=dtype_id)
@pytest.mark.parametrize("reverse", [False, True])
def test_cumprodf(shape, dtype, reverse):
    x = torch.rand(shape, dtype=dtype, device="cuda") * 2 - 1
    y = torch.zeros_like(x)
    grid = (1, 1, 1)
    ct.launch(torch.cuda.current_stream(), grid, cumprod_axis0, (x, y, reverse, shape[0], shape[1]))
    ref_result = torch.cumprod(x.flip(0), 0).flip(0) if reverse else torch.cumprod(x, 0)
    atol, rtol = (1e-5, 1e-6) if dtype is torch.float32 else (1e-4, 1e-1)
    torch.testing.assert_close(y, ref_result, atol=atol, rtol=rtol)


@pytest.mark.parametrize("shape", [(16, 32)])
@pytest.mark.parametrize("dtype", int_dtypes, ids=dtype_id)
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("low", [-100])
@pytest.mark.parametrize("high", [-20, 100])
def test_cumprodi(shape, dtype, reverse, low, high):
    x = torch.randint(low, high + 1, shape, dtype=dtype, device="cuda")
    y = torch.zeros_like(x)
    grid = (1, 1, 1)
    ct.launch(torch.cuda.current_stream(), grid, cumprod_axis0, (x, y, reverse, shape[0], shape[1]))
    ref_result = torch.cumprod(x.flip(0), 0).flip(0) if reverse else torch.cumprod(x, 0)
    ref_result = ref_result.to(dtype)
    torch.testing.assert_close(y, ref_result)


@pytest.mark.parametrize("shape", [(16, 32)])
@pytest.mark.parametrize("reverse", [False, True])
def test_cumprodb(shape, reverse):
    x = torch.randint(0, 2, shape, dtype=torch.bool, device="cuda")
    y = torch.zeros_like(x).to(torch.int32)
    grid = (1, 1, 1)
    ct.launch(torch.cuda.current_stream(), grid, cumprod_axis0,
              (x, y, reverse, shape[0], shape[1]))
    ref_result = torch.cumprod(x.flip(0), 0).flip(0) if reverse else torch.cumprod(x, 0)
    ref_result = ref_result.to(torch.int32)
    torch.testing.assert_close(y, ref_result)


def make_scan_rounding_mode(scan_op, rounding_mode):
    @ct.kernel
    def kernel(input, output, T: ct.Constant[int], N: ct.Constant[int]):
        px = ct.bid(0)
        tile = ct.load(input, index=(px, 0), shape=(T, N))
        out = scan_op(tile, axis=0, reverse=False, rounding_mode=rounding_mode)
        ct.store(output, index=(px, 0), tile=out)
    return kernel


@pytest.mark.use_mlir
@pytest.mark.parametrize("shape", [(16, 32)])
@pytest.mark.parametrize("dtype", float_dtypes, ids=dtype_id)
@pytest.mark.parametrize("op_func, tile_op", [(ct.cumsum, "addf"), (ct.cumprod, "mulf")])
@pytest.mark.parametrize("rounding_mode",
                         [RMd.RN, RMd.RZ, RMd.RM, RMd.RP, RMd.FULL, RMd.APPROX, RMd.RZI])
def test_scan_rounding_mode(
    shape, dtype, op_func, tile_op, rounding_mode
):
    should_raise_rounding_mode = rounding_mode in [RMd.RZI, RMd.APPROX, RMd.FULL]
    x = make_tensor(shape, dtype=dtype, device='cuda')
    y = torch.zeros_like(x)
    grid = (1, 1, 1)
    kernel = make_scan_rounding_mode(op_func, rounding_mode)
    if should_raise_rounding_mode:
        with pytest.raises(TileTypeError,
                           match=fr"Rounding mode {rounding_mode.value} is not supported"):
            ct.launch(torch.cuda.current_stream(), grid, kernel, (x, y, shape[0], shape[1]))
    else:
        bytecode = get_bytecode(kernel, (x, y, shape[0], shape[1]))
        if rounding_mode is RMd.RN:
            # Rmd.RN as the default rounding mode is not included in the mlir text
            check_directive = "// CHECK-NOT: rounding<{{[^>]*}}>"
        else:
            check_directive = (
                f"// CHECK: %[[RES:.*]] = {tile_op} %[[A:.*]] rounding<{rounding_mode.value}>"
            )
        filecheck(bytecode, check_directive)
        ct.launch(torch.cuda.current_stream(), grid, kernel, (x, y, shape[0], shape[1]))


def make_scan_flush_to_zero(scan_op, flush_to_zero):
    @ct.kernel
    def kernel(input, output, T: ct.Constant[int], N: ct.Constant[int]):
        px = ct.bid(0)
        tile = ct.load(input, index=(px, 0), shape=(T, N))
        out = scan_op(tile, axis=0, reverse=False, flush_to_zero=flush_to_zero)
        ct.store(output, index=(px, 0), tile=out)
    return kernel


@pytest.mark.use_mlir
@pytest.mark.parametrize("shape", [(16, 32)])
@pytest.mark.parametrize("dtype", float_dtypes, ids=dtype_id)
@pytest.mark.parametrize("op_func, tile_op", [(ct.cumsum, "addf"), (ct.cumprod, "mulf")])
@pytest.mark.parametrize("flush_to_zero", [True, False])
def test_scan_flush_to_zero(shape, dtype, op_func, tile_op, flush_to_zero):
    should_raise = flush_to_zero and (dtype != torch.float32)
    x = make_tensor(shape, dtype=dtype, device='cuda')
    y = torch.zeros_like(x)
    grid = (1, 1, 1)
    kernel = make_scan_flush_to_zero(op_func, flush_to_zero)
    if should_raise:
        with pytest.raises(TileTypeError,
                           match=r"Flush to zero can only be used for float32 type"):
            ct.launch(torch.cuda.current_stream(), grid, kernel, (x, y, shape[0], shape[1]))
    else:
        bytecode = get_bytecode(kernel, (x, y, shape[0], shape[1]))
        if flush_to_zero:
            check_directive = f"// CHECK: %[[RES:.*]] = {tile_op} %[[A:.*]] flush_to_zero :"
        else:
            check_directive = f"// CHECK: %[[RES:.*]] = {tile_op} %[[A:.*]]{{{{[[:space:]]*}}}}:"
        filecheck(bytecode, check_directive)
        ct.launch(torch.cuda.current_stream(), grid, kernel, (x, y, shape[0], shape[1]))
