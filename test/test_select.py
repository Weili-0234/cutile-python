# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from math import ceil
import cuda.tile as ct
from util import assert_equal
from conftest import float_dtypes, dtype_id
from torch.testing import make_tensor


@ct.kernel
def masked_copy(mask, x, y, z, TILE: ct.Constant[int]):
    bid = ct.bid(0)
    tx = ct.load(x, index=(bid,), shape=(TILE,))
    ty = ct.load(y, index=(bid,), shape=(TILE,))
    tm = ct.load(mask, index=(bid,), shape=(TILE,))
    tz = ct.where(tm, tx, ty)
    ct.store(z, index=(bid,), tile=tz)


@pytest.mark.parametrize("shape", [(128,)])
@pytest.mark.parametrize("tile", [128])
@pytest.mark.parametrize("dtype", float_dtypes, ids=dtype_id)
def test_mask_copy(shape, dtype, tile):
    x = make_tensor(shape, dtype=dtype, device="cuda") - 0.5
    y = torch.zeros_like(x)
    z = torch.zeros_like(x)
    mask = x > y
    grid = (ceil(shape[0] / tile), 1, 1)
    ct.launch(torch.cuda.current_stream(), grid, masked_copy, (mask, x, y, z, tile))
    ref = torch.where(x > y, x, y)
    assert_equal(z, ref)


@ct.kernel
def select_scalar(mask, z, TILE: ct.Constant[int]):
    bid = ct.bid(0)
    tm = ct.load(mask, index=(bid,), shape=(TILE,))
    tz = ct.where(tm, 1.0, -1.0)
    ct.store(z, index=(bid,), tile=tz)


@pytest.mark.parametrize("shape", [(128,)])
@pytest.mark.parametrize("tile", [128])
@pytest.mark.parametrize("dtype", float_dtypes, ids=dtype_id)
def test_select_scalar(shape, dtype, tile):
    x = make_tensor(shape, dtype=dtype, device="cuda") - 0.5
    y = torch.zeros_like(x)
    z = torch.zeros_like(x, dtype=torch.float32)
    mask = x > y
    grid = (ceil(shape[0] / tile), 1, 1)
    ct.launch(torch.cuda.current_stream(), grid, select_scalar, (mask, z, tile))
    ref = torch.where(x > y, 1.0, -1.0)
    assert_equal(z, ref)


@ct.kernel
def select_tensor_scalar(mask, x, z, MASK_N: ct.Constant[int], MASK_M: ct.Constant[int],
                         X_N: ct.Constant[int], X_M: ct.Constant[int]):
    tx = ct.load(x, index=(0, 0), shape=(X_N, X_M))
    tm = ct.load(mask, index=(0, 0), shape=(MASK_N, MASK_M))
    tz = ct.where(tm, tx, 0.)
    ct.store(z, index=(0, 0), tile=tz)


@pytest.mark.parametrize("dtype", float_dtypes, ids=dtype_id)
@pytest.mark.parametrize("x_tile", [(128, 128), (1, 128), (128, 1), (1, 1)])
@pytest.mark.parametrize("mask_tile", [(128, 128), (1, 128), (128, 1), (1, 1)])
def test_select_tensor_scalar(dtype, x_tile, mask_tile):
    x = make_tensor(x_tile, dtype=dtype, device="cuda") - 0.5
    mask = make_tensor(mask_tile, dtype=torch.bool, device="cuda")
    ref = torch.where(mask, x, 0.)

    z = torch.zeros_like(ref)
    ct.launch(torch.cuda.current_stream(), (1,), select_tensor_scalar,
              (mask, x, z, *mask_tile, *x_tile))
    assert_equal(z, ref)
