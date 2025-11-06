# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import cuda.tile as ct
from util import assert_equal
from conftest import float_dtypes, dtype_id
from torch.testing import make_tensor
from cuda.tile._exception import TileTypeError


@ct.kernel
def permute_tile_2d_explicit_axes(x, y, use_method: ct.Constant[bool]):
    tx = ct.load(x, index=(0, 0), shape=(128, 64))
    if use_method:
        ty = tx.permute(axes=(1, 0))
    else:
        ty = ct.permute(tx, (1, 0))
    ct.store(y, index=(0, 0), tile=ty)


@ct.kernel
def permute_tile_2d_negative_axes(x, y, use_method: ct.Constant[bool]):
    tx = ct.load(x, index=(0, 0), shape=(128, 64))
    if use_method:
        ty = tx.permute(axes=(-1, -2))
    else:
        ty = ct.permute(tx, (-1, -2))
    ct.store(y, index=(0, 0), tile=ty)


@ct.kernel
def transpose_tile_2d_explicit_axes(x, y, use_method: ct.Constant[bool]):
    tx = ct.load(x, index=(0, 0), shape=(128, 64))
    if use_method:
        ty = tx.transpose(axis0=1, axis1=0)
    else:
        ty = ct.transpose(tx, axis0=1, axis1=0)
    ct.store(y, index=(0, 0), tile=ty)


@ct.kernel
def transpose_tile_2d_default_axes(x, y, use_method: ct.Constant[bool]):
    tx = ct.load(x, index=(0, 0), shape=(128, 64))
    if use_method:
        ty = tx.transpose()
    else:
        ty = ct.transpose(tx)
    ct.store(y, index=(0, 0), tile=ty)


@pytest.mark.parametrize("kernel",
                         [permute_tile_2d_explicit_axes,
                          permute_tile_2d_negative_axes,
                          transpose_tile_2d_explicit_axes,
                          transpose_tile_2d_default_axes])
@pytest.mark.parametrize("dtype", float_dtypes, ids=dtype_id)
@pytest.mark.parametrize("use_method", [True, False])
def test_permute_2d(kernel, dtype, use_method):
    x = make_tensor((128, 64), dtype=dtype, device='cuda')
    y = torch.zeros((64, 128), dtype=dtype, device='cuda')
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (x, y, use_method))
    assert_equal(y, x.T)


@ct.kernel
def transpose_tile_3d_explicit_axes(x, y):
    tx = ct.load(x, index=(0, 0, 0), shape=(1, 8, 2))
    ty = ct.transpose(tx, axis0=1, axis1=-1)
    ct.store(y, index=(0, 0, 0), tile=ty)


@pytest.mark.parametrize("dtype", float_dtypes, ids=dtype_id)
def test_transpose_3d(dtype):
    x = make_tensor((1, 8, 2), dtype=dtype, device='cuda')
    y = torch.zeros((1, 2, 8), dtype=dtype, device='cuda')
    ct.launch(torch.cuda.current_stream(), (1,), transpose_tile_3d_explicit_axes, (x, y))
    assert_equal(y, torch.permute(x, (0, 2, 1)))


@ct.kernel
def transpose_tile_1d(x, y):
    tx = ct.load(x, index=(0,), shape=(128,))
    ty = ct.transpose(tx)
    ct.store(y, index=(0,), tile=ty)


def test_transpose_1d():
    x = make_tensor((128, ), dtype=torch.float32, device='cuda')
    y = torch.zeros((128, ), dtype=torch.float32, device='cuda')
    with pytest.raises(TileTypeError,
                       match="Cannot transpose a tile with fewer than 2 dimensions"):
        ct.launch(torch.cuda.current_stream(), (1,), transpose_tile_1d, (x, y))


@ct.kernel
def transpose_tile_2d_single_axis(x, y):
    tx = ct.load(x, index=(0, 0), shape=(128, 64))
    ty = ct.transpose(tx, axis0=0)
    ct.store(y, index=(0, 0), tile=ty)


def test_transpose_2d_single_axis():
    x = make_tensor((128, 64), dtype=torch.float32, device='cuda')
    y = torch.zeros((128, 64), dtype=torch.float32, device='cuda')
    with pytest.raises(TileTypeError,
                       match="transpose axes must either both be specified or both be None"):
        ct.launch(torch.cuda.current_stream(), (1,), transpose_tile_2d_single_axis, (x, y))


@ct.kernel
def transpose_tile_2d_same_axis(x, y):
    tx = ct.load(x, index=(0, 0), shape=(128, 64))
    ty = ct.transpose(tx, 0, 0)
    ct.store(y, index=(0, 0), tile=ty)


def test_transpose_2d_same_axis():
    x = make_tensor((128, 64), dtype=torch.float32, device='cuda')
    y = torch.zeros((128, 64), dtype=torch.float32, device='cuda')
    ct.launch(torch.cuda.current_stream(), (1,), transpose_tile_2d_same_axis, (x, y))
    assert_equal(y, torch.transpose(x, 0, 0))


@ct.kernel
def transpose_tile_3d_default_axes(x, y):
    tx = ct.load(x, index=(0, 0, 0), shape=(128, 64, 32))
    ty = ct.transpose(tx)
    ct.store(y, index=(0, 0, 0), tile=ty)


def test_transpose_3d_default_axes():
    x = make_tensor((128, 64, 32), dtype=torch.float32, device='cuda')
    y = torch.zeros((128, 64, 32), dtype=torch.float32, device='cuda')
    with pytest.raises(TileTypeError,
                       match="`axes` must be specified for tile with more than 2 dimensions"):
        ct.launch(torch.cuda.current_stream(), (1,), transpose_tile_3d_default_axes, (x, y))
