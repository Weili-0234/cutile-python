# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import cuda.tile as ct
from cuda.tile._exception import TileTypeError
from util import assert_equal


@ct.kernel
def copy_range(stop, out):
    idx = 0
    for i in range(stop):
        tile = ct.full((1,), fill_value=i, dtype=ct.int32)
        ct.store(out, (idx,), tile=tile)
        idx = idx + 1


@ct.kernel
def copy_range_2(start, stop, out):
    idx = 0
    for i in range(start, stop):
        tile = ct.full((1,), fill_value=i, dtype=ct.int32)
        ct.store(out, (idx,), tile=tile)
        idx = idx + 1


@ct.kernel
def copy_range_2_step_negative(start, stop, out):
    idx = 0
    for i in range(start, stop, -2):
        tile = ct.full((1,), fill_value=i, dtype=ct.int32)
        ct.store(out, (idx,), tile=tile)
        idx = idx + 1


@ct.kernel
def copy_range_3(start, stop, step, out):
    idx = 0
    for i in range(start, stop, step):
        tile = ct.full((1,), fill_value=i, dtype=ct.int32)
        ct.store(out, (idx,), tile=tile)
        idx = idx + 1


def test_range_stop():
    x = torch.zeros((10,), dtype=torch.int32, device='cuda')
    ct.launch(torch.cuda.default_stream(), (1,), copy_range, (10, x))
    assert_equal(x, torch.arange(10, device='cuda', dtype=torch.int32))


def test_range_start_stop():
    start, stop = 1, 11
    L = len(range(start, stop))
    x = torch.zeros((L,), dtype=torch.int32, device='cuda')
    ct.launch(torch.cuda.default_stream(), (1,), copy_range_2, (start, stop, x))
    assert_equal(x, torch.arange(start, stop, device='cuda', dtype=torch.int32))


def test_range_start_stop_negative_constant_step():
    start, stop = 1, 11
    L = len(range(start, stop))
    x = torch.zeros((L,), dtype=torch.int32, device='cuda')
    with pytest.raises(TileTypeError, match='Step must be positive, got -2'):
        ct.launch(torch.cuda.default_stream(), (1,), copy_range_2_step_negative, (start, stop, x))


def test_range_start_stop_positive_step():
    start, stop, step = 1, 11, 2
    L = len(range(start, stop, step))
    x = torch.zeros((L,), dtype=torch.int32, device='cuda')
    ct.launch(torch.cuda.default_stream(), (1,), copy_range_3, (start, stop, step, x))
    assert_equal(x, torch.arange(start, stop, step, device='cuda', dtype=torch.int32))


@pytest.mark.xfail(reason="Issue 314")
def test_range_negative_step():
    start, stop, step = 11, 1, -2
    L = len(range(start, stop, step))
    x = torch.zeros((L,), dtype=torch.int32, device='cuda')
    ct.launch(torch.cuda.default_stream(), (1,), copy_range_3, (start, stop, step, x))
    assert_equal(x, torch.arange(start, stop, step, device='cuda', dtype=torch.int32))


def test_range_scalar_type_error():
    x = torch.zeros((10,), dtype=torch.int32, device='cuda')
    with pytest.raises(TileTypeError, match='Expected a scalar or a 0D tile'):
        ct.launch(torch.cuda.default_stream(), (1,), copy_range, (x, x))


def test_range_integer_type_error():
    x = 1.5
    with pytest.raises(TileTypeError, match='Expected a signed integer'):
        ct.launch(torch.cuda.default_stream(), (1,), copy_range, (x, x))
