# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch
import pytest

import cuda.tile as ct
from util import assert_equal


@ct.kernel
def add_arrays(arrays, out):
    res = ct.zeros((16, 16), dtype=out.dtype)
    for i in range(len(arrays)):
        t = ct.load(arrays[i], (0, 0), (16, 16))
        res += t
    ct.store(out, (0, 0), res)


@ct.kernel
def add_arrays_with_const_index(arrays, out):
    tx = ct.load(arrays[0], (0, 0), (16, 16))
    ty = ct.load(arrays[1], (0, 0), (16, 16))
    tz = ct.load(arrays[2], (0, 0), (16, 16))
    res = tx + ty + tz
    ct.store(out, (0, 0), res)


@ct.kernel
def add_arrays_with_0d_tile_index(arrays, out):
    bid = ct.full((), 0, dtype=ct.int32)
    tx = ct.load(arrays[bid], (0, 0), (16, 16))
    ty = ct.load(arrays[bid + 1], (0, 0), (16, 16))
    tz = ct.load(arrays[bid + 2], (0, 0), (16, 16))
    res = tx + ty + tz
    ct.store(out, (0, 0), res)


@pytest.mark.parametrize("kernel", [
    add_arrays,
    add_arrays_with_const_index,
    add_arrays_with_0d_tile_index
    ])
def test_add_list_of_arrays(kernel):
    arrays = [torch.randint(0, 100, (16, 16), dtype=torch.int32, device="cuda") for _ in range(3)]
    out = torch.zeros(16, 16, dtype=torch.int32, device="cuda")
    ref = sum(arrays)

    ct.launch(torch.cuda.current_stream(), (1,), kernel, (arrays, out))
    assert_equal(out, ref)
