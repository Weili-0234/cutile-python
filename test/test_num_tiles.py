# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import cuda.tile as ct
import torch
import math
import pytest


@ct.kernel
def check_dim_0d(x):
    n = ct.num_tiles(x, axis=0, shape=())
    ct.store(x, 0, tile=n)


@ct.kernel
def check_dim_1d(x, M: ct.Constant[int]):
    n = ct.num_tiles(x, axis=0, shape=M)
    ct.store(x, 0, tile=n)


@ct.kernel
def check_dim_2d(x, M: ct.Constant[int]):
    n = ct.num_tiles(x, axis=0, shape=(M, M))
    ct.store(x, (0, 0), tile=n)


@pytest.mark.parametrize("shape", [(5,), (10, 10)])
@pytest.mark.parametrize("tile_size", [1, 2])
def test_num_tiles(shape, tile_size):
    x = torch.zeros(shape, dtype=torch.int32, device='cuda')
    stream = torch.cuda.current_stream()
    if len(shape) == 1:
        if tile_size == 1:
            ct.launch(stream, (1,), check_dim_0d, (x,))
        else:
            ct.launch(stream, (1,), check_dim_1d, (x, tile_size))
        res = x[0].item()
    else:
        ct.launch(stream, (1,), check_dim_2d, (x, tile_size))
        res = x[0][0].item()
    assert res == math.ceil(shape[0] / tile_size)
