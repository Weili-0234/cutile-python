# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from util import assert_close
from conftest import float_dtypes, dtype_id
from torch.testing import make_tensor

import cuda.tile as ct
from cuda.tile._numeric_semantics import PaddingMode


@ct.kernel
def layernorm(A, output, eps, N: ct.Constant[int], TILE_N: ct.Constant[int]):
    bid_m = ct.bid(0)

    num_tiles = ct.num_tiles(A, axis=1, shape=(1, TILE_N))
    total_mean = ct.full((1, TILE_N), 0, dtype=ct.float32)
    for j in range(num_tiles):
        tx = ct.load(A, index=(bid_m, j), shape=(1, TILE_N), padding_mode=PaddingMode.ZERO)
        total_mean += tx
    mean = ct.sum(total_mean, axis=1, keepdims=True) / N

    total_var = ct.full((1, TILE_N), 0, dtype=ct.float32)
    for j in range(num_tiles):
        tx = ct.load(A, index=(bid_m, j), shape=(1, TILE_N), padding_mode=PaddingMode.ZERO)
        centered = tx - mean
        total_var += centered * centered
    var = ct.sum(total_var, axis=1, keepdims=True) / N

    for j in range(num_tiles):
        tx = ct.load(A, index=(bid_m, j), shape=(1, TILE_N), padding_mode=PaddingMode.ZERO)
        normalized = (tx - mean) / ct.sqrt(var + eps)
        normalized = ct.astype(normalized, output.dtype)
        ct.store(output, index=(bid_m, j), tile=normalized)


def layernorm_gold(A, eps):
    """
    Performs layer normalization on the input tensor.

    Args:
        A: Input tensor to normalize
        eps: Small epsilon value for numerical stability

    Returns:
        Tensor containing the layer normalized input
    """
    mean = torch.mean(A, dim=-1, keepdim=True)
    var = torch.var(A, dim=-1, keepdim=True, unbiased=False)
    return (A - mean) / torch.sqrt(var + eps)


@pytest.mark.parametrize("shape", [(128, 256), (120, 256), (128, 250)])
@pytest.mark.parametrize("tile", [16, 32])
@pytest.mark.parametrize("dtype", float_dtypes, ids=dtype_id)
def test_layernorm(shape, tile, dtype):
    x = make_tensor(shape, dtype=dtype, device="cuda")
    y = torch.zeros_like(x)
    eps = 1e-5
    grid = (shape[0], 1, 1)
    ct.launch(torch.cuda.current_stream(), grid, layernorm, (x, y, eps, shape[1], tile))
    ref_result = layernorm_gold(x, eps)
    atol, rtol = {
        torch.float32: (1e-4, 1e-3),
        torch.float16: (1e-3, 1e-2),
        torch.bfloat16: (1e-2, 1e-2),
    }[dtype]
    assert_close(y, ref_result, atol=atol, rtol=rtol)
