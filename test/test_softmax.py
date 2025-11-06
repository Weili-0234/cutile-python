# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from math import ceil
from util import assert_close
from conftest import float_dtypes, dtype_id
from torch.testing import make_tensor

# example-begin imports
import cuda.tile as ct
# example-end imports


# example-begin softmax
@ct.kernel
def softmax(input, output, B: ct.Constant[int], N: ct.Constant[int]):
    rows = ct.load(input, index=(ct.bid(0), 0), shape=(B, N))
    numerator = ct.exp(rows - ct.max(rows, axis=1, keepdims=True))
    denominator = ct.sum(numerator, axis=1, keepdims=True)
    ct.store(output, index=(ct.bid(0), 0), tile=numerator / denominator)
# example-end softmax


@ct.kernel
def softmax_per_row(input, output,
                    num_rows: ct.Constant[int],
                    num_cols: ct.Constant[int]):
    bidx = ct.bid(0)
    num_blocks = ct.num_blocks(0)
    for i in range(bidx, num_rows, num_blocks):
        row = ct.load(input, index=(i, 0), shape=(1, num_cols))
        numerator = ct.exp(row - ct.max(row, axis=1, keepdims=True))
        denominator = ct.sum(numerator, axis=1, keepdims=True)
        ct.store(output, index=(i, 0), tile=numerator / denominator)


def softmax_gold(input):
    max_input = torch.max(input, dim=1, keepdim=True).values
    exps = torch.exp(input - max_input)
    sum_exps = torch.sum(exps, dim=1, keepdim=True)
    return exps / sum_exps


@pytest.mark.parametrize("shape", [(512, 128)])
@pytest.mark.parametrize("tile", [16])
@pytest.mark.parametrize("dtype", float_dtypes, ids=dtype_id)
def test_softmax(shape, tile, dtype):
    x = make_tensor(shape, dtype=dtype, device="cuda")
    y = torch.zeros_like(x)
    grid = (ceil(shape[0] / tile), 1, 1)
    ct.launch(torch.cuda.current_stream(), grid, softmax, (x, y, tile, shape[1]))
    ref_result = softmax_gold(x)
    atol, rtol = (1e-4, 1e-5) if dtype == torch.float32 else (1e-2, 1e-1)
    assert_close(y, ref_result, atol=atol, rtol=rtol)


@pytest.mark.parametrize("shape", [(512, 128)])
@pytest.mark.parametrize("tile", [16])
@pytest.mark.parametrize("dtype", float_dtypes, ids=dtype_id)
def test_softmax_per_row(shape, tile, dtype):
    x = make_tensor(shape, dtype=dtype, device="cuda")
    y = torch.zeros_like(x)
    grid = (ceil(shape[0] / tile), 1, 1)
    ct.launch(torch.cuda.current_stream(), grid, softmax_per_row, (x, y, shape[0], shape[1]))
    ref_result = softmax_gold(x)
    atol, rtol = (1e-4, 1e-5) if dtype == torch.float32 else (1e-2, 1e-1)
    assert_close(y, ref_result, atol=atol, rtol=rtol)
