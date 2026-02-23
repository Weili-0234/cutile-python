# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch.cuda

import cuda.tile as ct
from cuda.tile import TileValueError


def test_too_few_values_to_unpack():
    @ct.kernel
    def kernel():
        t = 1, 2
        a, b, c = t
    with pytest.raises(TileValueError, match="Too few values to unpack"):
        ct.launch(torch.cuda.current_stream(), (1,), kernel, ())


def test_too_many_values_to_unpack():
    @ct.kernel
    def kernel():
        t = 1, 2, 3
        a, b = t
    with pytest.raises(TileValueError, match="Too many values to unpack"):
        ct.launch(torch.cuda.current_stream(), (1,), kernel, ())


def test_unpack_nested_tuple():
    @ct.kernel
    def kernel(x):
        t = (1, (2, 3, 4)), 5
        [(a, [b, c, d]), e] = t
        ct.scatter(x, 0, a)
        ct.scatter(x, 1, b)
        ct.scatter(x, 2, c)
        ct.scatter(x, 3, d)
        ct.scatter(x, 4, e)

    x = torch.zeros((5,), dtype=torch.int32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (x,))
    assert x.tolist() == [1, 2, 3, 4, 5]
