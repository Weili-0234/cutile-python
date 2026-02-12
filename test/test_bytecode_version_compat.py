# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from unittest.mock import patch

import cuda.tile as ct
from cuda.tile._bytecode.version import BytecodeVersion
from cuda.tile._compile import compile_tile
from cuda.tile._compiler_options import CompilerOptions
from cuda.tile._exception import TileUnsupportedFeatureError
from cuda.tile._numeric_semantics import RoundingMode


def compile_with_version(pyfunc, args, version: BytecodeVersion):
    with patch('cuda.tile._compile._get_max_supported_bytecode_version',
               return_value=version):
        return compile_tile(pyfunc, args, CompilerOptions())


def tensor(dtype=torch.float32):
    return torch.zeros(64, dtype=dtype, device='cuda')


def test_atan2_requires_13_2():
    def kernel(x, y, z):
        tx = ct.load(x, 0, shape=64)
        ty = ct.load(y, 0, shape=64)
        ct.store(z, 0, tile=ct.atan2(tx, ty))

    with pytest.raises(TileUnsupportedFeatureError, match=r"atan2 requires tileiras 13\.2"):
        compile_with_version(kernel, (tensor(), tensor(), tensor()), BytecodeVersion.V_13_1)


def test_tanh_rounding_mode_requires_13_2():
    def kernel(x, y):
        tx = ct.load(x, 0, shape=64)
        ct.store(y, 0, tile=ct.tanh(tx, rounding_mode=RoundingMode.APPROX))

    with pytest.raises(TileUnsupportedFeatureError,
                       match=r"tanh rounding_mode=approx requires tileiras 13\.2"):
        compile_with_version(kernel, (tensor(), tensor()), BytecodeVersion.V_13_1)


def test_tanh_without_rounding_mode_works_on_13_1():
    def kernel(x, y):
        tx = ct.load(x, 0, shape=64)
        ct.store(y, 0, tile=ct.tanh(tx))

    # Should not raise version error
    compile_with_version(kernel, (tensor(), tensor()), BytecodeVersion.V_13_1)
