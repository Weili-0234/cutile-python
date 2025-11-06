# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# example-begin imports
import cuda.tile as ct
# example-end imports


# example-begin constant
def needs_constant(x: ct.Constant):
    pass


def needs_constant_int(x: ct.Constant[int]):
    pass
# example-end constant


# TODO: Run with `mypy --check-untyped-defs` or another static type checker.
def test_constant_type_hints() -> None:
    int_constant: ct.Constant[int] = 42
    float_constant: ct.Constant[float] = 3.14

    needs_constant(int_constant)
    needs_constant(float_constant)
    needs_constant_int(int_constant)
    needs_constant_int(float_constant)  # Should fail type checking
