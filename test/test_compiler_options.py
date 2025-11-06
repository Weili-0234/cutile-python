# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


import cuda.tile as ct
import pytest


def test_invalid_target_name():
    err = r"Invalid GPU architecture name: sm100, expected sm_<major><minor>"
    with pytest.raises(ValueError, match=err):
        ct.ByTarget(sm100=4)
