# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import sys
import subprocess
import torch
import pytest

from torch.testing import make_tensor
import cuda.tile as ct
from util import assert_equal


@ct.kernel
def kernel_ct_assert_with_msg(x, cond, y, TILE: ct.Constant[int]):
    bid = ct.bid(0)
    cond_tile = ct.load(cond, index=(bid,), shape=(TILE,))
    ct.assert_(cond_tile, "assert failed")
    tx = ct.load(x, index=(bid,), shape=(TILE,))
    ct.store(y, index=(bid,), tile=tx)


@ct.kernel
def kernel_ct_assert_without_msg(x, cond, y, TILE: ct.Constant[int]):
    bid = ct.bid(0)
    cond_tile = ct.load(cond, index=(bid,), shape=(TILE,))
    ct.assert_(cond_tile)
    tx = ct.load(x, index=(bid,), shape=(TILE,))
    ct.store(y, index=(bid,), tile=tx)


def _run_kernel_subprocess_ct(cond: bool, has_msg: bool):
    TILE = 128
    x = make_tensor((TILE*2, ), dtype=torch.float32, device="cuda")
    y = torch.zeros_like(x)
    cond_array = torch.full((TILE*2, ), True, device="cuda")
    if not cond:
        # make one element false, so the first tile should fail and the second tile should pass
        cond_array[0] = False
    if has_msg:
        ct.launch(torch.cuda.current_stream(), (2,), kernel_ct_assert_with_msg,
                  (x, cond_array, y, TILE))
    else:
        ct.launch(torch.cuda.current_stream(), (2,), kernel_ct_assert_without_msg,
                  (x, cond_array, y, TILE))
    torch.cuda.synchronize()
    assert_equal(y, x)


@pytest.mark.parametrize("cond", [False, True])
@pytest.mark.parametrize("has_msg", [False, True])
def test_ct_assert(cond, has_msg):
    args = [sys.executable, __file__]
    if cond:
        args.append("--cond")
    if has_msg:
        args.append("--has-msg")
    proc = subprocess.run(args, capture_output=True)
    if cond:
        assert proc.returncode == 0
    else:
        assert proc.returncode != 0
        if has_msg:
            actual_outs = [line for line in proc.stdout.decode("UTF-8").splitlines()
                           if line]
            assert len(actual_outs) == 1
            assert "assert failed" in actual_outs[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cond", action="store_true")
    parser.add_argument("--has-msg", action="store_true")
    args = parser.parse_args()
    _run_kernel_subprocess = _run_kernel_subprocess_ct
    _run_kernel_subprocess(args.cond, args.has_msg)
