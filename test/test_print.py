# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import math
import sys
import subprocess
import ast
import torch
import numpy as np
import pytest

from math import ceil
import cuda.tile as ct

# FIXME: Default opt_level causes print to be out of order.
# Remove when it is fixed in tile compiler.


@ct.kernel(opt_level=0)
def kernel_printf(x, TILE: ct.Constant[int]):
    bid = ct.bid(0)
    tx = ct.load(x, index=(bid,), shape=(TILE,))
    ct.printf("tile[%d]:%.5f\n", bid, tx)


@ct.kernel(opt_level=0)
def kernel_printd(x, TILE: ct.Constant[int]):
    bid = ct.bid(0)
    tx = ct.load(x, index=(bid,), shape=(TILE,))
    ct.printf("tile[%d]:%d\n", bid, tx)


def _run_kernel_subprocess(shape: str, dtype_str: str, tile: str):
    shape = ast.literal_eval(shape)
    dtype = getattr(torch, dtype_str)
    tile = int(tile)
    x = torch.arange(torch.prod(torch.tensor(shape)), device='cuda').reshape(shape).to(dtype)
    grid = (ceil(shape[0] / tile), 1, 1)
    if "float" in dtype_str:
        ct.launch(torch.cuda.current_stream(), grid, kernel_printf, (x, tile))
    elif "int" in dtype_str:
        ct.launch(torch.cuda.current_stream(), grid, kernel_printd, (x, tile))
    else:
        raise ValueError(f"Unsupported dtype: {dtype_str}")
    torch.cuda.synchronize()


@pytest.mark.parametrize("shape", [(8,), (16,)])
@pytest.mark.parametrize("tile", [8])
@pytest.mark.parametrize("dtype_str", ["float32", "float16", "int32"])
def test_print_1d(shape, tile, dtype_str):
    proc = subprocess.run(
        [sys.executable, __file__, "run_kernel",
         str(shape), dtype_str, str(tile)],
        capture_output=True,
    )
    print(proc.stderr.decode(), file=sys.stderr)
    assert proc.returncode == 0

    actual_outs = [line for line in proc.stdout.decode("UTF-8").splitlines()
                   if line]
    dtype = getattr(np, dtype_str)
    x = np.arange(np.prod(shape)).reshape(shape).astype(dtype)
    num_tiles = math.ceil(shape[0] / tile)
    for i in range(num_tiles):
        start_idx, end_idx = tile*i, tile*(i+1)
        if "float" in dtype_str:
            formatted_x = ', '.join([f"{elem:.5f}" for elem in x[start_idx:end_idx]])
        elif "int" in dtype_str:
            formatted_x = ', '.join([f"{elem}" for elem in x[start_idx:end_idx]])
        else:
            raise ValueError(f"Unsupported dtype: {dtype_str}")
        expected_out = f"tile[{i}]:[{formatted_x}]"
        assert expected_out in actual_outs


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "run_kernel":
        _run_kernel_subprocess(*sys.argv[2:])
