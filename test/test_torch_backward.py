# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import cuda.tile as ct
import torch
from util import assert_equal


@ct.kernel
def relu_forward(x, y, TILE_SIZE: ct.Constant[int]):
    for i in range(ct.num_tiles(x, 0, TILE_SIZE)):
        tx = ct.load(x, i, TILE_SIZE)
        tx = max(tx, 0)
        ct.store(y, i, tile=tx)


@ct.kernel
def relu_backward(x, dy, TILE_SIZE: ct.Constant[int]):
    for i in range(ct.num_tiles(x, 0, TILE_SIZE)):
        tile_x = ct.load(x, i, TILE_SIZE)
        tile_dy = ct.load(dy, i, TILE_SIZE)
        tile_dx = ct.where(tile_x < 0, 0, 1) * tile_dy
        ct.store(dy, i, tile=tile_dx)


class MyReLU(torch.autograd.Function):
    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(inputs[0])

    @staticmethod
    def forward(x):
        ret = x.clone()
        ct.launch(torch.cuda.current_stream(), (1,),
                  relu_forward, (x, ret, 16))
        return ret

    @staticmethod
    def backward(ctx, *dy):
        x, = ctx.saved_tensors
        dx = dy[0].clone()
        ct.launch(torch.cuda.current_stream(), (1,),
                  relu_backward, (x, dx, 16))
        return dx


def test_backward_relu():
    x = torch.nn.Parameter(torch.randn(5, requires_grad=True, device='cuda'))
    y = MyReLU.apply(x)
    y.sum().backward()
    ref_grad = torch.where(x < 0, 0, 1).to(x.dtype)
    assert_equal(x.grad, ref_grad)
