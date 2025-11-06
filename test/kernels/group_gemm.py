# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import cuda.tile as ct

ConstInt = ct.Constant[int]


@ct.kernel
def grouped_matmul_kernel(As, Bs, Cs,
                          num_sm: ConstInt,
                          tm: ConstInt, tn: ConstInt, tk: ConstInt):
    tile_idx = ct.bid(0)
    last_problem_end = 0
    group_size = len(As)
    zero_pad = ct.PaddingMode.ZERO
    for g in range(group_size):
        Ai = As[g]
        Bi = Bs[g]
        Ci = Cs[g]
        num_m_tiles = ct.num_tiles(Ai, 0, (tm, tk))
        num_k_tiles = ct.num_tiles(Ai, 1, (tm, tk))
        num_n_tiles = ct.num_tiles(Bi, 1, (tk, tn))
        num_tiles = num_m_tiles * num_n_tiles
        while (tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles):
            tile_idx_in_gemm = tile_idx - last_problem_end
            tile_m_idx = tile_idx_in_gemm // num_n_tiles
            tile_n_idx = tile_idx_in_gemm % num_n_tiles
            acc = ct.zeros((tm, tn), dtype=ct.float32)
            for kk in range(num_k_tiles):
                ta = ct.load(Ai, (tile_m_idx, kk), shape=(tm, tk), padding_mode=zero_pad)
                tb = ct.load(Bi, (kk, tile_n_idx), shape=(tk, tn), padding_mode=zero_pad)
                acc = ct.mma(ta, tb, acc)
            acc = ct.astype(acc, Ci.dtype)
            ct.store(Ci, (tile_m_idx, tile_n_idx), tile=acc)
            tile_idx += num_sm
        last_problem_end = last_problem_end + num_tiles
