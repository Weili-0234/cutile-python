# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import cuda.tile as ct

ConstInt = ct.Constant[int]


@ct.kernel
def fft_kernel(x, y, W0, W1, W2, T0, T1,
               N: ConstInt,
               F0: ConstInt,
               F1: ConstInt,
               F2: ConstInt):
    F0F1 = F0 * F1
    F1F2 = F1 * F2
    F0F2 = F0 * F2

    # Load inputs from gmem
    batch = ct.bid(0)
    X_ri = ct.reshape(ct.load(x, index=(batch, 0, 0), shape=(1, N, 2)), (N, 2))

    # Split real and imaginary
    X_r = ct.reshape(ct.extract(X_ri, index=(0, 0), shape=(N, 1)), (F0, F1, F2))
    X_i = ct.reshape(ct.extract(X_ri, index=(0, 1), shape=(N, 1)), (F0, F1, F2))

    # Load all data and separate real and imaginary parts
    W0_ri = ct.load(W0, index=(0, 0, 0), shape=(F0, F0, 2))
    W0_r = ct.reshape(ct.extract(W0_ri, index=(0, 0, 0), shape=(F0, F0, 1)), (F0, F0))
    W0_i = ct.reshape(ct.extract(W0_ri, index=(0, 0, 1), shape=(F0, F0, 1)), (F0, F0))

    W1_ri = ct.load(W1, index=(0, 0, 0), shape=(F1, F1, 2))
    W1_r = ct.reshape(ct.extract(W1_ri, index=(0, 0, 0), shape=(F1, F1, 1)), (F1, F1))
    W1_i = ct.reshape(ct.extract(W1_ri, index=(0, 0, 1), shape=(F1, F1, 1)), (F1, F1))

    W2_ri = ct.load(W2, index=(0, 0, 0), shape=(F2, F2, 2))
    W2_r = ct.reshape(ct.extract(W2_ri, index=(0, 0, 0), shape=(F2, F2, 1)), (F2, F2))
    W2_i = ct.reshape(ct.extract(W2_ri, index=(0, 0, 1), shape=(F2, F2, 1)), (F2, F2))

    T0_ri = ct.load(T0, index=(0, 0, 0), shape=(F0, F1F2, 2))
    T0_r = ct.reshape(ct.extract(T0_ri, index=(0, 0, 0), shape=(F0, F1F2, 1)), (N, 1))
    T0_i = ct.reshape(ct.extract(T0_ri, index=(0, 0, 1), shape=(F0, F1F2, 1)), (N, 1))

    T1_ri = ct.load(T1, index=(0, 0, 0), shape=(F1, F2, 2))
    T1_r = ct.reshape(ct.extract(T1_ri, index=(0, 0, 0), shape=(F1, F2, 1)), (F1F2, 1))
    T1_i = ct.reshape(ct.extract(T1_ri, index=(0, 0, 1), shape=(F1, F2, 1)), (F1F2, 1))

    # CT0 --> Contract over first dimension
    X_r = ct.reshape(X_r, (F0, F1F2))
    X_i = ct.reshape(X_i, (F0, F1F2))
    X_r_ = ct.reshape(ct.matmul(W0_r, X_r) - ct.matmul(W0_i, X_i), (N, 1))
    X_i_ = ct.reshape(ct.matmul(W0_i, X_r) + ct.matmul(W0_r, X_i), (N, 1))

    # Twiddle & Permute 0 --> Apply twiddles and permute for second dimension
    X_r = T0_r * X_r_ - T0_i * X_i_
    X_i = T0_i * X_r_ + T0_r * X_i_
    X_r = ct.permute(ct.reshape(X_r, (F0, F1, F2)), (1, 2, 0))  # F0, F1, F2 -> F1, F2, F0
    X_i = ct.permute(ct.reshape(X_i, (F0, F1, F2)), (1, 2, 0))  # F0, F1, F2 -> F1, F2, F0

    # CT1 --> Contract over second dimension
    X_r = ct.reshape(X_r, (F1, F0F2))
    X_i = ct.reshape(X_i, (F1, F0F2))
    X_r_ = ct.reshape(ct.matmul(W1_r, X_r) - ct.matmul(W1_i, X_i), (F1F2, F0))
    X_i_ = ct.reshape(ct.matmul(W1_i, X_r) + ct.matmul(W1_r, X_i), (F1F2, F0))

    # Twiddle & Permute 1 --> Apply twiddles and permute for third dimension
    X_r = T1_r * X_r_ - T1_i * X_i_
    X_i = T1_i * X_r_ + T1_r * X_i_
    X_r = ct.permute(ct.reshape(X_r, (F1, F2, F0)), (1, 2, 0))  # F1, F2, F0 -> F2, F0, F1
    X_i = ct.permute(ct.reshape(X_i, (F1, F2, F0)), (1, 2, 0))  # F1, F2, F0 -> F2, F0, F1

    # Contract over third dimension
    X_r = ct.reshape(X_r, (F2, F0F1))
    X_i = ct.reshape(X_i, (F2, F0F1))
    X_r_ = ct.matmul(W2_r, X_r) - ct.matmul(W2_i, X_i)
    X_i_ = ct.matmul(W2_i, X_r) + ct.matmul(W2_r, X_i)

    # Final permutation to reverse original order (bit scramble)
    X_r = ct.permute(ct.reshape(X_r_, (F2, F0, F1)), (0, 2, 1))  # F2, F0, F1 -> F2, F1, F0
    X_i = ct.permute(ct.reshape(X_i_, (F2, F0, F1)), (0, 2, 1))  # F2, F0, F1 -> F2, F1, F0
    X_r = ct.reshape(X_r, (N, 1))
    X_i = ct.reshape(X_i, (N, 1))

    # Final reshape and store
    X_ri = ct.reshape(ct.cat((X_r, X_i), axis=1), (1, N, 2))
    ct.store(y, index=(batch, 0, 0), tile=X_ri)
