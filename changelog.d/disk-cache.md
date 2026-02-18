<!--- SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0 -->

- Add bytecode-to-cubin disk cache to avoid recompilation of unchanged kernels.
  Controlled by ``CUDA_TILE_CACHE_DIR`` and ``CUDA_TILE_CACHE_SIZE``.
