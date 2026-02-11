<!--- SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0 -->

- Add `ct.isnan()`.
- Fix a bug where `nan != nan` returns False.
- `~x` for const boolean `x` will raise an TypeError to prevent inconsistent
  result comparing to `~x` on a boolean Tile.
