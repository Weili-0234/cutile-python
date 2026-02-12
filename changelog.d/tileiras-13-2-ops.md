<!--- SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0 -->

Add support for tileiras 13.2 features:
- New `ct.atan2(y, x)` operation for computing the arctangent of y/x
- Optional `rounding_mode` parameter for `ct.tanh()` (supports `RoundingMode.FULL` and `RoundingMode.APPROX`)

Both features require tileiras 13.2 and will raise a clear error message when used with older versions.
