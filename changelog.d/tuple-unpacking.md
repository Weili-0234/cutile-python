<!--- SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0 -->

- Added a missing check for unpacking a tuple with too many values. For example, ``a, b = 1, 2, 3``
  now raises an error, instead of silently discarding the extra value.
- Added support for unpacking nested tuples (e.g, ``a, (b, c) = t``) , as well as using square
  brackets for unpacking (e.g., ``[a, b] = 1, 2``).
- Fixed the missing column indicator in error messages when the underlined text is only one
  character wide.