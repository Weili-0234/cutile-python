<!--- SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0 -->

- Fix "potentially undefined variable `$retval`" error when a helper function
  returns after a ``while`` loop that contains no early return.
