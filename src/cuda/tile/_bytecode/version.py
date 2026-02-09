# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import enum


class BytecodeVersion(enum.IntEnum):
    V_13_1 = 130100
    V_13_2 = 130200

    def major(self) -> int:
        return self._value_ // 10000

    def minor(self) -> int:
        return (self._value_ // 100) % 100

    def tag(self) -> int:
        return self._value_ % 100
