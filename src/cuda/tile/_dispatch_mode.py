# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import threading
from contextlib import contextmanager

from ._exception import TileStaticEvalError


class DispatchMode:
    @staticmethod
    def get_current() -> "DispatchMode":
        return _current_mode.mode

    @contextmanager
    def as_current(self):
        old_mode = _current_mode.mode
        _current_mode.mode = self
        try:
            yield self
        finally:
            _current_mode.mode = old_mode

    def call_tile_function_from_host(self, func, args, kwargs):
        raise NotImplementedError()


class NormalMode(DispatchMode):
    def call_tile_function_from_host(self, func, args, kwargs):
        raise RuntimeError("Tile functions can only be called from tile code.")


class StaticEvalMode(DispatchMode):
    def call_tile_function_from_host(self, func, args, kwargs):
        from cuda.tile import static_eval
        if func is static_eval:
            raise TileStaticEvalError("static_eval() cannot be used inside"
                                      " another static_eval() expression.")
        else:
            raise TileStaticEvalError("Tile functions cannot be called inside static_eval().")


class _CurrentModeTL(threading.local):
    mode: DispatchMode = NormalMode()


_current_mode = _CurrentModeTL()
