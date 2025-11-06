# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch


class EventTimeStamp:

    def __init__(self):
        self.event = torch.cuda.Event(enable_timing=True)
        self.event.record()

    def __sub__(self, other: "EventTimeStamp"):
        torch.cuda.synchronize()
        return other.event.elapsed_time(self.event) / 1000


def time():
    return EventTimeStamp()
