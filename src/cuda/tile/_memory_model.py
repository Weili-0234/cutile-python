# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from enum import Enum


class MemoryScope(Enum):
    """
    Memory scope for memory operations. Scope is the set of threads that
    may interact directly with that operation and establish any of the relations
    described in the memory model.
    """
    TL_BLK = "tl_blk"
    DEVICE = "device"
    SYS = "sys"


class MemoryOrder(Enum):
    """
    Memory order for memory operations.
    """
    RELAXED = "relaxed"
    ACQUIRE = "acquire"
    RELEASE = "release"
    ACQ_REL = "acq_rel"
    # TODO: expose WEAK for load/store?
