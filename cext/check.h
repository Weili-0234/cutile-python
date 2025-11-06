/*
 * SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <Python.h>

// Like assert() but can't be disabled
#define CHECK(cond) do { \
        if (!(cond)) Py_FatalError("CHECK FAILED: " #cond); \
    } while (0)
