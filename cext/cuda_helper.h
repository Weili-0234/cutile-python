/*
 * SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "py.h"
#include <cuda.h>

Status cuda_helper_init(PyObject* m);

const char* get_cuda_error(CUresult res);

void try_init_cuda();
