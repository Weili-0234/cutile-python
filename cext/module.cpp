// SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

#include "py.h"

#include "cuda_loader.h"
#include "tile_kernel.h"
#include "cuda_helper.h"

#ifdef _WIN32
extern "C" int _fltused = 0;
#endif


static PyModuleDef module_def = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "cuda.tile._cext",
    .m_size = 0,
};

PyMODINIT_FUNC PyInit__cext() {
    if (!cuda_loader_init())
        return nullptr;

    try_init_cuda();

    PyPtr m = steal(PyModule_Create(&module_def));
    if (!m) return nullptr;

    if (!tile_kernel_init(m.get()))
        return nullptr;

    if (!cuda_helper_init(m.get()))
        return nullptr;

    return m.release();
}

