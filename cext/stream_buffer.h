/*
 * SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "cuda_loader.h"

struct DualPointer {
    // Pointer to host page-locked memory
    void* host;

    // Pointer to device memory of the same size
    CUdeviceptr device;

    inline operator bool () const {
        return host;
    }
};

struct StreamBufferPool;
struct StreamBuffer;


class StreamBufferTransaction {
public:
    StreamBufferTransaction() = default;

    StreamBufferTransaction(StreamBufferPool* pool, StreamBuffer* sb, CUstream stream)
        : pool_(pool), sb_(sb), stream_(stream)
    {}

    StreamBufferTransaction(const StreamBufferTransaction&) = delete;
    void operator= (const StreamBufferTransaction&) = delete;

    StreamBufferTransaction(StreamBufferTransaction&& other)
        : pool_(other.pool_), sb_(other.sb_), stream_(other.stream_)
    {
        other.pool_ = nullptr;
        other.sb_ = nullptr;
        other.stream_ = nullptr;
    }

    StreamBufferTransaction& operator= (StreamBufferTransaction&& other) {
        if (this != &other) {
            close();
            pool_ = other.pool_;
            sb_ = other.sb_;
            stream_ = other.stream_;
            other.pool_ = nullptr;
            other.sb_ = nullptr;
            other.stream_ = nullptr;
        }
        return *this;
    }

    ~StreamBufferTransaction() { close(); }

    explicit operator bool () const { return sb_; }

    DualPointer allocate(size_t size);

    void close();

private:
    StreamBufferPool* pool_ = nullptr;
    StreamBuffer* sb_ = nullptr;
    CUstream stream_ = {};
};

StreamBufferPool* stream_buffer_pool_new();

StreamBufferTransaction stream_buffer_transaction_open(StreamBufferPool* pool, CUstream stream);

