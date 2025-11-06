// SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

#include "stream_buffer.h"
#include "check.h"
#include "vec.h"


static constexpr size_t kAlignment = 16;
static constexpr size_t kInitialChunkCapacity = 32 * 1024;
static constexpr size_t kMinChunkToAllocationRatio = 4;


static inline DualPointer dualptr_offset(DualPointer ptr, size_t offset) {
    return {static_cast<char*>(ptr.host) + offset, ptr.device + offset};
}

static inline void dual_ptr_free(DualPointer ptr) {
    g_cuMemFreeHost(ptr.host);
    g_cuMemFree(ptr.device);
}

struct Chunk {
    DualPointer ptr;
    size_t capacity;
    size_t available;
    CUevent event;
    Chunk* next;
};

struct StreamBuffer {
    bool open_transaction;
    Chunk* head;
    Chunk* transaction_head;
    Chunk* tail;
    StreamBuffer* next_free = nullptr;
};

struct StreamBufferPool {
    Vec<unsigned long long> stream_ids;
    Vec<StreamBuffer*> stream_buffers;

    size_t cur_chunk_capacity = kInitialChunkCapacity;

    Chunk* chunk_freelist = nullptr;
    StreamBuffer* sb_freelist = nullptr;
};

StreamBufferPool* stream_buffer_pool_new() {
    return new StreamBufferPool();
}

static void delete_chunk(Chunk* chunk) {
    dual_ptr_free(chunk->ptr);
    g_cuEventDestroy(chunk->event);
    delete chunk;
}

static void reclaim_chunk(Chunk* chunk, StreamBufferPool* pool) {
    if (chunk->capacity == pool->cur_chunk_capacity) {
        chunk->next = pool->chunk_freelist;
        pool->chunk_freelist = chunk;
    } else {
        delete_chunk(chunk);
    }
}

static void poll_events(StreamBuffer* sb, StreamBufferPool* pool) {
    while (sb->head != sb->transaction_head) {
        Chunk* chunk = sb->head;
        if (g_cuEventQuery(chunk->event) != CUDA_SUCCESS)
            break;

        sb->head = chunk->next;
        if (!sb->head) sb->tail = nullptr;
        reclaim_chunk(chunk, pool);
    }
}

static void reclaim(StreamBufferPool* pool) {
    size_t i = 0;
    size_t n = pool->stream_buffers.size();

    StreamBuffer** stream_buffers = pool->stream_buffers.data();
    while (i < n) {
        StreamBuffer* sb = stream_buffers[i];
        poll_events(sb, pool);
        if (!sb->head && !sb->open_transaction) {
            sb->next_free = pool->sb_freelist;
            pool->sb_freelist = sb;
            --n;
            stream_buffers[i] = stream_buffers[n];
            pool->stream_ids[i] = pool->stream_ids[n];
        } else {
            ++i;
        }
    }
    pool->stream_ids.resize(n);
    pool->stream_buffers.resize(n);
}

static Chunk* new_chunk(size_t capacity) {
    DualPointer ptr;
    if (g_cuMemAlloc(&ptr.device, capacity) != CUDA_SUCCESS)
        return nullptr;

    if (g_cuMemAllocHost(&ptr.host, capacity) != CUDA_SUCCESS) {
        g_cuMemFree(ptr.device);
        return nullptr;
    }

    CUevent event;
    if (g_cuEventCreate(&event, CU_EVENT_DISABLE_TIMING) != CUDA_SUCCESS) {
        g_cuMemFree(ptr.device);
        g_cuMemFreeHost(ptr.host);
        return nullptr;
    }

    Chunk* ret = new Chunk;
    ret->ptr = ptr;
    ret->available = ret->capacity = capacity;
    ret->event = event;
    ret->next = nullptr;
    return ret;
}


static void delete_free_chunks(StreamBufferPool* pool) {
    for (Chunk *chunk = pool->chunk_freelist, *next; chunk; chunk = next) {
        next = chunk->next;
        delete_chunk(chunk);
    }
    pool->chunk_freelist = nullptr;
}

static StreamBuffer* alloc_stream_buffer(StreamBufferPool* pool, CUstream stream) {
    StreamBuffer* sb = pool->sb_freelist;
    if (sb) {
        pool->sb_freelist = sb->next_free;
        sb->next_free = nullptr;
        return sb;
    } else {
        sb = new StreamBuffer();
    }
    return sb;
}

static Chunk* allocate_chunk(StreamBufferPool* pool, StreamBuffer* sb,
                             size_t requested_alloc_size) {
    size_t chunk_capacity = pool->cur_chunk_capacity;

    if (chunk_capacity / kMinChunkToAllocationRatio >= requested_alloc_size) {
        if (!pool->chunk_freelist)
            reclaim(pool);

        if (pool->chunk_freelist) {
            // Fast path: get a chunk from the free list
            Chunk* ret = pool->chunk_freelist;
            CHECK(ret->capacity == chunk_capacity);
            pool->chunk_freelist = ret->next;
            ret->available = chunk_capacity;
            ret->next = nullptr;
            return ret;
        }
    } else {
        // To avoid reallocating too often, make sure that at least
        // kMinChunkToAllocationRatio allocations fit into the new chunk
        do {
            if (chunk_capacity >= SIZE_MAX / 2)
                return {};
            chunk_capacity *= 2;
        } while (chunk_capacity / kMinChunkToAllocationRatio < requested_alloc_size);
        pool->cur_chunk_capacity = chunk_capacity;
        delete_free_chunks(pool);
    }
    return new_chunk(chunk_capacity);
}

static size_t find_stream(const Vec<unsigned long long>& stream_ids, unsigned long long stream_id) {
    for (size_t n = stream_ids.size(), i = 0; i < n; ++i) {
        if (stream_ids[i] == stream_id)
            return i;
    }
    return SIZE_MAX;
}

StreamBufferTransaction stream_buffer_transaction_open(StreamBufferPool* pool, CUstream stream) {
    unsigned long long stream_id;
    if (g_cuStreamGetId(stream, &stream_id) != CUDA_SUCCESS)
        return {};

    const Vec<unsigned long long>& stream_ids = pool->stream_ids;
    StreamBuffer* sb;
    size_t i = find_stream(stream_ids, stream_id);
    if (i == SIZE_MAX) {
        sb = alloc_stream_buffer(pool, stream);
        pool->stream_ids.push_back(stream_id);
        pool->stream_buffers.push_back(sb);
    } else {
        sb = pool->stream_buffers[i];
    }
    CHECK(!sb->open_transaction);
    sb->open_transaction = true;
    sb->transaction_head = nullptr;
    return StreamBufferTransaction(pool, sb, stream);
}

DualPointer StreamBufferTransaction::allocate(size_t size) {
    StreamBufferPool* pool = this->pool_;
    StreamBuffer* sb = this->sb_;

    if (size >= SIZE_MAX - kAlignment)
        return {};

    size = (size + kAlignment - 1) & ~static_cast<size_t>(kAlignment - 1);

    Chunk* chunk = sb->tail;
    if (!chunk || chunk->available < size) {
        chunk = allocate_chunk(pool, sb, size);
        if (!chunk) return {};
        if (sb->tail)
            sb->tail->next = chunk;
        else
            sb->head = chunk;
        sb->tail = chunk;
    }

    if (!sb->transaction_head)
        sb->transaction_head = chunk;

    size_t pos = chunk->available -= size;
    return dualptr_offset(chunk->ptr, pos);
}

void StreamBufferTransaction::close() {
    StreamBuffer* sb = this->sb_;
    if (!sb) return;

    CUstream stream = this->stream_;

    this->pool_ = nullptr;
    this->sb_ = nullptr;
    this->stream_ = {};

    for (Chunk* chunk = sb->transaction_head; chunk; chunk = chunk->next) {
        CUresult res = g_cuEventRecord(chunk->event, stream);
        CHECK(res == CUDA_SUCCESS);  // TODO: is there a more graceful way to recover?
    }

    sb->open_transaction = false;
    sb->transaction_head = nullptr;
}
