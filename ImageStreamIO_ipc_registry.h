/**
 * @file    ImageStreamIO_ipc_registry.h
 * @brief   Process-local registry for CUDA IPC memory handles.
 *
 * cudaIpcOpenMemHandle is inter-process only: calling it a second time for
 * the same handle within the same process returns cudaErrorInvalidDeviceContext.
 * This registry maps handle bytes -> (device pointer, refcount) so subsequent
 * opens within the same process are served from a cache instead of calling the
 * CUDA API again.  The handle is closed (and its slot freed) only when the
 * refcount reaches 0.
 *
 * Storage and implementations live in ImageStreamIO_ipc_registry.c.
 *
 * NOTE: these functions are not thread-safe.  If concurrent access from
 * multiple threads is needed, protect each function body in the .c file with
 * a pthread_mutex_t.
 */

#ifndef IMAGESTREAMIO_IPC_REGISTRY_H
#define IMAGESTREAMIO_IPC_REGISTRY_H

#ifndef HAVE_CUDA
#error "ImageStreamIO_ipc_registry.h requires HAVE_CUDA to be defined (CUDA support must be enabled)"
#endif

#include <cuda_runtime.h>

#define ISIO_MAX_IPC_HANDLES 128

typedef struct {
    cudaIpcMemHandle_t handle;
    void              *ptr;
    int                refcount; /* 0 means slot is free */
} IsioIpcEntry;

extern IsioIpcEntry _isio_ipc_registry[ISIO_MAX_IPC_HANDLES];

/** @brief Return the cached device pointer for h, or NULL if not found. */
void *_isio_ipc_lookup(const cudaIpcMemHandle_t *h);

/** @brief Insert a new handle+pointer pair with refcount=1.
 *         No-op if the table is full. */
void  _isio_ipc_insert(const cudaIpcMemHandle_t *h, void *ptr);

/** @brief Increment the refcount for h.  No-op if h is not in the registry. */
void  _isio_ipc_retain(const cudaIpcMemHandle_t *h);

/** @brief Decrement the refcount for h.
 *         Calls cudaIpcCloseMemHandle and frees the slot when count reaches 0.
 *         No-op if h is not in the registry. */
void  _isio_ipc_release(const cudaIpcMemHandle_t *h);

#endif /* IMAGESTREAMIO_IPC_REGISTRY_H */
