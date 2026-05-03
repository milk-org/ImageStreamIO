/**
 * @file    ImageStreamIO_ipc_registry.c
 * @brief   Process-local registry for CUDA IPC memory handles — implementation.
 *
 * See ImageStreamIO_ipc_registry.h for the API description.
 *
 * NOTE: these functions are not thread-safe.  If concurrent access from
 * multiple threads is needed, protect each function body with a
 * pthread_mutex_t.
 */

#include "ImageStreamIO_ipc_registry.h"

#include <string.h> /* memcmp */

/* Storage for the registry table (defined here, declared extern in the header). */
IsioIpcEntry _isio_ipc_registry[ISIO_MAX_IPC_HANDLES];

void *_isio_ipc_lookup(const cudaIpcMemHandle_t *h)
{
    for (int i = 0; i < ISIO_MAX_IPC_HANDLES; i++) {
        if (_isio_ipc_registry[i].refcount > 0 &&
                memcmp(&_isio_ipc_registry[i].handle, h, sizeof(*h)) == 0)
            return _isio_ipc_registry[i].ptr;
    }
    return NULL;
}

void _isio_ipc_insert(const cudaIpcMemHandle_t *h, void *ptr)
{
    for (int i = 0; i < ISIO_MAX_IPC_HANDLES; i++) {
        if (_isio_ipc_registry[i].refcount == 0) {
            _isio_ipc_registry[i].handle   = *h;
            _isio_ipc_registry[i].ptr      = ptr;
            _isio_ipc_registry[i].refcount = 1;
            return;
        }
    }
}

void _isio_ipc_retain(const cudaIpcMemHandle_t *h)
{
    for (int i = 0; i < ISIO_MAX_IPC_HANDLES; i++) {
        if (_isio_ipc_registry[i].refcount > 0 &&
                memcmp(&_isio_ipc_registry[i].handle, h, sizeof(*h)) == 0) {
            _isio_ipc_registry[i].refcount++;
            return;
        }
    }
}

void _isio_ipc_release(const cudaIpcMemHandle_t *h)
{
    for (int i = 0; i < ISIO_MAX_IPC_HANDLES; i++) {
        if (_isio_ipc_registry[i].refcount > 0 &&
                memcmp(&_isio_ipc_registry[i].handle, h, sizeof(*h)) == 0) {
            _isio_ipc_registry[i].refcount--;
            if (_isio_ipc_registry[i].refcount == 0) {
                cudaIpcCloseMemHandle(_isio_ipc_registry[i].ptr);
                _isio_ipc_registry[i].ptr = NULL;
            }
            return;
        }
    }
}
