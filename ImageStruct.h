/**
 * @file    ImageStruct.h
 * @brief   Image structure definition
 *
 * The IMAGE structure is defined here
 * Supports shared memory, low latency IPC through semaphores
 *
 * Dynamic allocation within IMAGE:
 * IMAGE includes a pointer to an array of IMAGE_METADATA (usually only one element, >1 element for polymorphism)
 * IMAGE includes a pointer to an array of KEYWORDS
 * IMAGE includes a pointer to a data array
 *
 *
 * @bug No known bugs.
 *
 */

#ifndef _IMAGESTRUCT_H
#define _IMAGESTRUCT_H

#define IMAGESTRUCT_VERSION "0.0.01"

#include <semaphore.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <unistd.h>

#ifdef HAVE_CUDA
// CUDA runtime includes
#include <cuda_runtime_api.h>
#else
// CUDA cudaIpcMemHandle_t is a struct of 64 bytes
// This is needed for for compatibility between ImageStreamIO
// compiled with or without HAVE_CUDA precompiler flag
typedef char cudaIpcMemHandle_t[64];
#endif

#include "ImageError.h"

#ifdef __MACH__
#include <mach/mach_time.h>
#define CLOCK_REALTIME 0
#define CLOCK_MONOTONIC 0
static int clock_gettime(int clk_id, struct mach_timespec *t) {
    mach_timebase_info_data_t timebase;
    mach_timebase_info(&timebase);
    uint64_t time;
    time = mach_absolute_time();
    double nseconds =
        ((double)time * (double)timebase.numer) / ((double)timebase.denom);
    double seconds =
        ((double)time * (double)timebase.numer) / ((double)timebase.denom * 1e9);
    t->tv_sec = seconds;
    t->tv_nsec = nseconds;
    return EXIT_SUCCESS;
}
#else
#include <time.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

// comment this line if data should not be packed
// packing data should be use with extreme care, so it is recommended to disable this feature
//#define DATA_PACKED

#define SHAREDMEMDIR    "/milk/shm"          /**< default location of file mapped semaphores, can be over-ridden by env variable MILK_SHM_DIR */

#define SEMAPHORE_MAXVAL    10            /**< maximum value for each of the semaphore, mitigates warm-up time when processes catch up with data that has accumulated */
#define SEMAPHORE_INITVAL    0            /**< maximum value for each of the semaphore, mitigates warm-up time when processes catch up with data that has accumulated */
#define IMAGE_NB_SEMAPHORE  10            /**< Number of semaphores per image */

// Data types are defined as machine-independent types for portability

#define _DATATYPE_UINT8                                1  /**< uint8_t       = char */
#define SIZEOF_DATATYPE_UINT8                          1

#define _DATATYPE_INT8                                 2  /**< int8_t   */
#define SIZEOF_DATATYPE_INT8                           1

#define _DATATYPE_UINT16                               3  /**< uint16_t      usually = unsigned short int */
#define SIZEOF_DATATYPE_UINT16                         2

#define _DATATYPE_INT16                                4  /**< int16_t       usually = short int          */
#define SIZEOF_DATATYPE_INT16                          2

#define _DATATYPE_UINT32                               5  /**< uint32_t      usually = unsigned int       */
#define SIZEOF_DATATYPE_UINT32                         4

#define _DATATYPE_INT32                                6  /**< int32_t       usually = int                */
#define SIZEOF_DATATYPE_INT32                          4

#define _DATATYPE_UINT64                               7  /**< uint64_t      usually = unsigned long      */
#define SIZEOF_DATATYPE_UINT64                         8

#define _DATATYPE_INT64                                8  /**< int64_t       usually = long               */
#define SIZEOF_DATATYPE_INT64                          8

#define _DATATYPE_HALF                                13  /**< IEE 754 half-precision 16-bit (uses uint16_t for storage) */
#define SIZEOF_DATATYPE_HALF                           2

#define _DATATYPE_FLOAT                                9  /**< IEEE 754 single-precision binary floating-point format: binary32 */
#define SIZEOF_DATATYPE_FLOAT                          4

#define _DATATYPE_DOUBLE                              10 /**< IEEE 754 double-precision binary floating-point format: binary64 */
#define SIZEOF_DATATYPE_DOUBLE                         8

#define _DATATYPE_COMPLEX_FLOAT                       11  /**< complex_float  */
#define SIZEOF_DATATYPE_COMPLEX_FLOAT                  8

#define _DATATYPE_COMPLEX_DOUBLE                      12  /**< complex double */
#define SIZEOF_DATATYPE_COMPLEX_DOUBLE                16

#define _DATATYPE_EVENT_UI8_UI8_UI16_UI8              20
#define SIZEOF_DATATYPE_EVENT_UI8_UI8_UI16_UI8         5

#define Dtype                                          9   /**< default data type for floating point */
#define CDtype                                        11   /**< default data type for complex */

// Type of stream

#define CIRCULAR_BUFFER  0x0001  /**< Circular buffer, slice z axis is encoding time -> record writetime array */
#define MATH_DATA        0x0002  /**< Image is mathematical vector or matrix */
#define IMG_RECV         0x0004  /**< Image is stream received from another computer */
#define IMG_SENT         0x0008  /**< Image is stream sent to other computer */

// axis0 definition

#define ZAXIS_UNDEF      0x00000  /**< undefined (default) */
#define ZAXIS_SPACIAL    0x10000  /**< spatial coordinate */
#define ZAXIS_TEMPORAL   0x20000  /**< temporal coordinate */
#define ZAXIS_WAVELENGTH 0x30000  /**< wavelength coordinate */
#define ZAXIS_MAPPING    0x40000  /**< mapping index */

/** @brief  Keyword
 * The IMAGE_KEYWORD structure includes :
 * 	- name
 * 	- type
 * 	- value
 */
typedef struct {
    char name[16];         /**< keyword name                                                   */
    char type;             /**< N: unused, L: long, D: double, S: 16-char string               */
    uint64_t : 0;  // align array to 8-byte boundary for speed

    union {
        int64_t numl;
        double  numf;
        char    valstr[16];
    } value;

    char comment[80];
#ifdef DATA_PACKED
} __attribute__((__packed__)) IMAGE_KEYWORD;
#else
} IMAGE_KEYWORD;
#endif

/** @brief structure holding two 8-byte integers
 *
 * Used in an union with struct timespec to ensure fixed 16 byte length
 */
typedef struct {
    int64_t firstlong;
    int64_t secondlong;
} TIMESPECFIXED;

typedef struct {
    float re;
    float im;
} complex_float;

typedef struct {
    double re;
    double im;
} complex_double;




/** @brief Image metadata
 *
 *
 *
 */
typedef struct
{
    char version[32];
    /** Image structure version.
     *
     * should be equal to IMAGESTRUCT_VERSION
     *
     * Will be tested to ensure current software revision matches data.
     * If does not match, return error message with both versions.
     */

    /** @brief Image Name */
    char name[80];


    /** @brief Number of axis
     *
     * @warning 1, 2 or 3. Values above 3 not supported.
     */
    uint8_t naxis;


    /** @brief Image size along each axis
     *
     *  If naxis = 1 (1D image), size[1] and size[2] are irrelevant
     */
    uint32_t size[3];


    /** @brief Number of elements in image
     *
     * This is computed upon image creation
     */
    uint64_t nelement;



    /** @brief Data type
     *
     * Encoded according to data type defines.
     *  -  1: uint8_t
     * 	-  2: int8_t
     * 	-  3: uint16_t
     * 	-  4: int16_t
     * 	-  5: uint32_t
     * 	-  6: int32_t
     * 	-  7: uint64_t
     * 	-  8: int64_t
     * 	-  9: IEEE 754 single-precision binary floating-point format: binary32
     *  - 10: IEEE 754 double-precision binary floating-point format: binary64
     *  - 11: complex_float
     *  - 12: complex double
     *  - 13: half precision floating-point
     *
     */
    uint8_t datatype;





    uint64_t imagetype;              /**< image type */
    /**
     * 0x 0000 0000 0000 0001  Circular buffer, slice z axis is encoding time -> record writetime array
     * 0x 0000 0000 0000 0002  Image is mathematical vector or matrix
     * 0x 0000 0000 0000 0004  Image is stream received from another computer
     * 0x 0000 0000 0000 0008  Image is stream sent to other computer
     *
     * 0x 0000 0000 000X 0000  axis[0] encoding code (0-15):
     *    0: undefined (default)
     *    1: spatial coordinate
     *    2: temporal coordinate
     *    3: wavelength coordinate
     *    4: mapping index
     *
     *
     *
     */




    // relative timers using time relative to process start

    // double creationtime;             /**< Creation / load time of data structure (since process start)  */    
    // double lastaccesstime;           /**< last time the image was accessed  (since process start)                      */



    // absolute timers using struct timespec

    struct timespec creationtime;
    struct timespec lastaccesstime;

    struct timespec atime;             /**< time at which data was acquires/created. This time CAN be copied from input to output */
    struct timespec writetime;         /**< last write time into data array         */




    uint8_t  shared;                   /**< 1 if in shared memory                                                        */
    int8_t   location;                 /**< -1 if in CPU memory, >=0 if in GPU memory on `location` device               */
    uint8_t  status;                   /**< 1 to log image (default); 0 : do not log: 2 : stop log (then goes back to 2) */
    uint64_t flag;                     /**< bitmask, encodes read/write permissions.... NOTE: enum instead of defines */

    uint8_t  logflag;                  /**< set to 1 to start logging         */
    uint16_t sem;                      /**< number of semaphores in use, specified at image creation      */


    uint64_t : 0; // align array to 8-byte boundary for speed

    uint64_t cnt0;               	/**< counter (incremented if image is updated)                                    */
    uint64_t cnt1;               	/**< in 3D rolling buffer image, this is the last slice written                   */
    uint64_t cnt2;                  /**< in event mode, this is the # of events                                       */

    uint8_t  write;               	/**< 1 if image is being written                                                  */



    uint16_t NBkw;                  /**< number of keywords (max: 65536)                                              */

    cudaIpcMemHandle_t cudaMemHandle;

#ifdef DATA_PACKED
} __attribute__((__packed__)) IMAGE_METADATA;
#else
} IMAGE_METADATA;
#endif




/** @brief IMAGE structure
 * The IMAGE structure includes :
 *   - an array of IMAGE_KEWORD structures
 *   - an array of IMAGE_METADATA structures (usually only 1 element)
 *
 *
 */
typedef struct /**< structure used to store data arrays                      */
{
    char name[80];     /**< local name (can be different from name in shared memory) */
    // mem offset = 80

    /** @brief Image usage flag
     *
     * 1 if image is used, 0 otherwise. \n
     * This flag is used when an array of IMAGE type is held in memory as a way to store multiple images. \n
     * When an image is freed, the corresponding memory (in array) is freed and this flag set to zero. \n
     * The active images can be listed by looking for IMAGE[i].used==1 entries.\n
     *
     */
    uint8_t used;

    int32_t shmfd; /**< if shared memory, file descriptor */

    uint64_t memsize; /**< total size in memory if shared    */

    sem_t *semlog; /**< pointer to semaphore for logging  (8 bytes on 64-bit system) */

    IMAGE_METADATA *md;

    
    uint64_t : 0; // align array to 8-byte boundary for speed

    /** @brief data storage array
     *
     * The array is declared as a union, so that multiple data types can be supported \n
     *
     * For 2D image with pixel indices ii (x-axis) and jj (y-axis), the pixel values are stored as array.<TYPE>[ jj * md[0].size[0] + ii ] \n
     * image md[0].size[0] is x-axis size, md[0].size[1] is y-axis size
     *
     * For 3D image with pixel indices ii (x-axis), jj (y-axis) and kk (z-axis), the pixel values are stored as array.<TYPE>[ kk * md[0].size[1] * md[0].size[0] + jj * md[0].size[0] + ii ] \n
     * image md[0].size[0] is x-axis size, md[0].size[1] is y-axis size, md[0].size[2] is z-axis size
     *
     * @note Up to this point, all members of the structure have a fixed memory offset to the start point
     */
    union {
        void *raw;  // raw pointer

        uint8_t *UI8;  // char
        int8_t *SI8;

        uint16_t *UI16;  // unsigned short
        int16_t *SI16;

        uint32_t *UI32;
        int32_t *SI32;  // int

        uint64_t *UI64;
        int64_t *SI64;  // long

        float *F;
        double *D;

        complex_float *CF;
        complex_double *CD;

    } array; /**< pointer to data array */


    sem_t **semptr;                    /**< array of pointers to semaphores   (each 8 bytes on 64-bit system) */

    IMAGE_KEYWORD *kw;

    // PID of process that read shared memory stream
    // Initialized at 0. Otherwise, when process is waiting on semaphore, its PID is written in this array
    // The array can be used to look for available semaphores
    pid_t *semReadPID;

    // PID of the process writing the data
    pid_t *semWritePID;

    uint64_t *flagarray;               /**<  flag for each slice if needed (depends on imagetype) */
    uint64_t *cntarray;                /**< For circular buffer: counter array for circular buffer, copy of cnt0 onto slice index  */
    struct timespec *atimearray;       /**< For each slice index: time at which data was acquires/created. This time CAN be copied from input to output */    
    struct timespec *writetimearray;   /**< For each slice index: time at which data was acquires/created. This time CAN be copied from input to output */   

    // total size is 152 byte = 1216 bit
#ifdef DATA_PACKED
} __attribute__((__packed__)) IMAGE;
#else
} IMAGE;
#endif

#ifdef __cplusplus
}  // extern "C"
#endif

#endif
