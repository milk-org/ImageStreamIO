/**
 * @file    ImageStreamIO.c
 * @brief   Read and Create image
 *
 * Read and create images and streams (shared memory)
 *
 *
 *
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif//_GNU_SOURCE

#include <math.h>
#include <pthread.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <dirent.h>

#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h> // for open
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <semaphore.h>
#include <unistd.h> // for close

#ifdef USE_CFITSIO
#include <fitsio.h>
#endif

// shared memory and semaphores file permission
#define FILEMODE 0666

#if defined NDEBUG
#define DEBUG_TRACEPOINT_LOG(...)
#else
#define DEBUG_TRACEPOINT_LOG(...)             \
    do                                        \
    {                                         \
        char msg[1000];                       \
        snprintf(msg, 1000, __VA_ARGS__);     \
        ImageStreamIO_write_process_log(msg); \
    } while (0)
#endif

// Handle old fitsios
#ifndef ULONGLONG_IMG
#define ULONGLONG_IMG (80)
#endif

#include "ImageStreamIO.h"
#include <stdbool.h>

static int INITSTATUS_ImageStreamIO = 0;

void __attribute__((constructor)) libinit_ImageStreamIO()
{
    if (INITSTATUS_ImageStreamIO == 0)
    {
        init_ImageStreamIO();
        INITSTATUS_ImageStreamIO = 1;
    }
}

errno_t init_ImageStreamIO()
{
    // any initialization needed ?

    return IMAGESTREAMIO_SUCCESS;
}

// Forward dec'l
errno_t ImageStreamIO_printERROR_(const char *file, const char *func, int line,
                                  errno_t code, char *errmessage);
errno_t ImageStreamIO_printWARNING(char *warnmessage);

errno_t (*internal_printError)(const char *, const char *, int, errno_t,
                               char *) = &ImageStreamIO_printERROR_;

errno_t ImageStreamIO_set_default_printError()
{
    internal_printError = &ImageStreamIO_printERROR_;

    return IMAGESTREAMIO_SUCCESS;
}

errno_t ImageStreamIO_set_printError(errno_t (*new_printError)(const char *,
                                     const char *, int, errno_t, char *))
{
    internal_printError = new_printError;

    return IMAGESTREAMIO_SUCCESS;
}

#define ImageStreamIO_printERROR(code, msg) \
    if (internal_printError)                \
        internal_printError(__FILE__, __func__, __LINE__, code, (char*)msg);

#ifdef HAVE_CUDA
void check(cudaError_t result, char const *const func, const char *const file,
           int const line)
{
    if (result)
    {
        cudaDeviceReset();
        // Make sure we call CUDA Device Reset
        ImageStreamIO_printERROR_(file, func, line, result, "CUDA error");
    }
}

// This will output the proper CUDA error strings in the event
// that a CUDA host call returns an error
#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)
#endif



/**
 * @brief Write entry into debug log
 *
 *
 */
errno_t ImageStreamIO_write_process_log(
    char *msg)
{
    FILE *fplog;
    char fname[STRINGMAXLEN_FILE_NAME];
    pid_t thisPID;

    thisPID = getpid();
    snprintf(fname, STRINGMAXLEN_FILE_NAME, "logreport.%05d.log", thisPID);

    struct tm *uttime;
    time_t tvsec0;

    fplog = fopen(fname, "a");
    if (fplog != NULL)
    {
        struct timespec tnow;
        //        time_t now;
        clock_gettime(CLOCK_ISIO, &tnow);
        tvsec0 = tnow.tv_sec;
        uttime = gmtime(&tvsec0);
        fprintf(fplog, "%04d%02d%02dT%02d%02d%02d.%09ld %s\n",
                1900 + uttime->tm_year, 1 + uttime->tm_mon, uttime->tm_mday, uttime->tm_hour,
                uttime->tm_min, uttime->tm_sec, tnow.tv_nsec,
                msg);

        fclose(fplog);
    }

    return 0;
}



/**
 * Print error to stderr
 *
 *
 */
errno_t ImageStreamIO_printERROR_(
    const char *file,
    const char *func,
    int line,
    __attribute__((unused)) errno_t code,
    char *errmessage)
{
    fprintf(stderr,
            "%c[%d;%dmERROR [ FILE: %s   FUNCTION: %s   LINE: %d ]  %c[%d;m\n",
            (char)27, 1, 31, file, func, line, (char)27, 0);
    if (errno != 0)
    {
        char buff[256];

        // Test for which version of strerror_r we're using (XSI or GNU)
#if ((_POSIX_C_SOURCE >= 200112L || _XOPEN_SOURCE >= 600) && \
     !defined(_GNU_SOURCE))
        if (strerror_r(errno, buff, sizeof(buff)) == 0)
        {
            fprintf(stderr, "C Error: %s\n", buff);
        }
        else
        {
            fprintf(stderr, "Unknown C Error\n");
        }
#else
        // GNU strerror_r does not necessarily use buff, and uses errno to report
        // errors.
        int _errno = errno;
        errno = 0;
        char *estr = strerror_r(_errno, buff, sizeof(buff));

        if (errno == 0)
            fprintf(stderr, "%c[%d;%dmC Error: %s%c[%d;m\n", (char)27, 1, 31, estr,
                    27, 0);
        else
            fprintf(stderr, "%c[%d;%dmUnknown C Error%c[%d;m\n", (char)27, 1, 31, 27,
                    0);

        errno = _errno; // restore it in case it's used later.
#endif
    }
    else
        fprintf(stderr, "%c[%d;%dmNo C error (errno = 0)%c[%d;m\n", (char)27, 1, 31,
                27, 0);

    fprintf(stderr, "%c[%d;%dm %s  %c[%d;m\n", (char)27, 1, 31, errmessage,
            (char)27, 0);

    return IMAGESTREAMIO_SUCCESS;
}



/**
 * Print warning to stderr
 *
 *
 */
errno_t ImageStreamIO_printWARNING(
    char *warnmessage)
{
    //int fn = fileno(stderr);

    fprintf(stderr, "%c[%d;%dmWARNING   %c[%d;m\n",
            (char)27, 1, 35, (char)27, 0);
    fprintf(stderr, "%c[%d;%dm (PID %d) %s  %c[%d;m\n", (char)27, 1, 35, (int) getpid(), warnmessage,
            (char)27, 0);

    return IMAGESTREAMIO_SUCCESS;
}




/* ============================================================================================================================================================================================== */
/* @name 0. Utilities */
/* ============================================================================================================================================================================================== */

errno_t ImageStreamIO_readBufferAt(
    const IMAGE *image,
    const unsigned int slice_index,
    void **buffer)
{

    if ((image->md->imagetype & 0xF) != CIRCULAR_BUFFER)
    {
        *buffer = (void *)image->array.UI8;
        return IMAGESTREAMIO_SUCCESS;
    }

    if (slice_index >= image->md->size[2])
    {
        *buffer = NULL;
        return IMAGESTREAMIO_FAILURE;
    }
    const uint64_t frame_size = image->md->size[0] * image->md->size[1];
    const int size_element = ImageStreamIO_typesize(image->md->datatype);
    *buffer = (void *)(image->array.UI8 + slice_index * frame_size * size_element);

    return IMAGESTREAMIO_SUCCESS;
}



errno_t ImageStreamIO_shmdirname(
    char *shmdname)
{
    DIR *tmpdir = NULL;  // Initialize to failure (NULL dir stream)

    // first, we try the env variable if it exists
    char *MILK_SHM_DIR = getenv("MILK_SHM_DIR");
    if (MILK_SHM_DIR != NULL)
    {
        snprintf(shmdname,STRINGMAXLEN_DIR_NAME, "%s", MILK_SHM_DIR);
        // does this direcory exist ?
        tmpdir = opendir(shmdname);
        if (!tmpdir)
        {   // Print warning about envvar if envvar has in valid dir
            printf(" [ WARNING ] '%s' does not exist\n", MILK_SHM_DIR);
        }
    }
    // second, we try SHAREDMEMDIR default
    if (!tmpdir)
    {
        snprintf(shmdname, STRINGMAXLEN_DIR_NAME, "%s", SHAREDMEMDIR);
        tmpdir = opendir(shmdname);
    }
    // if all above fails, set to /tmp
    if (!tmpdir)
    {
        snprintf(shmdname, STRINGMAXLEN_DIR_NAME, "%s", "/tmp");
        tmpdir = opendir(shmdname);
    }
    // Failure:  no directories were found that could be opened
    if (!tmpdir) {
        exit(EXIT_FAILURE);
    }

    // Success:  close directory stream; dirname is in shdname
    closedir(tmpdir);

    return IMAGESTREAMIO_SUCCESS;
}

errno_t ImageStreamIO_filename(
    char *file_name,
    size_t ssz,
    const char *im_name)
{
    static char shmdirname[STRINGMAXLEN_DIR_NAME];
    static int initSHAREDMEMDIR = 0;

    if (initSHAREDMEMDIR == 0)
    {
        ImageStreamIO_shmdirname(shmdirname);
        initSHAREDMEMDIR = 1;
    }

    int rv = snprintf(file_name, ssz, "%s/%s.im.shm", shmdirname, im_name);

    if ((rv > 0) && (rv < (int)ssz)) {
        return IMAGESTREAMIO_SUCCESS;
    }

    if (rv < 0)
    {
        ImageStreamIO_printERROR(IMAGESTREAMIO_FAILURE, strerror(errno));
    }
    else
    {
        ImageStreamIO_printERROR(IMAGESTREAMIO_FAILURE,
                                 "string not large enough for file name");
    }
    return IMAGESTREAMIO_FAILURE;
}


int ImageStreamIO_typesize(
    uint8_t datatype)
{
    switch (datatype)
    {
    case _DATATYPE_UINT8:
        return SIZEOF_DATATYPE_UINT8;
    case _DATATYPE_INT8:
        return SIZEOF_DATATYPE_INT8;
    case _DATATYPE_UINT16:
        return SIZEOF_DATATYPE_UINT16;
    case _DATATYPE_INT16:
        return SIZEOF_DATATYPE_INT16;
    case _DATATYPE_UINT32:
        return SIZEOF_DATATYPE_UINT32;
    case _DATATYPE_INT32:
        return SIZEOF_DATATYPE_INT32;
    case _DATATYPE_UINT64:
        return SIZEOF_DATATYPE_UINT64;
    case _DATATYPE_INT64:
        return SIZEOF_DATATYPE_INT64;
    case _DATATYPE_HALF:
        return SIZEOF_DATATYPE_HALF;
    case _DATATYPE_FLOAT:
        return SIZEOF_DATATYPE_FLOAT;
    case _DATATYPE_DOUBLE:
        return SIZEOF_DATATYPE_DOUBLE;
    case _DATATYPE_COMPLEX_FLOAT:
        return SIZEOF_DATATYPE_COMPLEX_FLOAT;
    case _DATATYPE_COMPLEX_DOUBLE:
        return SIZEOF_DATATYPE_COMPLEX_DOUBLE;
    default:
        break;
    }
    ImageStreamIO_printERROR(IMAGESTREAMIO_INVALIDARG, "invalid type code");
    return -1; // This is an in-band error code, so can't be > 0.
}

const char *ImageStreamIO_typename(
    uint8_t datatype)
{
    switch (datatype)
    {
    case _DATATYPE_UINT8:
        return "UINT8";
    case _DATATYPE_INT8:
        return "INT8";
    case _DATATYPE_UINT16:
        return "UINT16";
    case _DATATYPE_INT16:
        return "INT16";
    case _DATATYPE_UINT32:
        return "UINT32";
    case _DATATYPE_INT32:
        return "INT32";
    case _DATATYPE_UINT64:
        return "UINT64";
    case _DATATYPE_INT64:
        return "INT64";
    case _DATATYPE_HALF:
        return "FLT16";
    case _DATATYPE_FLOAT:
        return "FLT32";
    case _DATATYPE_DOUBLE:
        return "FLT64";
    case _DATATYPE_COMPLEX_FLOAT:
        return "CPLX32";
    case _DATATYPE_COMPLEX_DOUBLE:
        return "CPLX64";
    default:
        break;
    }
    return "unknown";
}

const char *ImageStreamIO_typename_7(
    uint8_t datatype)
{
    switch (datatype)
    {
    case _DATATYPE_UINT8:
        return "UINT8  ";
    case _DATATYPE_INT8:
        return "INT8   ";
    case _DATATYPE_UINT16:
        return "UINT16 ";
    case _DATATYPE_INT16:
        return "INT16  ";
    case _DATATYPE_UINT32:
        return "UINT32 ";
    case _DATATYPE_INT32:
        return "INT32  ";
    case _DATATYPE_UINT64:
        return "UINT64 ";
    case _DATATYPE_INT64:
        return "INT64  ";
    case _DATATYPE_HALF:
        return "FLT16  ";
    case _DATATYPE_FLOAT:
        return "FLOAT  ";
    case _DATATYPE_DOUBLE:
        return "DOUBLE ";
    case _DATATYPE_COMPLEX_FLOAT:
        return "CFLOAT ";
    case _DATATYPE_COMPLEX_DOUBLE:
        return "CDOUBLE";
    default:
        break;
    }
    return "unknown";
}

int ImageStreamIO_checktype(uint8_t datatype, int complex_allowed)
{
    switch (datatype) {
    case _DATATYPE_UINT8:
    case _DATATYPE_INT8:
    case _DATATYPE_UINT16:
    case _DATATYPE_INT16:
    case _DATATYPE_UINT32:
    case _DATATYPE_INT32:
    case _DATATYPE_UINT64:
    case _DATATYPE_INT64:
    case _DATATYPE_HALF:
    case _DATATYPE_FLOAT:
    case _DATATYPE_DOUBLE:
        return 0;

    case _DATATYPE_COMPLEX_FLOAT:
    case _DATATYPE_COMPLEX_DOUBLE:
        return complex_allowed ? 0 : -1;

    default:
        break;
    }
    ImageStreamIO_printERROR(IMAGESTREAMIO_INVALIDARG, "invalid type code");
    return -1; // This is an in-band error code, so can't be > 0.
}

const char *ImageStreamIO_typename_short(
    uint8_t datatype)
{
    switch (datatype)
    {
    case _DATATYPE_UINT8:
        return " UI8";
    case _DATATYPE_INT8:
        return "  I8";
    case _DATATYPE_UINT16:
        return "UI16";
    case _DATATYPE_INT16:
        return " I16";
    case _DATATYPE_UINT32:
        return "UI32";
    case _DATATYPE_INT32:
        return " I32";
    case _DATATYPE_UINT64:
        return "UI64";
    case _DATATYPE_INT64:
        return " I64";
    case _DATATYPE_HALF:
        return " F16";
    case _DATATYPE_FLOAT:
        return " FLT";
    case _DATATYPE_DOUBLE:
        return " DBL";
    case _DATATYPE_COMPLEX_FLOAT:
        return "CFLT";
    case _DATATYPE_COMPLEX_DOUBLE:
        return "CDBL";
    default:
        break;
    }
    return " ???";
}

int ImageStreamIO_floattype(
    uint8_t datatype)
{
    switch (datatype)
    {
    case _DATATYPE_UINT8:
        return _DATATYPE_FLOAT;
    case _DATATYPE_INT8:
        return _DATATYPE_FLOAT;
    case _DATATYPE_UINT16:
        return _DATATYPE_FLOAT;
    case _DATATYPE_INT16:
        return _DATATYPE_FLOAT;
    case _DATATYPE_UINT32:
        return _DATATYPE_FLOAT;
    case _DATATYPE_INT32:
        return _DATATYPE_FLOAT;
    case _DATATYPE_UINT64:
        return _DATATYPE_DOUBLE;
    case _DATATYPE_INT64:
        return _DATATYPE_DOUBLE;
    case _DATATYPE_HALF:
        return _DATATYPE_HALF;
    case _DATATYPE_FLOAT:
        return _DATATYPE_FLOAT;
    case _DATATYPE_DOUBLE:
        return _DATATYPE_DOUBLE;
    case _DATATYPE_COMPLEX_FLOAT:
        return _DATATYPE_COMPLEX_FLOAT;
    case _DATATYPE_COMPLEX_DOUBLE:
        return _DATATYPE_COMPLEX_DOUBLE;
    default:
        break;
    }
    ImageStreamIO_printERROR(IMAGESTREAMIO_INVALIDARG, "invalid type code");
    return -1; // This is an in-band error code, so can't be > 0.
}

int ImageStreamIO_FITSIOdatatype(uint8_t datatype)
{
    switch (datatype)
    {
#ifdef USE_CFITSIO
    case _DATATYPE_UINT8:
        return TBYTE;
    case _DATATYPE_INT8:
        return TSBYTE;
    case _DATATYPE_UINT16:
        return TUSHORT;
    case _DATATYPE_INT16:
        return TSHORT;
    case _DATATYPE_UINT32:
        return TUINT;
    case _DATATYPE_INT32:
        return TINT;
    case _DATATYPE_UINT64:
        return TULONG;
    case _DATATYPE_INT64:
        return TLONG;
    case _DATATYPE_FLOAT:
        return TFLOAT;
    case _DATATYPE_DOUBLE:
        return TDOUBLE;
#endif
    default:
        break;
    }
    ImageStreamIO_printERROR(IMAGESTREAMIO_INVALIDARG,
                             "bitpix not implemented for type");
    return -1; // This is an in-band error code, must be unique from valid BITPIX values.
}

int ImageStreamIO_FITSIObitpix(
    uint8_t datatype)
{
    switch (datatype)
    {
#ifdef USE_CFITSIO
    case _DATATYPE_UINT8:
        return BYTE_IMG;
    case _DATATYPE_INT8:
        return SBYTE_IMG;
    case _DATATYPE_UINT16:
        return USHORT_IMG;
    case _DATATYPE_INT16:
        return SHORT_IMG;
    case _DATATYPE_UINT32:
        return ULONG_IMG;
    case _DATATYPE_INT32:
        return LONG_IMG;
    case _DATATYPE_UINT64:
        return ULONGLONG_IMG;
    case _DATATYPE_INT64:
        return LONGLONG_IMG;
    case _DATATYPE_FLOAT:
        return FLOAT_IMG;
    case _DATATYPE_DOUBLE:
        return DOUBLE_IMG;
#endif
    default:
        break;
    }
    ImageStreamIO_printERROR(IMAGESTREAMIO_INVALIDARG,
                             "bitpix not implemented for type");
    return -1; // This is an in-band error code, must be unique from valid BITPIX values.
}



uint64_t ImageStreamIO_offset_data(
    IMAGE *image,
    void *map)
{
    uint8_t datatype = image->md->datatype;
    u_int64_t offset = 0;

    // printf("datatype = %d\n", (int)datatype);
    // fflush(stdout);

    if (image->md->location >= 0)
    {
        image->array.raw = ImageStreamIO_get_image_d_ptr(image);
        offset = 0;
    }
    else
    {
        image->array.raw = map;
        offset = ImageStreamIO_typesize(datatype) * image->md->nelement;
    }

    return offset;
}



uint64_t ImageStreamIO_initialize_buffer(
    IMAGE *image)
{
    // void *map;  // pointed cast in bytes
    const size_t size_element = ImageStreamIO_typesize(image->md->datatype);

    if (image->md->location == -1)
    {
        if (image->md->shared == 1)
        {
            memset(image->array.raw, 0, image->md->nelement * size_element);
        }
        else
        {
            image->array.raw = calloc((size_t)image->md->nelement, size_element);
            if (image->array.raw == NULL)
            {
                ImageStreamIO_printERROR(IMAGESTREAMIO_BADALLOC, "memory allocation failed");
                fprintf(stderr, "%c[%d;%dm", (char)27, 1, 31);
                fprintf(stderr, "Image name = %s\n", image->name);
                fprintf(stderr, "Image size = ");
                fprintf(stderr, "%ld", (long)image->md->size[0]);
                int i;
                for (i = 1; i < image->md->naxis; i++)
                {
                    fprintf(stderr, "x%ld", (long)image->md->size[i]);
                }
                fprintf(stderr, "\n");
                fprintf(stderr, "Requested memory size = %ld elements = %f Mb\n",
                        (long)image->md->nelement,
                        1.0 / 1024 / 1024 * image->md->nelement * sizeof(uint8_t));
                fprintf(stderr, " %c[%d;m", (char)27, 0);
                exit(EXIT_FAILURE); ///\todo Is this really an exit or should we return?
            }
        }
    }
    else if (image->md->location >= 0)
    {
#ifdef HAVE_CUDA
        checkCudaErrors(cudaSetDevice(image->md->location));
        checkCudaErrors(
            cudaMalloc(&image->array.raw, size_element * image->md->nelement));
        if (image->md->shared == 1)
        {
            checkCudaErrors(
                cudaIpcGetMemHandle(&image->md->cudaMemHandle, image->array.raw));
        }
#else
        ImageStreamIO_printERROR(IMAGESTREAMIO_NOTIMPL,
                                 "unsupported location, milk needs to be compiled with -DUSE_CUDA=ON"); ///\todo should this return an error?
#endif
    }

    return ImageStreamIO_offset_data(image, image->array.raw);
} // uint64_t ImageStreamIO_initialize_buffer(IMAGE *image)



/**
 * @brief Get inode using shmim file descriptor, write inode to IMAGE
 *
 *
 */
errno_t
ImageStreamIO_store_image_inode(IMAGE* image)
{
    // - Retrieve status of file referenced by FD; close file on error
    struct stat file_stat;
    if (fstat(image->shmfd, &file_stat) < 0)
    {
        close(image->shmfd);
        ImageStreamIO_printERROR(IMAGESTREAMIO_INODE, "Error getting inode");
        return IMAGESTREAMIO_INODE;
    }
    // - Save inode as metadata, which metadata are also in the file
    image->md->inode = file_stat.st_ino;
    return IMAGESTREAMIO_SUCCESS;
}



/**
 * @brief Check image->md->inode against inode from shmim name
 *
 * \returns IMAGESTREAMIO_SUCCESS if ->md->inode matches the shmim inode
 * \returns _INODE if ->md->inode doesn't match the shmim name inode
 * \returns _FAILURE if the shmim name inode could not be retrieved
 *
 */
errno_t
ImageStreamIO_check_image_inode(IMAGE* image)
{
    // - Build filename from shmim name image->md->name
    char SM_fname[STRINGMAXLEN_FILE_NAME] = {0};
    if (IMAGESTREAMIO_SUCCESS
            != ImageStreamIO_filename(SM_fname, sizeof(SM_fname), image->md->name))
    {
        return IMAGESTREAMIO_FAILURE;  // _filename did _printERROR
    }

    // - Retrieve status of file referenced by SM_fname
    struct stat file_stat;
    if (stat(SM_fname, &file_stat) < 0)
    {
        ImageStreamIO_printERROR(IMAGESTREAMIO_INODE, "Error getting inode");
        return IMAGESTREAMIO_FAILURE;
    }

    // - Return success or failure if inode matches or not, respectively
    return image->md->inode == file_stat.st_ino ? IMAGESTREAMIO_SUCCESS : IMAGESTREAMIO_INODE;
}




/* ===============================================================================================
 */
/* ===============================================================================================
 */
/* @name 1. READ / WRITE STREAM
 *
 */
/* ===============================================================================================
 */
/* ===============================================================================================
 */

errno_t ImageStreamIO_createIm(
    IMAGE *image,
    const char *name,
    long naxis,
    uint32_t *size,
    uint8_t datatype,
    int shared,
    int NBkw,
    int CBsize)
{
    return ImageStreamIO_createIm_gpu(image, name, naxis, size, datatype, -1,
                                      shared, IMAGE_NB_SEMAPHORE, NBkw,
                                      MATH_DATA, (uint32_t)CBsize);
}

errno_t ImageStreamIO_image_sizing(
    IMAGE *image,
    uint8_t* map)
{
    // Error checking
    if (!image)
    {
        ImageStreamIO_printERROR(IMAGESTREAMIO_INVALIDARG,
                                 "Error calling ImageStreamIO_image_sizing, "
                                 "null IMAGE pointer");
        return IMAGESTREAMIO_INVALIDARG;
    }
    if (!image->md)
    {
        ImageStreamIO_printERROR(IMAGESTREAMIO_INVALIDARG,
                                 "Error calling ImageStreamIO_image_sizing, "
                                 "null IMAGE_METADATA pointer");
        return IMAGESTREAMIO_INVALIDARG;
    }
    if (!*image->md->name)
    {
        ImageStreamIO_printERROR(IMAGESTREAMIO_INVALIDARG,
                                 "Error calling ImageStreamIO_image_sizing, "
                                 "invalid shmim name, or null pointer to same");
        return IMAGESTREAMIO_INVALIDARG;
    }
    switch (image->md->naxis)
    {
    case 3:
        if (image->md->size[2] < 1)
        {
            ImageStreamIO_printERROR(IMAGESTREAMIO_INVALIDARG,
                                     "Error calling ImageStreamIO_image_sizing, "
                                     "invalid size of third axis");
            return IMAGESTREAMIO_INVALIDARG;
        }
    // N.B. no break, fall through to previous axis
    case 2:
        if (image->md->size[1] < 1)
        {
            ImageStreamIO_printERROR(IMAGESTREAMIO_INVALIDARG,
                                     "Error calling ImageStreamIO_image_sizing, "
                                     "invalid size of second axis");
            return IMAGESTREAMIO_INVALIDARG;
        }
    // N.B. no break, fall through to previous axis
    case 1:
        if (image->md->size[0] < 1)
        {
            ImageStreamIO_printERROR(IMAGESTREAMIO_INVALIDARG,
                                     "Error calling ImageStreamIO_image_sizing, "
                                     "invalid size of first axis");
            return IMAGESTREAMIO_INVALIDARG;
        }
        // N.B. break; all axes indicated by naxis are valid
        break;
    default:
        ImageStreamIO_printERROR(IMAGESTREAMIO_INVALIDARG,
                                 "Error calling ImageStreamIO_image_sizing, "
                                 "invalid number of axes");
        return IMAGESTREAMIO_INVALIDARG;
    }
    if (image->md->location < -1)
    {
        ///\todo should error code differ between printERROR and return?
        ImageStreamIO_printERROR(IMAGESTREAMIO_INVALIDARG,
                                 "Error calling ImageStreamIO_image_sizing, "
                                 "unknown location");
        return IMAGESTREAMIO_FAILURE;
    }
    if (ImageStreamIO_checktype(image->md->datatype, true))
    {
        ImageStreamIO_printERROR(IMAGESTREAMIO_INVALIDARG,
                                 "Error calling ImageStreamIO_image_sizing, "
                                 "invalid datatype");
        return IMAGESTREAMIO_INVALIDARG;
    }
    if (((image->md->imagetype & 0xF000F) == CIRCULAR_BUFFER) &&
            (image->md->naxis != 3))
    {
        ImageStreamIO_printERROR(IMAGESTREAMIO_INVALIDARG,
                                 "Error calling ImageStreamIO_image_sizing, "
                                 "temporal circular buffer needs 3 dimensions");
        return IMAGESTREAMIO_INVALIDARG;
    }

    // compute total shared size of shmim from IMAGE_METADATA parameters
    if (image->md->shared == 1)
    {
        image->memsize = sizeof(IMAGE_METADATA);
        map += sizeof(IMAGE_METADATA);
        // image->md will be assigned elsewhere

        if (image->md->location == -1)
        {
            // image on CPU
            // printf("shared memory space in CPU RAM = %ud bytes\n", image->memsize);
            image->array.raw = map;
            map            += image->md->imdatamemsize;
            image->memsize += image->md->imdatamemsize;
        }
        else
        {
            // GPU - array pointer will be assigned later via ImageStreamIO_get_img_d_ptr(...)
            image->array.raw = NULL;
        }

        strncpy(image->name, image->md->name, STRINGMAXLEN_IMAGE_NAME); // local name
        // Ensure image and image metadata names are null-terminated
        image->name[STRINGMAXLEN_IMAGE_NAME-1] = '\0';

        image->kw       = (IMAGE_KEYWORD *)(map);
        map            += sizeof(IMAGE_KEYWORD) * image->md->NBkw;
        image->memsize += sizeof(IMAGE_KEYWORD) * image->md->NBkw;


        image->semfile  = (SEMFILEDATA*)(map);
        map            += sizeof(SEMFILEDATA) * image->md->sem;
        image->memsize += sizeof(SEMFILEDATA) * image->md->sem;

        image->semlog   = (sem_t *)(map);
        map            += sizeof(sem_t);
        image->memsize += sizeof(sem_t); // for semlog

        // one read PID array, one write PID array
        image->semReadPID = (pid_t *)(map);
        map            += sizeof(pid_t) * image->md->sem;
        image->memsize += sizeof(pid_t) * image->md->sem;

        image->semWritePID = (pid_t *)(map);
        map            += sizeof(pid_t) * image->md->sem;
        image->memsize += sizeof(pid_t) * image->md->sem;

        // semctrl
        image->semctrl  = (uint32_t *)(map);
        map            += sizeof(uint32_t) * image->md->sem;
        image->memsize += sizeof(uint32_t) * image->md->sem;

        // semstatus
        image->semstatus = (uint32_t *)(map);
        map            += sizeof(uint32_t) * image->md->sem;
        image->memsize += sizeof(uint32_t) * image->md->sem;

        image->streamproctrace = (STREAM_PROC_TRACE *)(map);
        map            += sizeof(STREAM_PROC_TRACE) * image->md->NBproctrace;
        image->memsize += sizeof(STREAM_PROC_TRACE) * image->md->NBproctrace;

        if ((image->md->imagetype & 0xF000F) ==
                (CIRCULAR_BUFFER | ZAXIS_TEMPORAL)) // Circular buffer
        {
            image->atimearray = (struct timespec *)(map);
            map            += sizeof(struct timespec) * image->md->size[2];
            image->memsize += sizeof(struct timespec) * image->md->size[2];

            image->writetimearray = (struct timespec *)(map);
            map            += sizeof(struct timespec) * image->md->size[2];
            image->memsize += sizeof(struct timespec) * image->md->size[2];

            image->cntarray = (uint64_t *)(map);
            map            += sizeof(uint64_t) * image->md->size[2];
            image->memsize += sizeof(uint64_t) * image->md->size[2];
        }

        // fast circular buffer metadata
        image->CircBuff_md = (CBFRAMEMD *)(map);
        map            += sizeof(CBFRAMEMD) * image->md->CBsize;
        image->memsize += sizeof(CBFRAMEMD) * image->md->CBsize;

        // fast circular buffer data buffer
        if (image->md->CBsize > 0)
        {
            image->CBimdata = map;
            map            += image->md->imdatamemsize * image->md->CBsize;
            image->memsize += image->md->imdatamemsize * image->md->CBsize;
        }
        else
        {
            image->CBimdata = NULL;
        }

#ifdef IMAGESTRUCT_WRITEHISTORY
        // write time buffer
        image->writehist = (FRAMEWRITEMD *)(map);
        map            += sizeof(FRAMEWRITEMD) * IMAGESTRUCT_FRAMEWRITEMDSIZE;
        image->memsize += sizeof(FRAMEWRITEMD) * IMAGESTRUCT_FRAMEWRITEMDSIZE;
#endif

    } // if (image->md->shared == 1)

    return IMAGESTREAMIO_SUCCESS;
} // errno_t ImageStreamIO_image_sizing(IMAGE *image, uint8_t* map)


errno_t ImageStreamIO_image_sizing_from_scratch(
    IMAGE *image,
    const char *name,
    long naxis,
    uint32_t *size,
    uint8_t datatype,
    int8_t location, // -1: CPU RAM, 0+ : GPU
    int shared,
    int NBsem,
    int NBkw,
    uint64_t imagetype,
    uint32_t CBsize, // circular buffer size (if shared), 0 if not used
    uint8_t* map)
{
    int NBproctrace = IMAGE_NB_PROCTRACE;
    IMAGE_METADATA local_metadata = { 0 };
    image->md = image->md ? image->md : &local_metadata;

    image->md->naxis = naxis;
    image->md->size[0] = naxis>0 ? size[0] : 0;
    image->md->size[1] = naxis>1 ? size[1] : 0;
    image->md->size[2] = naxis>2 ? size[2] : 0;
    image->md->datatype = datatype;
    image->md->location = location;
    image->md->shared = shared;
    image->md->sem = NBsem;
    image->md->NBkw = NBkw;
    image->md->imagetype = imagetype;
    image->md->CBsize = CBsize;
    image->md->NBproctrace = NBproctrace;

    image->md->nelement = image->md->size[0];
    for (long i = 1; i < image->md->naxis; ++i)
    {
        image->md->nelement *= image->md->size[i];
    }

    image->md->imdatamemsize = ImageStreamIO_typesize(image->md->datatype)
                               * image->md->nelement;

    if (name)
    {
        strncpy(image->md->name, name, STRINGMAXLEN_IMAGE_NAME);
        image->md->name[STRINGMAXLEN_IMAGE_NAME-1] = '\0';
    }
    else
    {
        // This will throw error in the ImageStreamIO_image_sizing call
        *image->md->name = '\0';
    }

    return ImageStreamIO_image_sizing(image, map);
} // errno_t ImageStreamIO_image_sizing_from_scratch(...)



errno_t ImageStreamIO_createIm_gpu(
    IMAGE *image,
    const char *name,
    long naxis,
    uint32_t *size,
    uint8_t datatype,
    int8_t location, // -1: CPU RAM, 0+ : GPU
    int shared,
    int NBsem,
    int NBkw,
    uint64_t imagetype,
    uint32_t CBsize) // circular buffer size (if shared), 0 if not used
{
    uint8_t *map = NULL;

    // compute total size to be allocated
    if (shared == 1)
    {

        ////////////////////////////////////////////////////////////////
        // Calculate the size of the shmim file (image->memsize)
        ////////////////////////////////////////////////////////////////

        errno_t ierrno = ImageStreamIO_image_sizing_from_scratch(
                             image, name, naxis, size, datatype
                             , location, shared, NBsem, NBkw
                             , imagetype, CBsize, (uint8_t*) NULL
                         );
        if (ierrno != IMAGESTREAMIO_SUCCESS) {
            return ierrno;
        }

        ////////////////////////////////////////////////////////////////
        // Open and map shmim file of the calculated size image->memsize
        ////////////////////////////////////////////////////////////////

        char SM_fname[STRINGMAXLEN_FILE_NAME] = {0};
        if (IMAGESTREAMIO_SUCCESS
                != ImageStreamIO_filename(SM_fname, sizeof(SM_fname), name))
        {
            return IMAGESTREAMIO_FAILURE;  // _filename did _printERROR
        }

        // - Ensure GPU SHM buffer file does not exist
        struct stat buffer;
        if ((stat(SM_fname, &buffer) == 0) && (location > -1))
        {
            ImageStreamIO_printERROR(IMAGESTREAMIO_FILEEXISTS,
                                     "Error creating GPU SHM buffer on an existing file");
            return IMAGESTREAMIO_FILEEXISTS;
        }

        // - Create and open shmim file as a new, empty (truncated) file
        //   - image->shmfd stores the shared memory file descriptor
        umask(0);
        errno = 0;
        image->shmfd = open(SM_fname
                            // (O_CREAT|O_EXCL) flags force new file
                            , O_RDWR | O_CREAT | O_EXCL | O_TRUNC
                            , (mode_t)FILEMODE
                           );
        if (image->shmfd == -1 && errno == EEXIST)
        {
            // - File was not created:  a file exists at path SM_fname;
            unlink(SM_fname);  // - unlink that file from its directory;
            errno = 0;         // - ignore any error from unlink;
            image->shmfd = open(SM_fname  // - and try again ...
                                , O_RDWR | O_CREAT | O_EXCL | O_TRUNC
                                , (mode_t)FILEMODE
                               );
        }
        if (image->shmfd == -1)
        {
            ImageStreamIO_printERROR(IMAGESTREAMIO_FILEOPEN,
                                     "Error opening file for writing");
            return IMAGESTREAMIO_FILEOPEN;
        }

        // - Seek to the end of the currently empty shmim file, ...
        if (lseek(image->shmfd, image->memsize - 1, SEEK_SET)
                != (off_t)(image->memsize-1))
        {
            close(image->shmfd);
            ImageStreamIO_printERROR(IMAGESTREAMIO_FILESEEK,
                                     "Error calling lseek() to 'stretch' the file");
            return IMAGESTREAMIO_FILESEEK;
        }

        // - ... then write a null at that sought position
        if (write(image->shmfd, "", 1) != (ssize_t)1)
        {
            close(image->shmfd);
            ImageStreamIO_printERROR(IMAGESTREAMIO_FILEWRITE,
                                     "Error writing last byte of the file");
            return IMAGESTREAMIO_FILEWRITE;
        }

        // - Map the file into memory, address will be pointed to by map
        map = (uint8_t *)mmap(0, image->memsize, PROT_READ | PROT_WRITE, MAP_SHARED,
                              image->shmfd, 0);
        if (map == MAP_FAILED)
        {
            close(image->shmfd);
            ImageStreamIO_printERROR(IMAGESTREAMIO_MMAP, "Error mmapping the file");
            return IMAGESTREAMIO_MMAP;
        }
        image->md = (IMAGE_METADATA *)map;
    }
    else
    {
        // not shared memory, local memory only
        image->shmfd = 0;
        image->memsize = 0;

        image->md = (IMAGE_METADATA *)malloc(sizeof(IMAGE_METADATA));
        if (image->md == NULL)
        {
            printf("Memory allocation error %s %d\n", __FILE__, __LINE__);
            abort();
        }
        image->md->shared = 0;
        image->md->inode = 0;
    }

    ////////////////////////////////////////////////////////////////
    // Load IMAGE_METADATA struct at address map with all parameters
    // and calculate sizes and address image-> pointers
    ////////////////////////////////////////////////////////////////
    errno_t ierrno = ImageStreamIO_image_sizing_from_scratch(
                         image, name, naxis, size, datatype
                         , location, shared, NBsem, NBkw
                         , imagetype, CBsize, map
                     );
    if (ierrno != IMAGESTREAMIO_SUCCESS) {
        return ierrno;
    }

    if (shared == 1)
    {
        // - Store the inode of the shmim flle into image->md->inode
        //   - On error, shmim will have been closed; return
        ierrno = ImageStreamIO_store_image_inode(image);
        if (ierrno != IMAGESTREAMIO_SUCCESS) {
            return ierrno;
        }
    }

    if(shared != 1)
    {
        strncpy(image->name, image->md->name, STRINGMAXLEN_IMAGE_NAME); // local name
        // Ensure image and image metadata names are null-terminated
        image->name[STRINGMAXLEN_IMAGE_NAME-1] = '\0';
    }

    image->md->creatorPID = getpid();
    image->md->ownerPID = 0; // default value, indicates unset

    image->md->CBsize = 0;
    image->md->CBindex = 0;
    image->md->CBcycle = 0;

#ifdef IMAGESTRUCT_WRITEHISTORY
    image->md->wCBindex = 0;
    image->md->wCBcycle = 0;
#endif





    if (image->md->NBkw > 0)
    {
        image->kw = (IMAGE_KEYWORD *)malloc(sizeof(IMAGE_KEYWORD) * image->md->NBkw);
        if (image->kw == NULL)
        {
            printf("Memory allocation error %s %d\n", __FILE__, __LINE__);
            abort();
        }
    }
    else
    {
        image->kw = NULL;
    }




    strncpy(image->md->version, IMAGESTRUCT_VERSION, 32);

    ImageStreamIO_initialize_buffer(image);

    clock_gettime(CLOCK_ISIO, &image->md->lastaccesstime);
    clock_gettime(CLOCK_ISIO, &image->md->creationtime);

    image->md->write = 0;
    image->md->cnt0 = 0;
    image->md->cnt1 = 0;

    if (shared == 1)
    {
        // - Allocate space for semaphore pointers;
        //   semaphore data are in the shmim
        image->semptr = (sem_t **)malloc(sizeof(sem_t **) * image->md->sem);
        if (image->semptr == NULL)
        {
            printf("Memory allocation error %s %d\n", __FILE__, __LINE__);
            abort();
        }

        // - Assign pointers; initialize the semphores and their data
        for (int semindex = 0; semindex < image->md->sem; semindex++)
        {
            image->semptr[semindex] = &image->semfile[semindex].semdata;
            sem_init(
                image->semptr[semindex], 1,
                SEMAPHORE_INITVAL); // SEMAPHORE_INITVAL defined in ImageStruct.h
            image->semReadPID[semindex] = -1;
            image->semWritePID[semindex] = -1;
            image->semctrl[semindex] = 0;
            image->semstatus[semindex] = 0;
        }

        for (int proctraceindex = 0; proctraceindex < image->md->NBproctrace; proctraceindex++)
        {
            image->streamproctrace[proctraceindex].procwrite_PID = 0;
            image->streamproctrace[proctraceindex].trigger_inode = 0;
            image->streamproctrace[proctraceindex].ts_procstart.tv_sec = 0;
            image->streamproctrace[proctraceindex].ts_procstart.tv_nsec = 0;
            image->streamproctrace[proctraceindex].ts_streamupdate.tv_sec = 0;
            image->streamproctrace[proctraceindex].ts_streamupdate.tv_nsec = 0;
            image->streamproctrace[proctraceindex].trigsemindex = -1;
            image->streamproctrace[proctraceindex].cnt0 = 0;
        }
    }
    else
    {
        image->md->sem = 0; // no shmim => semaphores
    }

// initialize keywords
    for (int kw = 0; kw < image->md->NBkw; kw++)
    {
        image->kw[kw].type = 'N';
    }

    image->used = 1;
    image->createcnt++;

// DEBUG_TRACEPOINT_LOG("%s %d image->md->sem = %d", __FILE__, __LINE__, image->md->sem);

    return IMAGESTREAMIO_SUCCESS;
} // errno_t ImageStreamIO_createIm_gpu(...)





errno_t ImageStreamIO_destroyIm(
    IMAGE *image)
{
    if(image->used == 1)
    {
        if (image->semptr)
        {
            for (int semindex=0; semindex<image->md->sem; ++semindex)
            {
                sem_destroy(image->semptr[semindex]);
            }
            free(image->semptr);
            image->semptr = NULL;
        }
        if (image->semlog)
        {
            sem_destroy(image->semlog);
            image->semlog = NULL;
        }

        if (image->memsize > 0)
        {
            char fname[512];
            close(image->shmfd);
            // Get this before unmapping.
            ImageStreamIO_filename(fname, sizeof(fname), image->md->name);
            munmap(image->md, image->memsize);
            image->md = NULL;
            image->kw = NULL;
            // Remove the file
            remove(fname);
        }
        else
        {
            free(image->array.UI8);
        }
        image->array.UI8 = NULL;

        if (image->md != NULL)
        {
            free(image->md);
            image->md = NULL;
        }

        image->kw = NULL;

        image->used = 0;
    }

    return IMAGESTREAMIO_SUCCESS;
}





errno_t ImageStreamIO_openIm(
    IMAGE *image,
    const char *name)
{
    return ImageStreamIO_read_sharedmem_image_toIMAGE(name, image);
}

void *ImageStreamIO_get_image_d_ptr(
    IMAGE *image)
{
    if (image->array.raw != NULL)
    {
        return image->array.raw;
    }

    void *d_ptr = NULL;
    if (image->md->location >= 0)
    {
#ifdef HAVE_CUDA
        checkCudaErrors(cudaSetDevice(image->md->location));
        checkCudaErrors(cudaIpcOpenMemHandle(&d_ptr, image->md->cudaMemHandle,
                                             cudaIpcMemLazyEnablePeerAccess));
#else
        ImageStreamIO_printERROR(IMAGESTREAMIO_NOTIMPL,
                                 "Error calling ImageStreamIO_get_image_d_ptr(), CACAO needs to be "
                                 "compiled with -DUSE_CUDA=ON"); ///\todo should this return a NULL?
#endif
    }
    else
    {
        ImageStreamIO_printERROR(IMAGESTREAMIO_NOTIMPL,
                                 "Error calling ImageStreamIO_get_image_d_ptr(), wrong location"); ///\todo should this return a NULL?
    }
    return d_ptr;
}

/**
 * ## Purpose
 *
 * Open existing shared memory image\n
 *
 *
 *
 * ## Details
 *
 */

errno_t ImageStreamIO_read_sharedmem_image_toIMAGE(
    const char *name,
    IMAGE *image)
{
    int SM_fd;
    char SM_fname[STRINGMAXLEN_FILE_NAME] = {0};

    // Build the shmim pathname
    errno_t ierrno;
    ierrno = ImageStreamIO_filename(SM_fname, sizeof(SM_fname), name);
    if (IMAGESTREAMIO_SUCCESS != ierrno)
    {
        image->used = 0;
        char wmsg[STRINGMAXLEN_IMAGE_NAME+50];
        snprintf(wmsg, sizeof(wmsg), "Cannot build file name from \"%s\"\n", name);
        ImageStreamIO_printWARNING(wmsg);
        return IMAGESTREAMIO_FILEOPEN;
    }

    // Open the shmim file
    SM_fd = open(SM_fname, O_RDWR);
    if (SM_fd == -1)
    {
        image->used = 0;
        char wmsg[STRINGMAXLEN_FILE_NAME+50];
        snprintf(wmsg, sizeof(wmsg), "Cannot open shm file \"%s\"\n", SM_fname);
        ImageStreamIO_printWARNING(wmsg);
        return IMAGESTREAMIO_FILEOPEN;
    }
    // open() was successful. We'll need to close SM_fd for any failed exit

    struct stat file_stat = {0};
    int tenths_timeout = 0;

    do // ensure shmim file size is adequate (greater than size of IMAGE_METADATA)
    {
        if (tenths_timeout > 99) // Fail after 100 tries (~10s)
        {
            close(SM_fd);
            ImageStreamIO_printERROR(IMAGESTREAMIO_FILEOPEN, "Error in the file (too small)");
            return IMAGESTREAMIO_FILEOPEN;
        }
        if (tenths_timeout++) {
            usleep(100000);    // wait 0.1s
        }
        file_stat.st_size = 0;
        fstat(SM_fd, &file_stat);
    } while (file_stat.st_size <= sizeof(IMAGE_METADATA));

    // printf("File %s size: %zd\n", SM_fname, file_stat.st_size); fflush(stdout); //TEST

    uint8_t *map_root = (uint8_t *)mmap(0, file_stat.st_size, PROT_READ | PROT_WRITE,
                                        MAP_SHARED, SM_fd, 0);
    if (map_root == MAP_FAILED)
    {
        ImageStreamIO_printERROR(IMAGESTREAMIO_MMAP, "Error mmapping the file");
        close(SM_fd);
        return IMAGESTREAMIO_MMAP;
    }
    // mmap() was successful. We'll need to unmap image->md for any failed exit

    image->md = (IMAGE_METADATA *)map_root;

    ierrno = ImageStreamIO_image_sizing(image, map_root);
    if (IMAGESTREAMIO_SUCCESS != ierrno || image->memsize != file_stat.st_size)
    {
        ImageStreamIO_printERROR(IMAGESTREAMIO_FILEOPEN, "Error in the file");
        munmap(image->md, file_stat.st_size);
        close(SM_fd);
        return IMAGESTREAMIO_FILEOPEN;
    }

    // Check the shmim version
    if (strcmp(image->md->version, IMAGESTRUCT_VERSION))
    {
        char errmsg[200];
        snprintf(errmsg, 200, "Stream %s corrupted, or incompatible version. Should be %s",
                 name, IMAGESTRUCT_VERSION);
        ImageStreamIO_printERROR(IMAGESTREAMIO_VERSION, errmsg);
        munmap(image->md, image->memsize);
        close(image->shmfd);
        return IMAGESTREAMIO_VERSION;
    }

    // some verification
    if (image->md->nelement > 10000000000)
    {
        printf("IMAGE \"%s\" SEEMS BIG... NOT LOADING\n", image->md->name);
        fflush(stdout);
        munmap(image->md, image->memsize);
        close(image->shmfd);
        return IMAGESTREAMIO_FAILURE;
    }

    // gain image data array pointer
    if (image->md->location >= 0)
    {
        ImageStreamIO_offset_data(image, image->array.raw);
    }
    if (image->array.raw == NULL)
    {
        printf("Fail to retrieve data pointer\n");
        fflush(stdout);
        munmap(image->md, image->memsize);
        close(SM_fd);
        return IMAGESTREAMIO_FAILURE;
    }

    image->semptr = (sem_t **)malloc(sizeof(sem_t **) * image->md->sem);
    if (image->semptr == NULL)
    {
        printf("Memory allocation error %s %d\n", __FILE__, __LINE__);
        munmap(image->md, image->memsize);
        close(SM_fd);
        abort();
    }
    for (long semindex = 0; semindex < image->md->sem; semindex++)
    {
        image->semptr[semindex] = &image->semfile[semindex].semdata;
    }

    return IMAGESTREAMIO_SUCCESS;
} // errno_t ImageStreamIO_read_sharedmem_image_toIMAGE(const char *name, IMAGE *image)








errno_t ImageStreamIO_closeIm(
    IMAGE *image)
{
    free(image->semptr);

    if (munmap(image->md, image->memsize) != 0)
    {
        ImageStreamIO_printERROR(IMAGESTREAMIO_MMAP, "error unmapping memory");
        return IMAGESTREAMIO_MMAP;
    }

    close(image->shmfd);

    return IMAGESTREAMIO_SUCCESS;
}

/* ===============================================================================================
 */
/* ===============================================================================================
 */
/* @name 2. MANAGE SEMAPHORES
 *
 */
/* ===============================================================================================
 */
/* ===============================================================================================
 */

/**
 * ## Purpose
 *
 * Posts semaphore of a shmim
 * if index < 0, post all semaphores
 *
 * ## Arguments
 *
 * @param[in]
 * image	IMAGE*
 * 			pointer to shmim
 *
 * @param[in]
 * index    semaphore index
 * 			index of semaphore to be posted
 *          if index=-1, post all semaphores
 **/
long ImageStreamIO_sempost(
    IMAGE *image,
    long index)
{
    pid_t writeProcessPID;

    writeProcessPID = getpid();

    if (index < 0)
    {
        long semindex;

        for (semindex = 0; semindex < image->md->sem; semindex++)
        {
            int semval;

            image->semWritePID[semindex] = writeProcessPID;

            sem_getvalue(image->semptr[semindex], &semval);
            if (semval < SEMAPHORE_MAXVAL)
            {
                sem_post(image->semptr[semindex]);
            }
        }
    }
    else
    {
        if (index > image->md->sem - 1)
            printf("ERROR: image %s semaphore # %ld does no exist\n", image->md->name,
                   index);
        else
        {
            int semval;

            sem_getvalue(image->semptr[index], &semval);
            if (semval < SEMAPHORE_MAXVAL)
            {
                sem_post(image->semptr[index]);
                image->semWritePID[index] = writeProcessPID;
            }
        }
    }

    if (image->semlog != NULL)
    {
        int semval;

        sem_getvalue(image->semlog, &semval);
        if (semval < SEMAPHORE_MAXVAL)
        {
            sem_post(image->semlog);
        }
    }

    return IMAGESTREAMIO_SUCCESS;
}

/**
 * ## Purpose
 *
 * Posts all semaphores of a shmim except one
 *
 * ## Arguments
 *
 * @param[in]
 * image	IMAGE*
 * 			pointer to shmim
 *
 * @param[in]
 * index    semaphore index
 * 			index of semaphore to be excluded
 **/
long ImageStreamIO_sempost_excl(
    IMAGE *image,
    long index)
{
    long semindex;

    pid_t writeProcessPID;

    writeProcessPID = getpid();

    for (semindex = 0; semindex < image->md->sem; semindex++)
    {
        if (semindex != index)
        {
            int semval;

            sem_getvalue(image->semptr[semindex], &semval);
            if (semval < SEMAPHORE_MAXVAL)
            {
                sem_post(image->semptr[semindex]);
                image->semWritePID[semindex] = writeProcessPID;
            }
        }
    }

    if (image->semlog != NULL)
    {
        int semval;

        sem_getvalue(image->semlog, &semval);
        if (semval < SEMAPHORE_MAXVAL)
        {
            sem_post(image->semlog);
            image->semWritePID[index] = writeProcessPID;
        }
    }

    return IMAGESTREAMIO_SUCCESS;
}

/**
 * ## Purpose
 *
 * Posts all semaphores of a shmim at regular time intervals
 *
 * ## Arguments
 *
 * @param[in]
 * image	IMAGE*
 * 			pointer to shmim
 *
 * @param[in]
 * index    semaphore index
 * 			is =-1, post all semaphores
 *
 * @param[in]
 * dtus     time interval [us]
 *
 */
long ImageStreamIO_sempost_loop(
    IMAGE *image,
    long index,
    long dtus)
{

    printf("semphore loop post, dtus = %ld\n", dtus);

    while (1)
    {
        ImageStreamIO_sempost(image, index);
        usleep(dtus);
    }

    return IMAGESTREAMIO_SUCCESS;
}

/**
 * ## Purpose
 *
 * Get available shmim semaphore index
 *
 * ## Arguments
 *
 * @param[in]
 * image	IMAGE*
 * 			pointer to shmim
 *
 * @param[in]
 * index    preferred semaphore index, if available
 *
 */
int ImageStreamIO_getsemwaitindex(
    IMAGE *image,
    int semindexdefault)
{
    pid_t readProcessPID;

    readProcessPID = getpid();

    // Attempt to find a semaphore already assigned to this PID
    for (int semindex = 0; semindex < image->md->sem; ++semindex)
    {
        if (image->semReadPID[semindex] == readProcessPID)
        {
            return semindex;
        }
    }

    // check that semindexdefault is within range
    if ((semindexdefault < image->md->sem) && (semindexdefault >= 0))
    {
        // Check if semindexdefault available
        if ((image->semReadPID[semindexdefault] == 0) ||
                (getpgid(image->semReadPID[semindexdefault]) < 0))
        {
            // if OK, then adopt it
            image->semReadPID[semindexdefault] = readProcessPID;
            return semindexdefault;
        }
    }

    // if not, look for available s

    for (int semindex = 0; semindex < image->md->sem; ++semindex)
    {
        if ((image->semReadPID[semindex] == 0) ||
                (getpgid(image->semReadPID[semindex]) < 0))
        {
            image->semReadPID[semindex] = readProcessPID;
            return semindex;
        }
    }

    // if no semaphore found, return -1
    return -1;
}

/**
 * ## Purpose
 *
 * Wait on a shmim semaphore
 *
 * ## Arguments
 *
 * @param[in]
 * image	IMAGE*
 * 			pointer to shmim
 *
 * @param[in]
 * index    semaphore index
 *
 **/
int ImageStreamIO_semwait(
    IMAGE *image,
    int index)
{
    if (index > image->md->sem - 1)
    {
        printf("ERROR: image %s semaphore # %d does not exist\n", image->md->name,
               index);
        return EXIT_FAILURE;
    }
    return sem_wait(image->semptr[index]);
}

int ImageStreamIO_semtrywait(
    IMAGE *image,
    int index)
{
    if (index > image->md->sem - 1)
    {
        printf("ERROR: image %s semaphore # %d does not exist\n", image->md->name,
               index);
        return EXIT_FAILURE;
    }
    return sem_trywait(image->semptr[index]);
}

int ImageStreamIO_semtimedwait(
    IMAGE *image,
    int index,
    const struct timespec *semwts)
{
    if (index > image->md->sem - 1)
    {
        printf("ERROR: image %s semaphore # %d does not exist\n", image->md->name,
               index);
        return EXIT_FAILURE;
    }
    return sem_timedwait(image->semptr[index], semwts);
}

/**
 * ## Purpose
 *
 * Flush shmim semaphore
 *
 * ## Arguments
 *
 * @param[in]
 * image	IMAGE*
 * 			pointer to shmim
 *
 * @param[in]
 * index    semaphore index
 * 			flush all semaphores if index<0
 *
 **/
long ImageStreamIO_semflush(
    IMAGE *image,
    long index)
{
    if (index < 0)
    {
        long semindex;

        for (semindex = 0; semindex < image->md->sem; semindex++)
        {
            int semval;
            int i;

            sem_getvalue(image->semptr[semindex], &semval);
            for (i = 0; i < semval; i++)
            {
                sem_trywait(image->semptr[semindex]);
            }
        }
    }
    else
    {
        if (index > image->md->sem - 1)
            printf("ERROR: image %s semaphore # %ld does not exist\n",
                   image->md->name, index);
        else
        {
            long semindex;
            int semval;
            int i;

            semindex = index;
            sem_getvalue(image->semptr[semindex], &semval);
            for (i = 0; i < semval; i++)
            {
                sem_trywait(image->semptr[semindex]);
            }
        }
    }

    return IMAGESTREAMIO_SUCCESS;
}

long ImageStreamIO_semvalue(
    IMAGE *image,
    long index)
{
    if(index > image->md->sem - 1)
        printf("ERROR: image %s semaphore # %ld does not exist\n",
               image->md->name, index);
    else
    {
        int semval;
        sem_getvalue(image->semptr[index], &semval);
        return semval;
    }
    return -1; // in-band error bad
}

// Function to be called each time image content is updated
// Increments counter, sets write flag to zero etc...
long ImageStreamIO_UpdateIm(
    IMAGE *image)
{
    if (image->md->shared == 1)
    {

        // update circular buffer if applicable
        if (image->md->CBsize > 0)
        {
            // write index
            uint32_t CBindexWrite = image->md->CBindex + 1;
            int CBcycleincrement = 0;
            if (CBindexWrite >= image->md->CBsize)
            {
                CBindexWrite = 0;
                CBcycleincrement = 1;
            }
            // destination pointer
            void *destptr;
            destptr = ((uint8_t*)image->CBimdata) +
                      (image->md->imdatamemsize * CBindexWrite);

            memcpy(destptr, image->array.raw,
                   image->md->imdatamemsize);

            image->md->CBcycle += CBcycleincrement;
            image->md->CBindex = CBindexWrite;
        }

        image->md->cnt0++;
        image->md->write = 0;

#ifdef IMAGESTRUCT_WRITEHISTORY
        // Update image write history
        image->md->wCBindex ++;
        if( image->md->wCBindex == IMAGESTRUCT_FRAMEWRITEMDSIZE )
        {
            image->md->wCBindex = 0;
        }
        {
            struct timespec ts;
            if(clock_gettime(CLOCK_ISIO, &ts) == -1)
            {
                perror("clock_gettime");
                exit(EXIT_FAILURE);
            }
            image->writehist[image->md->wCBindex].writetime = ts;
            image->writehist[image->md->wCBindex].cnt0 = image->md->cnt0;
            image->writehist[image->md->wCBindex].wpid = getpid();
        }
#endif



        ImageStreamIO_sempost(image, -1); // post all semaphores
    }

    return IMAGESTREAMIO_SUCCESS;
}
