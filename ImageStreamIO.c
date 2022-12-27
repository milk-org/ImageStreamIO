/**
 * @file    ImageStreamIO.c
 * @brief   Read and Create image
 *
 * Read and create images and streams (shared memory)
 *
 *
 *
 */

#define _GNU_SOURCE

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
        internal_printError(__FILE__, __func__, __LINE__, code, msg);

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
        clock_gettime(CLOCK_REALTIME, &tnow);
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
    int fn = fileno(stderr);

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
    int shmdirOK = 0;
    DIR *tmpdir;

    // first, we try the env variable if it exists
    char *MILK_SHM_DIR = getenv("MILK_SHM_DIR");
    if (MILK_SHM_DIR != NULL)
    {
        // printf(" [ MILK_SHM_DIR ] '%s'\n", MILK_SHM_DIR);
        snprintf(shmdname,STRINGMAXLEN_DIR_NAME, "%s", MILK_SHM_DIR);

        // does this direcory exist ?
        tmpdir = opendir(shmdname);
        if (tmpdir) // directory exits
        {
            shmdirOK = 1;
            closedir(tmpdir);
        }
        else
        {
            printf(" [ WARNING ] '%s' does not exist\n", MILK_SHM_DIR);
        }
    }

    // second, we try SHAREDMEMDIR default
    if (shmdirOK == 0)
    {
        tmpdir = opendir(SHAREDMEMDIR);
        if (tmpdir) // directory exits
        {
            snprintf(shmdname, STRINGMAXLEN_DIR_NAME, "%s", SHAREDMEMDIR);
            shmdirOK = 1;
            closedir(tmpdir);
        }
    }

    // if all above fails, set to /tmp
    if (shmdirOK == 0)
    {
        tmpdir = opendir("/tmp");
        if (!tmpdir)
        {
            exit(EXIT_FAILURE);
        }
        else
        {
            snprintf(shmdname, STRINGMAXLEN_DIR_NAME, "/tmp");
            shmdirOK = 1;
        }
    }

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

    if ((rv > 0) && (rv < (int)ssz))
    {
        return IMAGESTREAMIO_SUCCESS;
    }
    else if (rv < 0)
    {
        ImageStreamIO_printERROR(IMAGESTREAMIO_FAILURE, strerror(errno));
        return IMAGESTREAMIO_FAILURE;
    }
    else
    {
        ImageStreamIO_printERROR(IMAGESTREAMIO_FAILURE,
                                 "string not large enough for file name");
        return IMAGESTREAMIO_FAILURE;
    }
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
        ImageStreamIO_printERROR(IMAGESTREAMIO_INVALIDARG, "invalid type code");
        return -1; // This is an in-band error code, so can't be > 0.
    }
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
        return "unknown";
    }
}

int ImageStreamIO_checktype(uint8_t datatype, int complex_allowed) {

    int complex_retval = complex_allowed ? 0 : -1;

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
        return complex_retval;

    default:
        ImageStreamIO_printERROR(IMAGESTREAMIO_INVALIDARG, "invalid type code");
        return -1; // This is an in-band error code, so can't be > 0.
    }
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
        return " UI64";
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
        return " ???";
    }
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
        ImageStreamIO_printERROR(IMAGESTREAMIO_INVALIDARG, "invalid type code");
        return -1; // This is an in-band error code, so can't be > 0.
    }
}

int ImageStreamIO_bitpix(
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
        ImageStreamIO_printERROR(IMAGESTREAMIO_INVALIDARG,
                                 "bitpix not implemented for type");
        return -1; // This is an in-band error code, must be unique from valid BITPIX values.
    }
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
            memset(image->array.raw, '\0', image->md->nelement * size_element);
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
    uint32_t CBsize // circular buffer size (if shared), 0 if not used
)
{
    long nelement;

    uint8_t *map;

    int kw;

    // Get shm directory name (only on first call to this function)
    static char shmdirname[200];
    static int initSHAREDMEMDIR = 0;
    if (initSHAREDMEMDIR == 0)
    {
        ImageStreamIO_shmdirname(shmdirname);
        for (unsigned int stri = 0; stri < strlen(shmdirname); stri++)
            if (shmdirname[stri] == '/') // replace '/' by '.'
            {
                shmdirname[stri] = '.';
            }
        initSHAREDMEMDIR = 1;
    }

    int NBproctrace = IMAGE_NB_PROCTRACE;

    nelement = 1;
    for (long i = 0; i < naxis; i++)
    {
        nelement *= size[i];
    }
    uint64_t imdatamemsize = ImageStreamIO_typesize(datatype) * nelement;

    if (((imagetype & 0xF000F) == CIRCULAR_BUFFER) &&
            (naxis != 3))
    {
        ImageStreamIO_printERROR(IMAGESTREAMIO_INVALIDARG,
                                 "Error calling ImageStreamIO_createIm_gpu, "
                                 "temporal circular buffer needs 3 dimensions");
        return IMAGESTREAMIO_INVALIDARG;
    }

    // compute total size to be allocated
    if (shared == 1)
    {
        char sname[200];

        // create semlog
        size_t sharedsize = 0;     // shared memory size in bytes
        size_t datasharedsize = 0; // shared memory size in bytes used by the data

        snprintf(sname, sizeof(sname), "%s.%s_semlog", shmdirname, name);
        remove(sname);
        image->semlog = NULL;

        umask(0);
        if ((image->semlog = sem_open(sname, O_CREAT, FILEMODE, 1)) == SEM_FAILED)
        {
            fprintf(stderr, "Semaphore %s :", sname);
            ImageStreamIO_printERROR(IMAGESTREAMIO_SEMINIT,
                                     "semaphore creation / initialization");
        }
        else
        {
            sem_init(
                image->semlog, 1,
                SEMAPHORE_INITVAL); // SEMAPHORE_INITVAL defined in ImageStruct.h
        }
        sharedsize = sizeof(IMAGE_METADATA);
        datasharedsize = imdatamemsize;

        if (location == -1)
        {
            // printf("shared memory space in CPU RAM = %ud bytes\n", sharedsize);
            sharedsize += datasharedsize;
        }
        else if (location >= 0)
        {
            // printf("shared memory space in GPU%d RAM= %ud bytes\n", location,
            // sharedsize); //TEST
        }
        else
        {
            ImageStreamIO_printERROR(IMAGESTREAMIO_INVALIDARG, "Error location unknown");
        }

        sharedsize += sizeof(IMAGE_KEYWORD) * NBkw;


        sharedsize += sizeof(SEMFILEDATA) * NBsem;

        // one read PID array, one write PID array
        sharedsize += 2 * NBsem * sizeof(pid_t);

        // semctrl
        sharedsize += sizeof(uint32_t) * NBsem;

        // semstatus
        sharedsize += sizeof(uint32_t) * NBsem;

        sharedsize += sizeof(STREAM_PROC_TRACE) * NBproctrace;

        if ((imagetype & 0xF000F) ==
                (CIRCULAR_BUFFER | ZAXIS_TEMPORAL)) // Circular buffer
        {
            // room for atimearray, writetimearray and cntarray
            sharedsize += size[2] * (2 * sizeof(struct timespec) + sizeof(uint64_t));
        }

        // fast circular buffer metadata
        sharedsize += sizeof(CBFRAMEMD) * CBsize;

        // fast circular buffer data buffer
        sharedsize += datasharedsize * CBsize;

        char SM_fname[200];
        ImageStreamIO_filename(SM_fname, 200, name);

        struct stat buffer;
        if ((stat(SM_fname, &buffer) == 0) && (location > -1))
        {
            ImageStreamIO_printERROR(IMAGESTREAMIO_FILEEXISTS,
                                     "Error creating GPU SHM buffer on an existing file");
            return IMAGESTREAMIO_FILEEXISTS;
        }

        int SM_fd; // shared memory file descriptor
        umask(0);
        SM_fd = open(SM_fname, O_RDWR | O_CREAT | O_TRUNC, (mode_t)FILEMODE);
        if (SM_fd == -1)
        {
            ImageStreamIO_printERROR(IMAGESTREAMIO_FILEOPEN,
                                     "Error opening file for writing");
            return IMAGESTREAMIO_FILEOPEN;
        }

        image->shmfd = SM_fd;
        image->memsize = sharedsize;

        int result;
        result = lseek(SM_fd, sharedsize - 1, SEEK_SET);
        if (result == -1)
        {
            close(SM_fd);
            ImageStreamIO_printERROR(IMAGESTREAMIO_FILESEEK,
                                     "Error calling lseek() to 'stretch' the file");
            return IMAGESTREAMIO_FILESEEK;
        }

        result = write(SM_fd, "", 1);
        if (result != 1)
        {
            close(SM_fd);
            ImageStreamIO_printERROR(IMAGESTREAMIO_FILEWRITE,
                                     "Error writing last byte of the file");
            return IMAGESTREAMIO_FILEWRITE;
        }

        map = (uint8_t *)mmap(0, sharedsize, PROT_READ | PROT_WRITE, MAP_SHARED,
                              SM_fd, 0);
        if (map == MAP_FAILED)
        {
            close(SM_fd);
            ImageStreamIO_printERROR(IMAGESTREAMIO_MMAP, "Error mmapping the file");
            return IMAGESTREAMIO_MMAP;
        }

        image->md = (IMAGE_METADATA *)map;
        image->md->shared = 1;
        image->md->creatorPID = getpid();
        image->md->ownerPID = 0; // default value, indicates unset
        image->md->sem = NBsem;
        image->md->NBproctrace = NBproctrace;

        {
            struct stat file_stat;
            int ret;
            ret = fstat(SM_fd, &file_stat);
            if (ret < 0)
            {
                ImageStreamIO_printERROR(IMAGESTREAMIO_INODE, "Error getting inode");
                return IMAGESTREAMIO_INODE;
            }
            image->md->inode =
                file_stat.st_ino; // inode now contains inode number of the file with descriptor fd
        }

        map += sizeof(IMAGE_METADATA);

        if (location == -1)
        {
            image->array.raw = map;
            map += datasharedsize;
        }
        else if (location >= 0)
        {
            image->array.raw = NULL;
        }
        else
        {
            ImageStreamIO_printERROR(IMAGESTREAMIO_NOTIMPL, "Error location unknown");
            return IMAGESTREAMIO_NOTIMPL;
        }
        image->kw = (IMAGE_KEYWORD *)(map);
        map += sizeof(IMAGE_KEYWORD) * NBkw;

        image->semfile = (SEMFILEDATA*)(map);
        map += sizeof(SEMFILEDATA) * NBsem;

        image->semReadPID = (pid_t *)(map);
        map += sizeof(pid_t) * NBsem;

        image->semWritePID = (pid_t *)(map);
        map += sizeof(pid_t) * NBsem;

        image->semctrl = (uint32_t *)(map);
        map += sizeof(uint32_t) * NBsem;

        image->semstatus = (uint32_t *)(map);
        map += sizeof(uint32_t) * NBsem;

        image->streamproctrace = (STREAM_PROC_TRACE *)(map);
        map += sizeof(STREAM_PROC_TRACE) * NBproctrace;

        if ((imagetype & 0xF000F) ==
                (CIRCULAR_BUFFER | ZAXIS_TEMPORAL)) // If main image is circular buffer
        {
            image->atimearray = (struct timespec *)(map);
            map += sizeof(struct timespec) * size[2];

            image->writetimearray = (struct timespec *)(map);
            map += sizeof(struct timespec) * size[2];

            image->cntarray = (uint64_t *)(map);
            map += sizeof(uint64_t) * size[2];
        }

        image->CircBuff_md = (CBFRAMEMD *)(map);
        map += sizeof(CBFRAMEMD) * CBsize;

        if (CBsize > 0)
        {
            image->CBimdata = map;
        }
        else
        {
            image->CBimdata = NULL;
        }
        map += datasharedsize * CBsize;
        image->md->CBsize = CBsize;
        image->md->CBindex = 0;
        image->md->CBcycle = 0;
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
        if (NBkw > 0)
        {
            image->kw = (IMAGE_KEYWORD *)malloc(sizeof(IMAGE_KEYWORD) * NBkw);
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
        image->md->CBsize = 0;
        image->md->CBindex = 0;
        image->md->CBcycle = 0;
    }

    strncpy(image->md->version, IMAGESTRUCT_VERSION, 32);
    image->md->imagetype = imagetype; // Image is mathematical vector or matrix
    image->md->location = location;
    image->md->datatype = datatype;
    image->md->naxis = naxis;
    strncpy(image->name, name, STRINGMAXLEN_IMAGE_NAME - 1); // local name
    strncpy(image->md->name, name, STRINGMAXLEN_IMAGE_NAME - 1);
    for (long i = 0; i < naxis; i++)
    {
        image->md->size[i] = size[i];
    }
    image->md->nelement = nelement;
    image->md->imdatamemsize = imdatamemsize;

    image->md->NBkw = NBkw;

    ImageStreamIO_initialize_buffer(image);

    clock_gettime(CLOCK_REALTIME, &image->md->lastaccesstime);
    clock_gettime(CLOCK_REALTIME, &image->md->creationtime);
    // image->md->lastaccesstime =
    //     1.0 * timenow.tv_sec + 0.000000001 * timenow.tv_nsec;
    // image->md->creationtime = image->md->lastaccesstime;

    image->md->write = 0;
    image->md->cnt0 = 0;
    image->md->cnt1 = 0;

    if (shared == 1)
    {
        ImageStreamIO_createsem(image, NBsem); // IMAGE_NB_SEMAPHORE
        // defined in ImageStruct.h

        int semindex;
        for (semindex = 0; semindex < NBsem; semindex++)
        {
            image->semReadPID[semindex] = -1;
            image->semWritePID[semindex] = -1;
            image->semctrl[semindex] = 0;
            image->semstatus[semindex] = 0;
        }

        for (int proctraceindex = 0; proctraceindex < NBproctrace; proctraceindex++)
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
        image->md->sem = 0; // no semaphores
    }

    // initialize keywords
    for (kw = 0; kw < image->md->NBkw; kw++)
    {
        image->kw[kw].type = 'N';
    }

    image->used = 1;
    image->createcnt++;

    // DEBUG_TRACEPOINT_LOG("%s %d NBsem = %d", __FILE__, __LINE__, image->md->sem);

    return IMAGESTREAMIO_SUCCESS;
}



errno_t ImageStreamIO_destroyIm(
    IMAGE *image)
{
    if(image->used == 1)
    {
        // Get shm directory name (only on first call to this function)
        static char shmdirname[200];
        static int initSHAREDMEMDIR = 0;
        if (initSHAREDMEMDIR == 0)
        {
            unsigned int stri;

            ImageStreamIO_shmdirname(shmdirname);
            for (stri = 0; stri < strlen(shmdirname); stri++)
                if (shmdirname[stri] == '/') // replace leading '/' by '.'
                {
                    shmdirname[stri] = '.';
                }
            initSHAREDMEMDIR = 1;
        }

        char fname[200];

        // close and remove semlog
        sem_close(image->semlog);
        snprintf(fname, sizeof(fname), "/dev/shm/sem.%s.%s_semlog", shmdirname,
                 image->md->name);
        sem_unlink(fname);
        image->semlog = NULL;
        remove(fname);

        // close and remove all semaphores
        ImageStreamIO_destroysem(image);
        image->semptr = NULL;

        if (image->memsize > 0)
        {
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
 * Read shared memory image\n
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
    char SM_fname[STRINGMAXLEN_FILE_NAME];

    ImageStreamIO_filename(SM_fname, sizeof(SM_fname), name);

    SM_fd = open(SM_fname, O_RDWR);
    if (SM_fd == -1)
    {
        image->used = 0;
        char wmsg[200];
        snprintf(wmsg, 200, "Cannot open shm file \"%s\"\n", SM_fname);
        ImageStreamIO_printWARNING(wmsg);
        return IMAGESTREAMIO_FILEOPEN;
    }

    char sname[200];
    uint8_t *map;
    long s;
    struct stat file_stat;

    long snb = 0;
    int sOK = 1;

    // Get shm directory name (only on first call to this function)
    static char shmdirname[200];
    static int initSHAREDMEMDIR = 0;
    if (initSHAREDMEMDIR == 0)
    {
        unsigned int stri;

        ImageStreamIO_shmdirname(shmdirname);
        for (stri = 0; stri < strlen(shmdirname); stri++)
            if (shmdirname[stri] == '/') // replace leading '/' by '.'
            {
                shmdirname[stri] = '.';
            }
        initSHAREDMEMDIR = 1;
    }

    fstat(SM_fd, &file_stat);

    // printf("File %s size: %zd\n", SM_fname, file_stat.st_size); fflush(stdout); //TEST

    map = (uint8_t *)mmap(0, file_stat.st_size, PROT_READ | PROT_WRITE,
                          MAP_SHARED, SM_fd, 0);
    if (map == MAP_FAILED)
    {
        close(SM_fd);
        ImageStreamIO_printERROR(IMAGESTREAMIO_MMAP, "Error mmapping the file");
        return IMAGESTREAMIO_MMAP;
    }

    image->memsize = file_stat.st_size;
    image->shmfd = SM_fd;
    image->md = (IMAGE_METADATA *)map;
    image->md->shared = 1;

    if (strcmp(image->md->version, IMAGESTRUCT_VERSION))
    {
        char errmsg[200];
        snprintf(errmsg, 200, "Stream %s corrupted, or incompatible version. Should be %s",
                 name, IMAGESTRUCT_VERSION);
        ImageStreamIO_printERROR(IMAGESTREAMIO_VERSION, errmsg);
        return IMAGESTREAMIO_VERSION;
    }

    uint64_t size = 1;
    uint8_t axis;
    for (axis = 0; axis < image->md->naxis; ++axis)
    {
        size *= image->md->size[axis];
    }

    // some verification
    if (size > 10000000000)
    {
        printf("IMAGE \"%s\" SEEMS BIG... NOT LOADING\n", name);
        fflush(stdout);
        return IMAGESTREAMIO_FAILURE;
    }
    for (axis = 0; axis < image->md->naxis; ++axis)
    {
        if (image->md->size[axis] < 1)
        {
            printf("IMAGE \"%s\" AXIS %d SIZE < 1... NOT LOADING\n", name, axis);
            fflush(stdout);
            return IMAGESTREAMIO_FAILURE;
        }
    }

    map += sizeof(IMAGE_METADATA);

    // gain image data array pointer
    if (image->md->location >= 0)
    {
        image->array.raw = NULL;
    }
    map += ImageStreamIO_offset_data(image, map);
    if (image->array.raw == NULL)
    {
        printf("Fail to retrieve data pointer\n");
        fflush(stdout);
        return IMAGESTREAMIO_FAILURE;
    }

    // printf("%ld keywords\n", (long)image->md->NBkw); fflush(stdout); //TEST

    image->kw = (IMAGE_KEYWORD *)(map);
    map += sizeof(IMAGE_KEYWORD) * image->md->NBkw;
    ///<\todo can the following code be deleted?
    /*
      int kw;
      for (kw = 0; kw < image->md->NBkw; kw++) {
        if (image->kw[kw].type == 'L')
          printf("%d  %s %ld %s\n", kw, image->kw[kw].name,
                 image->kw[kw].value.numl, image->kw[kw].comment);
        if (image->kw[kw].type == 'D')
          printf("%d  %s %lf %s\n", kw, image->kw[kw].name,
                 image->kw[kw].value.numf, image->kw[kw].comment);
        if (image->kw[kw].type == 'S')
          printf("%d  %s %s %s\n", kw, image->kw[kw].name,
                 image->kw[kw].value.valstr, image->kw[kw].comment);
      }
    */

    image->semfile = (SEMFILEDATA *)(map);
    map += sizeof(SEMFILEDATA) * image->md->sem;

    image->semReadPID = (pid_t *)(map);
    map += sizeof(pid_t) * image->md->sem;

    image->semWritePID = (pid_t *)(map);
    map += sizeof(pid_t) * image->md->sem;

    image->semctrl = (uint32_t *)(map);
    map += sizeof(uint32_t) * image->md->sem;

    image->semstatus = (uint32_t *)(map);
    map += sizeof(uint32_t) * image->md->sem;

    image->streamproctrace = (STREAM_PROC_TRACE *)(map);
    map += sizeof(STREAM_PROC_TRACE) * image->md->NBproctrace;

    if ((image->md->imagetype & 0xF000F) ==
            (CIRCULAR_BUFFER | ZAXIS_TEMPORAL))
    {
        // printf("circuar buffer\n"); fflush(stdout); //TEST

        // Circular buffer
        image->atimearray = (struct timespec *)(map);
        map += sizeof(struct timespec) * image->md->size[2];

        image->writetimearray = (struct timespec *)(map);
        map += sizeof(struct timespec) * image->md->size[2];

        image->cntarray = (uint64_t *)(map);
        map += sizeof(uint64_t) * image->md->size[2];
    }

    if (image->md->CBsize > 0)
    {
        image->CircBuff_md = (CBFRAMEMD *)map;
        map += sizeof(CBFRAMEMD) * image->md->CBsize;

        image->CBimdata = map;
    }
    else
    {
        image->CircBuff_md = NULL;
        image->CBimdata = NULL;
    }

    strncpy(image->name, name, STRINGMAXLEN_IMAGE_NAME - 1);

    // looking for semaphores
    // printf("Looking for semaphores\n"); fflush(stdout); //TEST
    while (sOK == 1)
    {
        snprintf(sname, sizeof(sname), "%s.%s_sem%02ld", shmdirname, image->md->name,
                 snb);
        sem_t *stest;
        umask(0);
        if ((stest = sem_open(sname, 0, FILEMODE, 0)) == SEM_FAILED)
        {
            sOK = 0; // not an error here
        }
        else
        {
            sem_close(stest);
            snb++;
        }
    }

    //        image->md->sem = snb;
    image->semptr = (sem_t **)malloc(sizeof(sem_t *) * image->md->sem);
    if (image->semptr == NULL)
    {
        printf("Memory allocation error %s %d\n", __FILE__, __LINE__);
        abort();
    }
    for (s = 0; s < image->md->sem; s++)
    {
        snprintf(sname, sizeof(sname), "%s.%s_sem%02ld", shmdirname, image->md->name,
                 s);
        umask(0);
        if ((image->semptr[s] = sem_open(sname, 0, FILEMODE, 0)) == SEM_FAILED)
        {
            // printf("ERROR: could not open semaphore %s -> (re-)CREATING semaphore\n",
            //        sname);

            if ((image->semptr[s] = sem_open(sname, O_CREAT, FILEMODE, 1)) ==
                    SEM_FAILED)
            {
                ImageStreamIO_printERROR(IMAGESTREAMIO_SEMINIT, "semaphore initialization");
                return IMAGESTREAMIO_SEMINIT;
            }
            else
            {
                sem_init(
                    image->semptr[s], 1,
                    SEMAPHORE_INITVAL); // SEMAPHORE_INITVAL defined in ImageStruct.h
            }


            // get semaphore inode
            {
                struct stat file_stat;
                int ret;
                char fullsname[STRINGMAXLEN_SEMFILENAME];

                snprintf(fullsname, sizeof(sname), "/dev/shm/sem.%s", sname);


                int fd = open(fullsname, O_RDONLY);
                ret = fstat (fd, &file_stat);
                if (ret < 0) {
                    // error getting file stat
                }
                snprintf(image->semfile[s].fname, STRINGMAXLEN_SEMFILENAME, "%s", sname);

                image->semfile[s].inode = file_stat.st_ino;
                close(fd);
            }
        }
    }

    snprintf(sname, sizeof(sname), "%s.%s_semlog", shmdirname, image->md->name);
    umask(0);
    if ((image->semlog = sem_open(sname, 0, FILEMODE, 0)) == SEM_FAILED)
    {
        // printf("ERROR: could not open semaphore %s -> (re-)CREATING semaphore\n",
        //        sname);
        if ((image->semlog = sem_open(sname, O_CREAT, FILEMODE, 1)) == SEM_FAILED)
        {
            ImageStreamIO_printERROR(IMAGESTREAMIO_SEMINIT, "semaphore initialization");
            return IMAGESTREAMIO_SEMINIT;
        }
        else
        {
            sem_init(
                image->semlog, 1,
                SEMAPHORE_INITVAL); // SEMAPHORE_INITVAL defined in ImageStruct.h
        }
    }

    return IMAGESTREAMIO_SUCCESS;
}








errno_t ImageStreamIO_closeIm(
    IMAGE *image)
{
    long s;

    for (s = 0; s < image->md->sem; s++)
    {
        sem_close(image->semptr[s]);
    }

    free(image->semptr);

    sem_close(image->semlog);

    if (munmap(image->md, image->memsize) != 0)
    {
        ImageStreamIO_printERROR(IMAGESTREAMIO_MMAP, "error unmapping memory");
        return IMAGESTREAMIO_MMAP;
    }

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
 * Destroy semaphore of a shmim
 *
 * ## Arguments
 *
 * @param[in]
 * image	IMAGE*
 * 			pointer to shmim
 **/

errno_t ImageStreamIO_destroysem(
    IMAGE *image)
{
    // Get shm directory name (only on first call to this function)
    static char shmdirname[200];
    static int initSHAREDMEMDIR = 0;
    if (initSHAREDMEMDIR == 0)
    {
        unsigned int stri;

        ImageStreamIO_shmdirname(shmdirname);
        for (stri = 0; stri < strlen(shmdirname); stri++)
            if (shmdirname[stri] == '/') // replace leading '/' by '.'
            {
                shmdirname[stri] = '.';
            }
        initSHAREDMEMDIR = 1;
    }

    // Remove semaphores if any
    if (image->md->sem > 0)
    {
        // Close existing semaphores ...
        for (int s = 0; s < image->md->sem; s++)
        {
            if ((image->semptr != NULL) && (image->semptr[s] != NULL))
            {
                sem_close(image->semptr[s]);
            }

            // ... and remove associated files
            char fname[200];
            snprintf(fname, sizeof(fname), "/dev/shm/sem.%s.%s_sem%02d", shmdirname,
                     image->md->name, s);
            sem_unlink(fname);
            remove(fname);
        }
        image->md->sem = 0;
    }

    if (image->semptr != NULL)
    {
        free(image->semptr);
        image->semptr = NULL;
    }

    return (IMAGESTREAMIO_SUCCESS);
}

/**
 * ## Purpose
 *
 * Create semaphore of a shmim
 *
 * ## Arguments
 *
 * @param[in]
 * image	IMAGE*
 * 			pointer to shmim
 *
 * @param[in]
 * NBsem    number of semaphores to be created
 **/

int ImageStreamIO_createsem(
    IMAGE *image,
    long NBsem)
{
    // printf("Creating %ld semaphores\n", NBsem);

    // Get shm directory name (only on first call to this function)
    static char shmdirname[200];
    static int initSHAREDMEMDIR = 0;

    if (initSHAREDMEMDIR == 0)
    {
        unsigned int stri;

        ImageStreamIO_shmdirname(shmdirname);
        for (stri = 0; stri < strlen(shmdirname); stri++)
            if (shmdirname[stri] == '/') // replace leading '/' by '.'
            {
                shmdirname[stri] = '.';
            }
        initSHAREDMEMDIR = 1;
    }

    // Remove pre-existing semaphores if any
    // ImageStreamIO_destroysem(image);

    // printf("malloc semptr %ld entries\n", NBsem);
    image->semptr = (sem_t **)malloc(sizeof(sem_t **) * NBsem);
    if (image->semptr == NULL)
    {
        printf("Memory allocation error %s %d\n", __FILE__, __LINE__);
        abort();
    }

    for (int s = 0; s < NBsem; s++)
    {
        char sname[200];
        snprintf(sname, sizeof(sname), "%s.%s_sem%02d", shmdirname, image->md->name,
                 s);
        umask(0);
        if ((image->semptr[s] = sem_open(sname, O_CREAT, FILEMODE, 0)) == SEM_FAILED)
        {
            ImageStreamIO_printERROR(IMAGESTREAMIO_SEMINIT, "semaphore initilization");
        }
        else
        {
            sem_init(
                image->semptr[s], 1,
                SEMAPHORE_INITVAL); // SEMAPHORE_INITVAL defined in ImageStruct.h
        }

        // get semaphore inode
        {
            struct stat file_stat;
            int ret;
            char fullsname[STRINGMAXLEN_SEMFILENAME];

            snprintf(fullsname, sizeof(sname), "/dev/shm/sem.%s", sname);


            int fd = open(fullsname, O_RDONLY);
            ret = fstat (fd, &file_stat);
            if (ret < 0) {
                // error getting file stat
            }
            snprintf(image->semfile[s].fname, STRINGMAXLEN_SEMFILENAME, "%s", sname);

            image->semfile[s].inode = file_stat.st_ino;
            close(fd);
        }


        // Do this last so nobody accesses before init is done.
        image->md->sem = NBsem;
    }

    return IMAGESTREAMIO_SUCCESS;
}





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
        long s;

        for (s = 0; s < image->md->sem; s++)
        {
            int semval;

            image->semWritePID[s] = writeProcessPID;

            sem_getvalue(image->semptr[s], &semval);
            if (semval < SEMAPHORE_MAXVAL)
            {
                sem_post(image->semptr[s]);
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
    long s;

    pid_t writeProcessPID;

    writeProcessPID = getpid();

    for (s = 0; s < image->md->sem; s++)
    {
        if (s != index)
        {
            int semval;

            sem_getvalue(image->semptr[s], &semval);
            if (semval < SEMAPHORE_MAXVAL)
            {
                sem_post(image->semptr[s]);
                image->semWritePID[s] = writeProcessPID;
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

    // if not, look for available semindex

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
        long s;

        for (s = 0; s < image->md->sem; s++)
        {
            int semval;
            int i;

            sem_getvalue(image->semptr[s], &semval);
            for (i = 0; i < semval; i++)
            {
                sem_trywait(image->semptr[s]);
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
            long s;
            int semval;
            int i;

            s = index;
            sem_getvalue(image->semptr[s], &semval);
            for (i = 0; i < semval; i++)
            {
                sem_trywait(image->semptr[s]);
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
            destptr = image->CBimdata +
                      image->md->imdatamemsize * CBindexWrite;

            memcpy(destptr, image->array.raw,
                   image->md->imdatamemsize);

            image->md->CBcycle += CBcycleincrement;
            image->md->CBindex = CBindexWrite;
        }

        image->md->cnt0++;
        image->md->write = 0;
        ImageStreamIO_sempost(image, -1); // post all semaphores
    }

    return IMAGESTREAMIO_SUCCESS;
}