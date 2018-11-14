/**
 * @file    ImageStreamIO.c
 * @brief   Read and Create image
 *
 * Read and create images and streams (shared memory)
 *
 *
 *
 * @author  O. Guyon
 *
 *
 * @bug No known bugs.
 *
 */

#define _GNU_SOURCE

#include <malloc.h>
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

#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>  // for open
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <semaphore.h>
#include <unistd.h>  // for close

#include <fitsio.h>

// Handle old fitsios
#ifndef ULONGLONG_IMG
#define ULONGLONG_IMG (80)
#endif

#include "ImageStreamIO.h"

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

#ifdef HAVE_CUDA
#include "helper_cuda.h"
#endif

static int INITSTATUS_ImageStreamIO = 0;

void __attribute__((constructor)) libinit_ImageStreamIO() {
  if (INITSTATUS_ImageStreamIO == 0) {
    init_ImageStreamIO();
    INITSTATUS_ImageStreamIO = 1;
  }
}

int_fast8_t init_ImageStreamIO() {
  // any initialization needed ?

  return EXIT_SUCCESS;
}

#define ImageStreamIO_printERROR(msg) \
  ImageStreamIO_printERROR_(__FILE__, __func__, __LINE__, msg);

int ImageStreamIO_printERROR_(const char *file, const char *func, int line,
                              char *errmessage) {
  fprintf(stderr,
          "%c[%d;%dmERROR [ FILE: %s   FUNCTION: %s   LINE: %d ]  %c[%d;m\n",
          (char)27, 1, 31, file, func, line, (char)27, 0);
  if (errno != 0) {
    char buff[256];

// Test for which version of strerror_r we're using (XSI or GNU)
#if ((_POSIX_C_SOURCE >= 200112L || _XOPEN_SOURCE >= 600) && \
     !defined(_GNU_SOURCE))
    if (strerror_r(errno, buff, sizeof(buff)) == 0) {
      fprintf(stderr, "C Error: %s\n", buff);
    } else
      fprintf(stderr, "Unknown C Error\n");
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

    errno = _errno;  // restore it in case it's used later.
#endif

  } else
    fprintf(stderr, "%c[%d;%dmNo C error (errno = 0)%c[%d;m\n", (char)27, 1, 31,
            27, 0);

  fprintf(stderr, "%c[%d;%dm %s  %c[%d;m\n", (char)27, 1, 31, errmessage,
          (char)27, 0);

  return EXIT_SUCCESS;
}

/* ===============================================================================================
 */
/* ===============================================================================================
 */
/* @name 0. Utilities
 *
 */
/* ===============================================================================================
 */
/* ===============================================================================================
 */

int ImageStreamIO_filename(char *file_name, size_t ssz, const char *im_name) {
  int rv = snprintf(file_name, ssz, "%s/%s.im.shm", SHAREDMEMDIR, im_name);

  if (rv > 0 && rv < ssz)
    return 0;
  else if (rv < 0) {
    ImageStreamIO_printERROR(strerror(errno));
    return EXIT_FAILURE;
  } else {
    ImageStreamIO_printERROR("string not large enough for file name");
    return EXIT_FAILURE;
  }
}

int ImageStreamIO_typesize(uint8_t atype) {
  switch (atype) {
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
    case _DATATYPE_FLOAT:
      return SIZEOF_DATATYPE_FLOAT;
    case _DATATYPE_DOUBLE:
      return SIZEOF_DATATYPE_DOUBLE;
    case _DATATYPE_COMPLEX_FLOAT:
      return SIZEOF_DATATYPE_COMPLEX_FLOAT;
    case _DATATYPE_COMPLEX_DOUBLE:
      return SIZEOF_DATATYPE_COMPLEX_DOUBLE;
    case _DATATYPE_EVENT_UI8_UI8_UI16_UI8:
      return SIZEOF_DATATYPE_EVENT_UI8_UI8_UI16_UI8;

    default:
      ImageStreamIO_printERROR("invalid type code");
      return EXIT_FAILURE;
  }
}

int ImageStreamIO_bitpix(uint8_t atype) {
  switch (atype) {
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
    default:
      ImageStreamIO_printERROR("bitpix not implemented for type");
      return EXIT_FAILURE;
  }
}

uint64_t ImageStreamIO_offset_data(IMAGE *image, void *map) {
  uint8_t atype = image->md->atype;
  u_int64_t offset = 0;

  printf("atype = %d\n", (int)atype);
  fflush(stdout);

  if (atype == _DATATYPE_UINT8) {
    printf("atype = UINT8\n");
    image->array.UI8 = (uint8_t *)map;
  } else if (atype == _DATATYPE_INT8) {
    printf("atype = INT8\n");
    image->array.SI8 = (int8_t *)map;
  } else if (atype == _DATATYPE_UINT16) {
    printf("atype = UINT16\n");
    image->array.UI16 = (uint16_t *)map;
  } else if (atype == _DATATYPE_INT16) {
    printf("atype = INT16\n");
    image->array.SI16 = (int16_t *)map;
  } else if (atype == _DATATYPE_UINT32) {
    printf("atype = UINT32\n");
    image->array.UI32 = (uint32_t *)map;
  } else if (atype == _DATATYPE_INT32) {
    printf("atype = INT32\n");
    image->array.SI32 = (int32_t *)map;
  } else if (atype == _DATATYPE_UINT64) {
    printf("atype = UINT64\n");
    image->array.UI64 = (uint64_t *)map;
  } else if (atype == _DATATYPE_INT64) {
    printf("atype = INT64\n");
    image->array.SI64 = (int64_t *)map;
  } else if (atype == _DATATYPE_FLOAT) {
    printf("atype = FLOAT\n");
    image->array.F = (float *)map;
  } else if (atype == _DATATYPE_DOUBLE) {
    printf("atype = DOUBLE\n");
    image->array.D = (double *)map;
  } else if (atype == _DATATYPE_COMPLEX_FLOAT) {
    printf("atype = COMPLEX_FLOAT\n");
    image->array.CF = (complex_float *)map;
  } else if (atype == _DATATYPE_COMPLEX_DOUBLE) {
    printf("atype = COMPLEX_DOUBLE\n");
    image->array.CD = (complex_double *)map;
  }

  if (image->md[0].location >= 0) {
    printf("atype = GPUIPC\n");
    image->array.raw = NULL;
    offset = 0;
  } else {
    image->array.raw = map;
    offset = ImageStreamIO_typesize(atype) * image->md[0].nelement;
  }

  return offset;
}

int ImageStreamIO_initialize_buffer(IMAGE *image) {
  void *map;  // pointed cast in bytes
  const size_t size_element = ImageStreamIO_typesize(image->md[0].atype);

  if (image->md[0].location == -1) {
    if (image->md[0].shared == 1) {
      memset(image->array.raw, '\0', image->md[0].nelement * size_element);
    } else {
      image->array.raw = calloc((size_t)image->md[0].nelement, size_element);
      if (image->array.raw == NULL) {
        ImageStreamIO_printERROR("memory allocation failed");
        fprintf(stderr, "%c[%d;%dm", (char)27, 1, 31);
        fprintf(stderr, "Image name = %s\n", image->name);
        fprintf(stderr, "Image size = ");
        fprintf(stderr, "%ld", (long)image->md[0].size[0]);
        for (int i = 1; i < image->md[0].naxis; i++)
          fprintf(stderr, "x%ld", (long)image->md[0].size[i]);
        fprintf(stderr, "\n");
        fprintf(stderr, "Requested memory size = %ld elements = %f Mb\n",
                (long)image->md[0].nelement,
                1.0 / 1024 / 1024 * image->md[0].nelement * sizeof(uint8_t));
        fprintf(stderr, " %c[%d;m", (char)27, 0);
        exit(0);
      }
    }
#ifdef HAVE_CUDA
  } else if (image->md[0].location >= 0) {
    checkCudaErrors(cudaSetDevice(image->md[0].location));
    checkCudaErrors(
        cudaMalloc(&image->array.raw, size_element * image->md[0].nelement));
    if (image->md[0].shared == 1) {
      checkCudaErrors(
          cudaIpcGetMemHandle(&image->md[0].cudaMemHandle, image->array.raw));
    }
#endif
  }

  ImageStreamIO_offset_data(image, image->array.raw);

  return EXIT_SUCCESS;
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

int ImageStreamIO_createIm(IMAGE *image, const char *name, long naxis,
                           uint32_t *size, uint8_t atype, int shared,
                           int NBkw) {
  return ImageStreamIO_createIm_gpu(image, name, naxis, size, atype, -1, shared,
                                    NBkw);
}

int ImageStreamIO_createIm_gpu(IMAGE *image, const char *name, long naxis,
                               uint32_t *size, uint8_t atype, int8_t location,
                               int shared, int NBkw) {
  long i, ii;
  time_t lt;
  long nelement;
  struct timespec timenow;

  uint8_t *map;

  int kw;
  char comment[80];
  char kname[16];

  nelement = 1;
  for (i = 0; i < naxis; i++) nelement *= size[i];

  // compute total size to be allocated
  if (shared == 1) {
    char sname[200];

    // create semlog
    size_t sharedsize = 0;      // shared memory size in bytes
    size_t datasharedsize = 0;  // shared memory size in bytes used by the data

    snprintf(sname, sizeof(sname), "%s_semlog", name);
    remove(sname);
    image->semlog = NULL;

    if ((image->semlog = sem_open(sname, O_CREAT, 0644, 1)) == SEM_FAILED)
      perror("semaphore creation / initilization");
    else
      sem_init(image->semlog, 1, 0);

    sharedsize = sizeof(IMAGE_METADATA);

    datasharedsize = nelement * ImageStreamIO_typesize(atype);

    if (location == -1) {
      // printf("shared memory space in CPU RAM = %ud bytes\n", sharedsize);
      // //TEST
      sharedsize += datasharedsize;
    } else if (location >= 0) {
      // printf("shared memory space in GPU%d RAM= %ud bytes\n", location,
      // sharedsize); //TEST
    } else if (location < -1) {
      perror("Error location unknown");
    }

    sharedsize += NBkw * sizeof(IMAGE_KEYWORD);

    char SM_fname[200];
    ImageStreamIO_filename(SM_fname, 200, name);

    int SM_fd;  // shared memory file descriptor
    SM_fd = open(SM_fname, O_RDWR | O_CREAT | O_TRUNC, (mode_t)0600);
    if (SM_fd == -1) {
      perror("Error opening file for writing");
      exit(0);
    }

    image->shmfd = SM_fd;
    image->memsize = sharedsize;

    int result;
    result = lseek(SM_fd, sharedsize - 1, SEEK_SET);
    if (result == -1) {
      close(SM_fd);
      ImageStreamIO_printERROR("Error calling lseek() to 'stretch' the file");
      exit(0);
    }

    result = write(SM_fd, "", 1);
    if (result != 1) {
      close(SM_fd);
      perror("Error writing last byte of the file");
      exit(0);
    }

    map = mmap(0, sharedsize, PROT_READ | PROT_WRITE, MAP_SHARED, SM_fd, 0);
    if (map == MAP_FAILED) {
      close(SM_fd);
      perror("Error mmapping the file");
      exit(0);
    }

    image->md = (IMAGE_METADATA *)map;
    image->md[0].shared = 1;
    image->md[0].sem = 0;

    map += sizeof(IMAGE_METADATA);

    if (location == -1) {
      image->array.raw = map;
      map += datasharedsize;
    } else if (location >= 0) {
      image->array.raw = NULL;
    }
    image->kw = (IMAGE_KEYWORD *)(map);

  } else {
    image->shmfd = 0;
    image->memsize = 0;

    image->md = (IMAGE_METADATA *)malloc(sizeof(IMAGE_METADATA));
    image->md[0].shared = 0;
    if (NBkw > 0)
      image->kw = (IMAGE_KEYWORD *)malloc(sizeof(IMAGE_KEYWORD) * NBkw);
    else
      image->kw = NULL;
  }

  image->md[0].location = location;
  image->md[0].atype = atype;
  image->md[0].naxis = naxis;
  strncpy(image->name, name, 80);  // local name
  strncpy(image->md[0].name, name, 80);
  image->md[0].nelement = 1;
  for (i = 0; i < naxis; i++) {
    image->md[0].size[i] = size[i];
    image->md[0].nelement *= size[i];
  }
  image->md[0].NBkw = NBkw;

  ImageStreamIO_initialize_buffer(image);

  clock_gettime(CLOCK_REALTIME, &timenow);
  image->md[0].last_access =
      1.0 * timenow.tv_sec + 0.000000001 * timenow.tv_nsec;
  image->md[0].creation_time = image->md[0].last_access;
  image->md[0].write = 0;
  image->md[0].cnt0 = 0;
  image->md[0].cnt1 = 0;

  if (shared == 1)
    ImageStreamIO_createsem(image, 10);  // by default, create 10 semaphores
  else
    image->md[0].sem = 0;  // no semaphores

  // initialize keywords
  for (kw = 0; kw < image->md[0].NBkw; kw++) image->kw[kw].type = 'N';

  return EXIT_SUCCESS;
}

int ImageStreamIO_destroyIm(IMAGE *image) {
  if (image->memsize > 0) {
    char fname[200];

    // close and remove semlog
    sem_close(image->semlog);

    snprintf(fname, sizeof(fname), "/dev/shm/sem.%s_semlog", image->md[0].name);
    sem_unlink(fname);

    image->semlog = NULL;

    // close and remove all semaphores
    ImageStreamIO_destroysem(image);

    close(image->shmfd);

    // Get this before unmapping.
    ImageStreamIO_filename(fname, sizeof(fname), image->md[0].name);

    munmap(image->md, image->memsize);

    // Remove the file
    remove(fname);
  } else {
    free(image->array.UI8);

    free(image->md);
    if (image->kw) free(image->kw);
  }

  image->semlog = NULL;
  image->md = NULL;

  image->array.UI8 = NULL;

  image->semptr = NULL;
  image->kw = NULL;

  return 0;
}

int ImageStreamIO_openIm(IMAGE *image, const char *name) {
  return ImageStreamIO_read_sharedmem_image_toIMAGE(name, image);
}

void *ImageStreamIO_get_image_d_ptr(IMAGE *image) {
  if (image->array.raw != NULL) return image->array.raw;

  void *d_ptr = NULL;
  if (image->md[0].location >= 0) {
#ifdef HAVE_CUDA
    checkCudaErrors(cudaSetDevice(image->md[0].location));
    checkCudaErrors(cudaIpcOpenMemHandle(&d_ptr, image->md[0].cudaMemHandle,
                                         cudaIpcMemLazyEnablePeerAccess));
#else
    ImageStreamIO_printERROR(
        "Error calling ImageStreamIO_get_image_d_ptr(), CACAO needs to be "
        "compiled with -DUSE_CUDA=ON");
#endif
  } else {
    ImageStreamIO_printERROR(
        "Error calling ImageStreamIO_get_image_d_ptr(), wrong location");
  }
  return d_ptr;
}

int ImageStreamIO_read_sharedmem_image_toIMAGE(const char *name, IMAGE *image) {
  int SM_fd;
  char SM_fname[200];

  ImageStreamIO_filename(SM_fname, sizeof(SM_fname), name);

  SM_fd = open(SM_fname, O_RDWR);
  if (SM_fd == -1) {
    image->used = 0;
    ImageStreamIO_printERROR(SM_fname);
    return EXIT_FAILURE;
  }

  char sname[200];
  uint8_t *map;
  long s;
  struct stat file_stat;

  long snb = 0;
  int sOK = 1;

  fstat(SM_fd, &file_stat);
  printf("File %s size: %zd\n", SM_fname, file_stat.st_size);

  map =
      mmap(0, file_stat.st_size, PROT_READ | PROT_WRITE, MAP_SHARED, SM_fd, 0);
  if (map == MAP_FAILED) {
    close(SM_fd);
    perror("Error mmapping the file");
    exit(0);
  }

  image->memsize = file_stat.st_size;
  image->shmfd = SM_fd;
  image->md = (IMAGE_METADATA *)map;
  image->md[0].shared = 1;

  printf("image size = ");
  uint64_t size = 1;
  for (uint8_t axis = 0; axis < image->md[0].naxis; ++axis) {
    printf("%ld ", (long)image->md[0].size[axis]);
    size *= image->md[0].size[axis];
  }
  printf("\n");
  fflush(stdout);

  // some verification
  if (size > 10000000000) {
    printf("IMAGE \"%s\" SEEMS BIG... NOT LOADING\n", name);
    return EXIT_FAILURE;
  }
  for (uint8_t axis = 0; axis < image->md[0].naxis; ++axis) {
    if (image->md[0].size[axis] < 1) {
      printf("IMAGE \"%s\" AXIS %d SIZE < 1... NOT LOADING\n", name, axis);
      return EXIT_FAILURE;
    }
  }

  map += sizeof(IMAGE_METADATA);
  map += ImageStreamIO_offset_data(image, map);

  printf("%ld keywords\n", (long)image->md[0].NBkw);
  fflush(stdout);

  image->kw = (IMAGE_KEYWORD *)(map);

  int kw;
  for (kw = 0; kw < image->md[0].NBkw; kw++) {
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

  // mapv += sizeof(IMAGE_KEYWORD)*image->md[0].NBkw;

  strncpy(image->name, name, sizeof(name));

  // looking for semaphores
  while (sOK == 1) {
    snprintf(sname, sizeof(sname), "%s_sem%02ld", image->md[0].name, snb);
    sem_t *stest;
    if ((stest = sem_open(sname, 0, 0644, 0)) == SEM_FAILED)
      sOK = 0;
    else {
      sem_close(stest);
      snb++;
    }
  }
  printf("%ld semaphores detected  (image->md[0].sem = %d)\n", snb,
         (int)image->md[0].sem);

  //        image->md[0].sem = snb;
  image->semptr = (sem_t **)malloc(sizeof(sem_t *) * image->md[0].sem);
  for (s = 0; s < image->md[0].sem; s++) {
    snprintf(sname, sizeof(sname), "%s_sem%02ld", image->md[0].name, s);
    if ((image->semptr[s] = sem_open(sname, 0, 0644, 0)) == SEM_FAILED) {
      printf("ERROR: could not open semaphore %s -> (re-)CREATING semaphore\n",
             sname);

      if ((image->semptr[s] = sem_open(sname, O_CREAT, 0644, 1)) == SEM_FAILED)
        perror("semaphore initilization");
    }
  }

  snprintf(sname, sizeof(sname), "%s_semlog", image->md[0].name);
  if ((image->semlog = sem_open(sname, 0, 0644, 0)) == SEM_FAILED) {
    printf("ERROR: could not open semaphore %s\n", sname);
  }

  return EXIT_SUCCESS;
}

int ImageStreamIO_closeIm(IMAGE *image) {
  for (long s = 0; s < image->md[0].sem; s++) {
    sem_close(image->semptr[s]);
  }

  free(image->semptr);

  sem_close(image->semlog);

  return munmap(image->md, image->memsize);
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

/*
 * ## Purpose
 *
 * Destroy semaphore of a shmim
 *
 * ## Arguments
 *
 * @param[in]
 * image	IMAGE*
 * 			pointer to shmim
 */

int ImageStreamIO_destroysem(IMAGE *image) {
  long s;
  int r;
  char command[200];
  int semfile[100];

  // Remove semaphores if any
  if (image->md[0].sem > 0) {
    // Close existing semaphores ...
    for (s = 0; s < image->md[0].sem; s++) {
      sem_close(image->semptr[s]);
      char fname[200];
      sem_unlink(fname);

      // ... and remove associated files
      snprintf(fname, sizeof(fname), "/dev/shm/sem.%s_sem%02ld",
               image->md[0].name, s);
      remove(fname);
    }
    image->md[0].sem = 0;

    free(image->semptr);
    image->semptr = NULL;
  }
  return EXIT_SUCCESS;
}

/*
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
 */

int ImageStreamIO_createsem(IMAGE *image, long NBsem) {
  long s;
  int r;
  char command[200];
  int semfile[100];

  printf("Creating %ld semaphores\n", NBsem);

  // Remove pre-existing semaphores if any
  ImageStreamIO_destroysem(image);

  printf("malloc semptr %ld entries\n", NBsem);
  image->semptr = (sem_t **)malloc(sizeof(sem_t **) * NBsem);

  for (s = 0; s < NBsem; s++) {
    char sname[200];
    snprintf(sname, sizeof(sname), "%s_sem%02ld", image->md[0].name, s);

    if ((image->semptr[s] = sem_open(sname, O_CREAT, 0644, 0)) == SEM_FAILED) {
      perror("semaphore initilization");
    }

    image->md[0].sem =
        NBsem;  // Do this last so nobody accesses before init is done.
  }

  printf("image->md[0].sem = %ld\n", (long)image->md[0].sem);

  return EXIT_SUCCESS;
}

/*
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
 */
long ImageStreamIO_sempost(IMAGE *image, long index) {
  if (image->md[0].location >= 0) {
#ifdef HAVE_CUDA
    checkCudaErrors(cudaSetDevice(image->md[0].location));
    checkCudaErrors(cudaDeviceSynchronize());
#else
    ImageStreamIO_printERROR(
        "Error calling ImageStreamIO_sempost(), CACAO needs to be "
        "compiled with -DUSE_CUDA=ON");
#endif
  }

  if (index < 0) {
    long s;

    for (s = 0; s < image->md[0].sem; s++) {
      int semval;

      sem_getvalue(image->semptr[s], &semval);
      if (semval < SEMAPHORE_MAXVAL) sem_post(image->semptr[s]);
    }
  } else {
    if (index > image->md[0].sem - 1)
      printf("ERROR: image %s semaphore # %ld does no exist\n",
             image->md[0].name, index);
    else {
      int semval;

      sem_getvalue(image->semptr[index], &semval);
      if (semval < SEMAPHORE_MAXVAL) sem_post(image->semptr[index]);
    }
  }

  if (image->semlog != NULL) {
    int semval;

    sem_getvalue(image->semlog, &semval);
    if (semval < SEMAPHORE_MAXVAL) sem_post(image->semlog);
  }

  return EXIT_SUCCESS;
}

/*
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
 */
long ImageStreamIO_sempost_excl(IMAGE *image, long index) {
  long s;

  if (image->md[0].location >= 0) {
#ifdef HAVE_CUDA
    checkCudaErrors(cudaSetDevice(image->md[0].location));
    checkCudaErrors(cudaDeviceSynchronize());
#else
    ImageStreamIO_printERROR(
        "Error calling ImageStreamIO_sempost(), CACAO needs to be "
        "compiled with -DUSE_CUDA=ON");
#endif
  }

  for (s = 0; s < image->md[0].sem; s++) {
    if (s != index) {
      int semval;

      sem_getvalue(image->semptr[s], &semval);
      if (semval < SEMAPHORE_MAXVAL) sem_post(image->semptr[s]);
    }
  }

  if (image->semlog != NULL) {
    int semval;

    sem_getvalue(image->semlog, &semval);
    if (semval < SEMAPHORE_MAXVAL) sem_post(image->semlog);
  }

  return EXIT_SUCCESS;
}

/*
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
long ImageStreamIO_sempost_loop(IMAGE *image, long index, long dtus) {
  while (1) {
    ImageStreamIO_sempost(image, index);

    sleep(dtus);
  }

  return EXIT_SUCCESS;
}

/*
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
 */
long ImageStreamIO_semwait(IMAGE *image, long index) {
  if (index > image->md[0].sem - 1)
    printf("ERROR: image %s semaphore # %ld does not exist\n",
           image->md[0].name, index);
  else
    sem_wait(image->semptr[index]);

  return (EXIT_SUCCESS);
}

/*
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
 */
long ImageStreamIO_semflush(IMAGE *image, long index) {
  if (index < 0) {
    long s;

    for (s = 0; s < image->md[0].sem; s++) {
      int semval;
      int i;

      sem_getvalue(image->semptr[s], &semval);
      for (i = 0; i < semval; i++) sem_trywait(image->semptr[s]);
    }
  } else {
    if (index > image->md[0].sem - 1)
      printf("ERROR: image %s semaphore # %ld does not exist\n",
             image->md[0].name, index);
    else {
      long s;
      int semval;
      int i;

      s = index;
      sem_getvalue(image->semptr[s], &semval);
      for (i = 0; i < semval; i++) sem_trywait(image->semptr[s]);
    }
  }

  return (EXIT_SUCCESS);
}
