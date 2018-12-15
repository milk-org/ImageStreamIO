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

#include <stdint.h>
#include <unistd.h>
#include <malloc.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <signal.h> 

#include <semaphore.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <fcntl.h> // for open
#include <unistd.h> // for close
#include <errno.h>

#include <fitsio.h>

//Handle old fitsios
#ifndef ULONGLONG_IMG
#define ULONGLONG_IMG (80)
#endif


#include "ImageStreamIO.h"



#ifdef __MACH__
#include <mach/mach_time.h>
#define CLOCK_REALTIME 0
#define CLOCK_MONOTONIC 0
static int clock_gettime(int clk_id, struct mach_timespec *t){
    mach_timebase_info_data_t timebase;
    mach_timebase_info(&timebase);
    uint64_t time;
    time = mach_absolute_time();
    double nseconds = ((double)time * (double)timebase.numer)/((double)timebase.denom);
    double seconds = ((double)time * (double)timebase.numer)/((double)timebase.denom * 1e9);
    t->tv_sec = seconds;
    t->tv_nsec = nseconds;
    return EXIT_SUCCESS;
}
#else
#include <time.h>
#endif



static int INITSTATUS_ImageStreamIO = 0;




void __attribute__ ((constructor)) libinit_ImageStreamIO()
{
	if ( INITSTATUS_ImageStreamIO == 0 )
	{
		init_ImageStreamIO();
		INITSTATUS_ImageStreamIO = 1;
	}
}


int_fast8_t init_ImageStreamIO()
{
	// any initialization needed ?
	
	return EXIT_SUCCESS;
}





int ImageStreamIO_printERROR(const char *file, const char *func, int line, char *errmessage)
{
    fprintf(stderr,"%c[%d;%dmERROR [ FILE: %s   FUNCTION: %s   LINE: %d ]  %c[%d;m\n", (char) 27, 1, 31, file, func, line, (char) 27, 0);
    if( errno != 0)
    {
        char buff[256];
        
        //Test for which version of strerror_r we're using (XSI or GNU)
        #if ((_POSIX_C_SOURCE >= 200112L || _XOPEN_SOURCE >= 600) && !defined(_GNU_SOURCE))
           if( strerror_r( errno, buff, sizeof(buff) ) == 0 ) 
           {
              fprintf(stderr,"C Error: %s\n", buff );
           }
           else
             fprintf(stderr,"Unknown C Error\n");
        #else
           //GNU strerror_r does not necessarily use buff, and uses errno to report errors.
           int _errno = errno;
           errno = 0;
           char * estr = strerror_r( _errno, buff, sizeof(buff) );
        
           if(errno == 0) 
              fprintf(stderr,"%c[%d;%dmC Error: %s%c[%d;m\n", (char) 27, 1, 31, estr, 27, 0 );
           else 
              fprintf(stderr,"%c[%d;%dmUnknown C Error%c[%d;m\n", (char) 27, 1, 31, 27, 0);
           
           errno = _errno; //restore it in case it's used later.
        #endif
        
    }
    else
        fprintf(stderr,"%c[%d;%dmNo C error (errno = 0)%c[%d;m\n",(char) 27, 1, 31, 27, 0);

    fprintf(stderr,"%c[%d;%dm %s  %c[%d;m\n", (char) 27, 1, 31, errmessage, (char) 27, 0);

    return EXIT_SUCCESS;
}









/* =============================================================================================== */
/* =============================================================================================== */
/* @name 0. Utilities
 *  
 */
/* =============================================================================================== */
/* =============================================================================================== */

int ImageStreamIO_filename( char * file_name,    
                            size_t ssz,          
                            const char * im_name 
                          )
{
   int rv = snprintf(file_name, ssz, "%s/%s.im.shm", SHAREDMEMDIR, im_name); 

   if(rv > 0 && rv < ssz) return 0;
   else if(rv < 0)
   {
      ImageStreamIO_printERROR(__FILE__, __func__, __LINE__, strerror(errno));
      return -1;
   }
   else
   {
      ImageStreamIO_printERROR(__FILE__, __func__, __LINE__, "string not large enough for file name");
      return -1;
   }
}

int ImageStreamIO_typesize( uint8_t atype )
{
   switch(atype)
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
         ImageStreamIO_printERROR(__FILE__, __func__, __LINE__, "invalid type code");
         return -1;
   }
}

int ImageStreamIO_bitpix( uint8_t atype )
{
   switch(atype)
   {
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
         ImageStreamIO_printERROR(__FILE__, __func__, __LINE__, "bitpix not implemented for type");
         return -1;
   }
}
/* =============================================================================================== */
/* =============================================================================================== */
/* @name 1. READ / WRITE STREAM
 *  
 */
/* =============================================================================================== */
/* =============================================================================================== */


int ImageStreamIO_createIm( IMAGE       *image,
                            const char  *name,
                            long         naxis,
                            uint32_t    *size,
                            uint8_t      atype,
                            int          shared,
                            int          NBkw
                          )
{
    long i,ii;
    time_t lt;
    long nelement;
    struct timespec timenow;

    IMAGE_METADATA *map;
    char *mapv; // pointed cast in bytes

    int kw;
    char comment[80];
    char kname[16];

    nelement = 1;
    for(i=0; i<naxis; i++)
        nelement*=size[i];

    // compute total size to be allocated
    if(shared==1)
    {
        char sname[200];


        // create semlog
        size_t sharedsize = 0; // shared memory size in bytes

        snprintf(sname, sizeof(sname), "%s_semlog", name);
        remove(sname);
        image->semlog = NULL;

        if ((image->semlog = sem_open(sname, O_CREAT, 0644, 1)) == SEM_FAILED)
            perror("semaphore creation / initilization");
        else
            sem_init(image->semlog, 1, 0);


        sharedsize = sizeof(IMAGE_METADATA);

        if(atype == _DATATYPE_UINT8)
            sharedsize += nelement*SIZEOF_DATATYPE_UINT8;
        if(atype == _DATATYPE_INT8)
            sharedsize += nelement*SIZEOF_DATATYPE_INT8;

        if(atype == _DATATYPE_UINT16)
            sharedsize += nelement*SIZEOF_DATATYPE_UINT16;
        if(atype == _DATATYPE_INT16)
            sharedsize += nelement*SIZEOF_DATATYPE_INT16;

        if(atype == _DATATYPE_INT32)
            sharedsize += nelement*SIZEOF_DATATYPE_INT32;
        if(atype == _DATATYPE_UINT32)
            sharedsize += nelement*SIZEOF_DATATYPE_UINT32;


        if(atype == _DATATYPE_INT64)
            sharedsize += nelement*SIZEOF_DATATYPE_INT64;

        if(atype == _DATATYPE_UINT64)
            sharedsize += nelement*SIZEOF_DATATYPE_UINT64;


        if(atype == _DATATYPE_FLOAT)
            sharedsize += nelement*SIZEOF_DATATYPE_FLOAT;

        if(atype == _DATATYPE_DOUBLE)
            sharedsize += nelement*SIZEOF_DATATYPE_DOUBLE;

        if(atype == _DATATYPE_COMPLEX_FLOAT)
            sharedsize += nelement*SIZEOF_DATATYPE_COMPLEX_FLOAT;

        if(atype == _DATATYPE_COMPLEX_DOUBLE)
            sharedsize += nelement*SIZEOF_DATATYPE_COMPLEX_DOUBLE;


        sharedsize += NBkw*sizeof(IMAGE_KEYWORD);
        sharedsize += 2*IMAGE_NB_SEMAPHORE*sizeof(pid_t); // one read PID array, one write PID array

        char SM_fname[200];
        ImageStreamIO_filename(SM_fname, 200, name);

        int SM_fd; // shared memory file descriptor
        SM_fd = open(SM_fname, O_RDWR | O_CREAT | O_TRUNC, (mode_t)0600);
        if (SM_fd == -1) {
            perror("Error opening file for writing");
            exit(0);
        }




        image->shmfd = SM_fd;
        image->memsize = sharedsize;

        int result;
        result = lseek(SM_fd, sharedsize-1, SEEK_SET);
        if (result == -1) {
            close(SM_fd);
            ImageStreamIO_printERROR(__FILE__,__func__,__LINE__, "Error calling lseek() to 'stretch' the file");
            exit(0);
        }

        result = write(SM_fd, "", 1);
        if (result != 1) {
            close(SM_fd);
            perror("Error writing last byte of the file");
            exit(0);
        }

        map = (IMAGE_METADATA*) mmap(0, sharedsize, PROT_READ | PROT_WRITE, MAP_SHARED, SM_fd, 0);
        if (map == MAP_FAILED) {
            close(SM_fd);
            perror("Error mmapping the file");
            exit(0);
        }

        printf("shared memory space = %ld bytes\n", sharedsize); //TEST

        image->md = (IMAGE_METADATA*) map;
        image->md[0].shared = 1;
        image->md[0].sem = 0;
    }
    else
    {
        image->shmfd = 0;
        image->memsize = 0;

        image->md = (IMAGE_METADATA*) malloc(sizeof(IMAGE_METADATA));
        image->md[0].shared = 0;
        if(NBkw>0)
            image->kw = (IMAGE_KEYWORD*) malloc(sizeof(IMAGE_KEYWORD)*NBkw);
        else
            image->kw = NULL;
    }


    image->md[0].atype = atype;
    image->md[0].naxis = naxis;
    strncpy(image->name, name, 80); // local name
    strncpy(image->md[0].name, name, 80);
    for(i=0; i<naxis; i++)
        image->md[0].size[i] = size[i];
    image->md[0].NBkw = NBkw;


    if(atype == _DATATYPE_UINT8)
    {
        if(shared==1)
        {
            mapv = (char*) map;
            mapv += sizeof(IMAGE_METADATA);
            image->array.UI8 = (uint8_t*) (mapv);
            memset(image->array.UI8, '\0', nelement*sizeof(uint8_t));
            mapv += sizeof(uint8_t)*nelement;
            image->kw = (IMAGE_KEYWORD*) (mapv);
        }
        else
            image->array.UI8 = (uint8_t*) calloc ((size_t) nelement, sizeof(uint8_t));


        if(image->array.UI8 == NULL)
        {
            ImageStreamIO_printERROR(__FILE__,__func__,__LINE__,"memory allocation failed");
            fprintf(stderr,"%c[%d;%dm", (char) 27, 1, 31);
            fprintf(stderr,"Image name = %s\n",name);
            fprintf(stderr,"Image size = ");
            fprintf(stderr,"%ld", (long) size[0]);
            for(i=1; i<naxis; i++)
                fprintf(stderr,"x%ld", (long) size[i]);
            fprintf(stderr,"\n");
            fprintf(stderr,"Requested memory size = %ld elements = %f Mb\n", (long) nelement,1.0/1024/1024*nelement*sizeof(uint8_t));
            fprintf(stderr," %c[%d;m",(char) 27, 0);
            exit(0);
        }
    }

    if(atype == _DATATYPE_INT8)
    {
        if(shared==1)
        {
            mapv = (char*) map;
            mapv += sizeof(IMAGE_METADATA);
            image->array.SI8 = (int8_t*) (mapv);
            memset(image->array.SI8, '\0', nelement*sizeof(int8_t));
            mapv += sizeof(int8_t)*nelement;
            image->kw = (IMAGE_KEYWORD*) (mapv);
        }
        else
            image->array.SI8 = (int8_t*) calloc ((size_t) nelement, sizeof(int8_t));


        if(image->array.SI8 == NULL)
        {
            ImageStreamIO_printERROR(__FILE__,__func__,__LINE__,"memory allocation failed");
            fprintf(stderr,"%c[%d;%dm", (char) 27, 1, 31);
            fprintf(stderr,"Image name = %s\n",name);
            fprintf(stderr,"Image size = ");
            fprintf(stderr,"%ld", (long) size[0]);
            for(i=1; i<naxis; i++)
                fprintf(stderr,"x%ld", (long) size[i]);
            fprintf(stderr,"\n");
            fprintf(stderr,"Requested memory size = %ld elements = %f Mb\n", (long) nelement,1.0/1024/1024*nelement*sizeof(int8_t));
            fprintf(stderr," %c[%d;m",(char) 27, 0);
            exit(0);
        }
    }



    if(atype == _DATATYPE_UINT16)
    {
        if(shared==1)
        {
            mapv = (char*) map;
            mapv += sizeof(IMAGE_METADATA);
            image->array.UI16 = (uint16_t*) (mapv);
            memset(image->array.UI16, '\0', nelement*sizeof(uint16_t));
            mapv += sizeof(uint16_t)*nelement;
            image->kw = (IMAGE_KEYWORD*) (mapv);
        }
        else
            image->array.UI16 = (uint16_t*) calloc ((size_t) nelement, sizeof(uint16_t));

        if(image->array.UI16 == NULL)
        {
            ImageStreamIO_printERROR(__FILE__,__func__,__LINE__,"memory allocation failed");
            fprintf(stderr,"%c[%d;%dm", (char) 27, 1, 31);
            fprintf(stderr,"Image name = %s\n",name);
            fprintf(stderr,"Image size = ");
            fprintf(stderr,"%ld", (long) size[0]);
            for(i=1; i<naxis; i++)
                fprintf(stderr,"x%ld", (long) size[i]);
            fprintf(stderr,"\n");
            fprintf(stderr,"Requested memory size = %ld elements = %f Mb\n", (long) nelement, 1.0/1024/1024*nelement*sizeof(uint16_t));
            fprintf(stderr," %c[%d;m",(char) 27, 0);
            exit(0);
        }
    }

    if(atype == _DATATYPE_INT16)
    {
        if(shared==1)
        {
            mapv = (char*) map;
            mapv += sizeof(IMAGE_METADATA);
            image->array.SI16 = (int16_t*) (mapv);
            memset(image->array.SI16, '\0', nelement*sizeof(int16_t));
            mapv += sizeof(int16_t)*nelement;
            image->kw = (IMAGE_KEYWORD*) (mapv);
        }
        else
            image->array.SI16 = (int16_t*) calloc ((size_t) nelement, sizeof(int16_t));

        if(image->array.SI16 == NULL)
        {
            ImageStreamIO_printERROR(__FILE__,__func__,__LINE__,"memory allocation failed");
            fprintf(stderr,"%c[%d;%dm", (char) 27, 1, 31);
            fprintf(stderr,"Image name = %s\n",name);
            fprintf(stderr,"Image size = ");
            fprintf(stderr,"%ld", (long) size[0]);
            for(i=1; i<naxis; i++)
                fprintf(stderr,"x%ld", (long) size[i]);
            fprintf(stderr,"\n");
            fprintf(stderr,"Requested memory size = %ld elements = %f Mb\n", (long) nelement, 1.0/1024/1024*nelement*sizeof(int16_t));
            fprintf(stderr," %c[%d;m",(char) 27, 0);
            exit(0);
        }
    }


    if(atype == _DATATYPE_UINT32)
    {
        if(shared==1)
        {
            mapv = (char*) map;
            mapv += sizeof(IMAGE_METADATA);
            image->array.UI32 = (uint32_t*) (mapv);
            memset(image->array.UI32, '\0', nelement*sizeof(uint32_t));
            mapv += sizeof(uint32_t)*nelement;
            image->kw = (IMAGE_KEYWORD*) (mapv);
        }
        else
            image->array.UI32 = (uint32_t*) calloc ((size_t) nelement, sizeof(uint32_t));

        if(image->array.UI32 == NULL)
        {
            ImageStreamIO_printERROR(__FILE__,__func__,__LINE__,"memory allocation failed");
            fprintf(stderr,"%c[%d;%dm", (char) 27, 1, 31);
            fprintf(stderr,"Image name = %s\n",name);
            fprintf(stderr,"Image size = ");
            fprintf(stderr,"%ld", (long) size[0]);
            for(i=1; i<naxis; i++)
                fprintf(stderr,"x%ld", (long) size[i]);
            fprintf(stderr,"\n");
            fprintf(stderr,"Requested memory size = %ld elements = %f Mb\n", (long) nelement,1.0/1024/1024*nelement*sizeof(uint32_t));
            fprintf(stderr," %c[%d;m",(char) 27, 0);
            exit(0);
        }
    }



    if(atype == _DATATYPE_INT32)
    {
        if(shared==1)
        {
            mapv = (char*) map;
            mapv += sizeof(IMAGE_METADATA);
            image->array.SI32 = (int32_t*) (mapv);
            memset(image->array.SI32, '\0', nelement*sizeof(int32_t));
            mapv += sizeof(int32_t)*nelement;
            image->kw = (IMAGE_KEYWORD*) (mapv);
        }
        else
            image->array.SI32 = (int32_t*) calloc ((size_t) nelement, sizeof(int32_t));

        if(image->array.SI32 == NULL)
        {
            ImageStreamIO_printERROR(__FILE__,__func__,__LINE__,"memory allocation failed");
            fprintf(stderr,"%c[%d;%dm", (char) 27, 1, 31);
            fprintf(stderr,"Image name = %s\n",name);
            fprintf(stderr,"Image size = ");
            fprintf(stderr,"%ld", (long) size[0]);
            for(i=1; i<naxis; i++)
                fprintf(stderr,"x%ld", (long) size[i]);
            fprintf(stderr,"\n");
            fprintf(stderr,"Requested memory size = %ld elements = %f Mb\n", (long) nelement,1.0/1024/1024*nelement*sizeof(int32_t));
            fprintf(stderr," %c[%d;m",(char) 27, 0);
            exit(0);
        }
    }



    if(atype == _DATATYPE_UINT64)
    {
        if(shared==1)
        {
            mapv = (char*) map;
            mapv += sizeof(IMAGE_METADATA);
            image->array.UI64 = (uint64_t*) (mapv);
            memset(image->array.UI64, '\0', nelement*sizeof(uint64_t));
            mapv += sizeof(uint64_t)*nelement;
            image->kw = (IMAGE_KEYWORD*) (mapv);
        }
        else
            image->array.UI64 = (uint64_t*) calloc ((size_t) nelement, sizeof(uint64_t));

        if(image->array.SI64 == NULL)
        {
            ImageStreamIO_printERROR(__FILE__,__func__,__LINE__,"memory allocation failed");
            fprintf(stderr,"%c[%d;%dm", (char) 27, 1, 31);
            fprintf(stderr,"Image name = %s\n",name);
            fprintf(stderr,"Image size = ");
            fprintf(stderr,"%ld", (long) size[0]);
            for(i=1; i<naxis; i++)
                fprintf(stderr,"x%ld", (long) size[i]);
            fprintf(stderr,"\n");
            fprintf(stderr,"Requested memory size = %ld elements = %f Mb\n", (long) nelement,1.0/1024/1024*nelement*sizeof(uint64_t));
            fprintf(stderr," %c[%d;m",(char) 27, 0);
            exit(0);
        }
    }

    if(atype == _DATATYPE_INT64)
    {
        if(shared==1)
        {
            mapv = (char*) map;
            mapv += sizeof(IMAGE_METADATA);
            image->array.SI64 = (int64_t*) (mapv);
            memset(image->array.SI64, '\0', nelement*sizeof(int64_t));
            mapv += sizeof(int64_t)*nelement;
            image->kw = (IMAGE_KEYWORD*) (mapv);
        }
        else
            image->array.SI64 = (int64_t*) calloc ((size_t) nelement, sizeof(int64_t));

        if(image->array.SI64 == NULL)
        {
            ImageStreamIO_printERROR(__FILE__,__func__,__LINE__,"memory allocation failed");
            fprintf(stderr,"%c[%d;%dm", (char) 27, 1, 31);
            fprintf(stderr,"Image name = %s\n",name);
            fprintf(stderr,"Image size = ");
            fprintf(stderr,"%ld", (long) size[0]);
            for(i=1; i<naxis; i++)
                fprintf(stderr,"x%ld", (long) size[i]);
            fprintf(stderr,"\n");
            fprintf(stderr,"Requested memory size = %ld elements = %f Mb\n", (long) nelement,1.0/1024/1024*nelement*sizeof(int64_t));
            fprintf(stderr," %c[%d;m",(char) 27, 0);
            exit(0);
        }
    }


    if(atype == _DATATYPE_FLOAT)	{
        if(shared==1)
        {
            mapv = (char*) map;
            mapv += sizeof(IMAGE_METADATA);
            image->array.F = (float*) (mapv);
            memset(image->array.F, '\0', nelement*sizeof(float));
            mapv += sizeof(float)*nelement;
            image->kw = (IMAGE_KEYWORD*) (mapv);
        }
        else {
            // printf("allocating %ld bytes\n", nelement*sizeof(float));//TEST
            image->array.F = (float*) calloc ((size_t) nelement, sizeof(float));
        }

        if(image->array.F == NULL)
        {
            ImageStreamIO_printERROR(__FILE__,__func__,__LINE__,"memory allocation failed");
            fprintf(stderr,"%c[%d;%dm", (char) 27, 1, 31);
            fprintf(stderr,"Image name = %s\n",name);
            fprintf(stderr,"Image size = ");
            fprintf(stderr,"%ld", (long) size[0]);
            for(i=1; i<naxis; i++)
                fprintf(stderr,"x%ld", (long) size[i]);
            fprintf(stderr,"\n");
            fprintf(stderr,"Requested memory size = %ld elements = %f Mb\n", (long) nelement,1.0/1024/1024*nelement*sizeof(float));
            fprintf(stderr," %c[%d;m",(char) 27, 0);
            exit(0);
        }
    }

    if(atype == _DATATYPE_DOUBLE)
    {
        if(shared==1)
        {
            mapv = (char*) map;
            mapv += sizeof(IMAGE_METADATA);
            image->array.D = (double*) (mapv);
            memset(image->array.D, '\0', nelement*sizeof(double));
            mapv += sizeof(double)*nelement;
            image->kw = (IMAGE_KEYWORD*) (mapv);
        }
        else
            image->array.D = (double*) calloc ((size_t) nelement, sizeof(double));

        if(image->array.D == NULL)
        {
            ImageStreamIO_printERROR(__FILE__,__func__,__LINE__,"memory allocation failed");
            fprintf(stderr,"%c[%d;%dm", (char) 27, 1, 31);
            fprintf(stderr,"Image name = %s\n",name);
            fprintf(stderr,"Image size = ");
            fprintf(stderr,"%ld", (long) size[0]);
            for(i=1; i<naxis; i++)
                fprintf(stderr,"x%ld", (long) size[i]);
            fprintf(stderr,"\n");
            fprintf(stderr,"Requested memory size = %ld elements = %f Mb\n", (long) nelement,1.0/1024/1024*nelement*sizeof(double));
            fprintf(stderr," %c[%d;m",(char) 27, 0);
            exit(0);
        }
    }

    if(atype == _DATATYPE_COMPLEX_FLOAT)
    {
        if(shared==1)
        {
            mapv = (char*) map;
            mapv += sizeof(IMAGE_METADATA);
            image->array.CF = (complex_float*) (mapv);
            memset(image->array.CF, '\0', nelement*sizeof(complex_float));
            mapv += sizeof(complex_float)*nelement;
            image->kw = (IMAGE_KEYWORD*) (mapv);
        }
        else
            image->array.CF = (complex_float*) calloc ((size_t) nelement, sizeof(complex_float));

        if(image->array.CF == NULL)
        {
            ImageStreamIO_printERROR(__FILE__,__func__,__LINE__,"memory allocation failed");
            fprintf(stderr,"%c[%d;%dm", (char) 27, 1, 31);
            fprintf(stderr,"Image name = %s\n",name);
            fprintf(stderr,"Image size = ");
            fprintf(stderr,"%ld", (long) size[0]);
            for(i=1; i<naxis; i++)
                fprintf(stderr,"x%ld", (long) size[i]);
            fprintf(stderr,"\n");
            fprintf(stderr,"Requested memory size = %ld elements = %f Mb\n", (long) nelement,1.0/1024/1024*nelement*sizeof(complex_float));
            fprintf(stderr," %c[%d;m",(char) 27, 0);
            exit(0);
        }
    }

    if(atype == _DATATYPE_COMPLEX_DOUBLE)
    {
        if(shared==1)
        {
            mapv = (char*) map;
            mapv += sizeof(IMAGE_METADATA);
            image->array.CD = (complex_double*) (mapv);
            memset(image->array.CD, '\0', nelement*sizeof(complex_double));
            mapv += sizeof(complex_double)*nelement;
            image->kw = (IMAGE_KEYWORD*) (mapv);
        }
        else
            image->array.CD = (complex_double*) calloc ((size_t) nelement,sizeof(complex_double));

        if(image->array.CD == NULL)
        {
            ImageStreamIO_printERROR(__FILE__,__func__,__LINE__,"memory allocation failed");
            fprintf(stderr,"%c[%d;%dm", (char) 27, 1, 31);
            fprintf(stderr,"Image name = %s\n",name);
            fprintf(stderr,"Image size = ");
            fprintf(stderr,"%ld", (long) size[0]);
            for(i=1; i<naxis; i++)
                fprintf(stderr,"x%ld", (long) size[i]);
            fprintf(stderr,"\n");
            fprintf(stderr,"Requested memory size = %ld elements = %f Mb\n", (long) nelement,1.0/1024/1024*nelement*sizeof(complex_double));
            fprintf(stderr," %c[%d;m",(char) 27, 0);
            exit(0);
        }
    }

	if(shared==1)
        {
			mapv += sizeof(IMAGE_KEYWORD)*image->md[0].NBkw;
			image->semReadPID = (pid_t*) (mapv);
			
            mapv += sizeof(pid_t)*IMAGE_NB_SEMAPHORE;
            image->semWritePID = (pid_t*) (mapv);
        }
	

    clock_gettime(CLOCK_REALTIME, &timenow);
    image->md[0].last_access = 1.0*timenow.tv_sec + 0.000000001*timenow.tv_nsec;
    image->md[0].creation_time = image->md[0].last_access;
    image->md[0].write = 0;
    image->md[0].cnt0 = 0;
    image->md[0].cnt1 = 0;
    image->md[0].nelement = nelement;

    if(shared==1)
    {
        ImageStreamIO_createsem(image, IMAGE_NB_SEMAPHORE); // IMAGE_NB_SEMAPHORE defined in ImageStruct.h
        
        int semindex;
        for(semindex=0; semindex<IMAGE_NB_SEMAPHORE; semindex++)
        {
			image->semReadPID[semindex] = -1;
			image->semWritePID[semindex] = -1;
		}
    
    }
    else
    {
        image->md[0].sem = 0; // no semaphores
	}



    // initialize keywords
    for(kw=0; kw<image->md[0].NBkw; kw++)
        image->kw[kw].type = 'N';


    return EXIT_SUCCESS;
}









int ImageStreamIO_destroyIm( IMAGE *image )
{
   if(image->memsize > 0)
   {
      char fname[200];
      
      //close and remove semlog
      sem_close(image->semlog);
      
      
      snprintf(fname, sizeof(fname), "/dev/shm/sem.%s_semlog", image->md[0].name);
      sem_unlink(fname);
      
      image->semlog = NULL;
      
      //close and remove all semaphores
      if( image->md[0].sem>0 )
      {
         // Close existing semaphores ...
         long s;
         for(s=0; s < image->md[0].sem; s++)
         {
            sem_close(image->semptr[s]);
            
            
            snprintf(fname, sizeof(fname), "/dev/shm/sem.%s_sem%02ld", image->md[0].name, s);
            sem_unlink(fname);
         }
         image->md[0].sem = 0;

         free(image->semptr);
         image->semptr = NULL;
      }
      
      close(image->shmfd);
   
      
      //Get this before unmapping.
      ImageStreamIO_filename(fname, sizeof(fname), image->md[0].name);
      
      munmap(image->md, image->memsize);
      
      //Remove the file
      remove(fname);
   }
   else
   {
      free(image->array.UI8);
      
      free(image->md);
      if(image->kw) free(image->kw);
   }
   
   image->semlog = NULL;
   image->md = NULL;
   
   image->array.UI8 = NULL;
   
   image->semptr = NULL;
   image->kw = NULL;
   
   return 0;
      
}  


int ImageStreamIO_openIm(
	     IMAGE *image,     
    const char *name  
                        )
{
   return ImageStreamIO_read_sharedmem_image_toIMAGE(name, image);
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

int ImageStreamIO_read_sharedmem_image_toIMAGE(
    const char *name,
    IMAGE *image
)
{
    int SM_fd;
    char SM_fname[200];
    int rval = -1;


    ImageStreamIO_filename(SM_fname, sizeof(SM_fname), name);

    SM_fd = open(SM_fname, O_RDWR);
    if(SM_fd==-1)
    {
        image->used = 0;
        ImageStreamIO_printERROR(__FILE__, __func__, __LINE__, SM_fname);
        rval = -1;
        return(rval);
    }

    char sname[200];
    IMAGE_METADATA *map;
    long s;
    struct stat file_stat;

    long snb = 0;
    int sOK = 1;


    rval = 0; // we assume by default success

    fstat(SM_fd, &file_stat);
//    printf("File %s size: %zd\n", SM_fname, file_stat.st_size);




    map = (IMAGE_METADATA*) mmap(0, file_stat.st_size, PROT_READ | PROT_WRITE, MAP_SHARED, SM_fd, 0);
    if (map == MAP_FAILED) {
        close(SM_fd);
        perror("Error mmapping the file");
        rval = -1;
        exit(0);
    }



    image->memsize = file_stat.st_size;

    image->shmfd = SM_fd;



    image->md = map;

    uint8_t atype;
    atype = image->md[0].atype;
    image->md[0].shared = 1;


 //   printf("image size = %ld %ld\n", (long) image->md[0].size[0], (long) image->md[0].size[1]);
 //   fflush(stdout);
    // some verification
    if(image->md[0].size[0]*image->md[0].size[1]>10000000000)
    {
        printf("IMAGE \"%s\" SEEMS BIG... NOT LOADING\n", name);
        rval = -1;
        return(rval);
    }
    if(image->md[0].size[0]<1)
    {
        printf("IMAGE \"%s\" AXIS SIZE < 1... NOT LOADING\n", name);
        rval = -1;
        return(rval);
    }
    if(image->md[0].size[1]<1)
    {
        printf("IMAGE \"%s\" AXIS SIZE < 1... NOT LOADING\n", name);
        rval = -1;
        return(rval);
    }


    char *mapv;
    mapv = (char*) map;
    mapv += sizeof(IMAGE_METADATA);



   // printf("atype = %d\n", (int) atype);
   // fflush(stdout);

    if(atype == _DATATYPE_UINT8)
    {
//        printf("atype = UINT8\n");
        image->array.UI8 = (uint8_t*) mapv;
        mapv += SIZEOF_DATATYPE_UINT8 * image->md[0].nelement;
    }

    if(atype == _DATATYPE_INT8)
    {
 //       printf("atype = INT8\n");
        image->array.SI8 = (int8_t*) mapv;
        mapv += SIZEOF_DATATYPE_INT8 * image->md[0].nelement;
    }

    if(atype == _DATATYPE_UINT16)
    {
  //      printf("atype = UINT16\n");
        image->array.UI16 = (uint16_t*) mapv;
        mapv += SIZEOF_DATATYPE_UINT16 * image->md[0].nelement;
    }

    if(atype == _DATATYPE_INT16)
    {
  //      printf("atype = INT16\n");
        image->array.SI16 = (int16_t*) mapv;
        mapv += SIZEOF_DATATYPE_INT16 * image->md[0].nelement;
    }

    if(atype == _DATATYPE_UINT32)
    {
  //      printf("atype = UINT32\n");
        image->array.UI32 = (uint32_t*) mapv;
        mapv += SIZEOF_DATATYPE_UINT32 * image->md[0].nelement;
    }

    if(atype == _DATATYPE_INT32)
    {
   //     printf("atype = INT32\n");
        image->array.SI32 = (int32_t*) mapv;
        mapv += SIZEOF_DATATYPE_INT32 * image->md[0].nelement;
    }

    if(atype == _DATATYPE_UINT64)
    {
   //     printf("atype = UINT64\n");
        image->array.UI64 = (uint64_t*) mapv;
        mapv += SIZEOF_DATATYPE_UINT64 * image->md[0].nelement;
    }

    if(atype == _DATATYPE_INT64)
    {
   //     printf("atype = INT64\n");
        image->array.SI64 = (int64_t*) mapv;
        mapv += SIZEOF_DATATYPE_INT64 * image->md[0].nelement;
    }

    if(atype == _DATATYPE_FLOAT)
    {
   //     printf("atype = FLOAT\n");
        image->array.F = (float*) mapv;
        mapv += SIZEOF_DATATYPE_FLOAT * image->md[0].nelement;
    }

    if(atype == _DATATYPE_DOUBLE)
    {
   //     printf("atype = DOUBLE\n");
        image->array.D = (double*) mapv;
        mapv += SIZEOF_DATATYPE_COMPLEX_DOUBLE * image->md[0].nelement;
    }

    if(atype == _DATATYPE_COMPLEX_FLOAT)
    {
   //     printf("atype = COMPLEX_FLOAT\n");
        image->array.CF = (complex_float*) mapv;
        mapv += SIZEOF_DATATYPE_COMPLEX_FLOAT * image->md[0].nelement;
    }

    if(atype == _DATATYPE_COMPLEX_DOUBLE)
    {
   //     printf("atype = COMPLEX_DOUBLE\n");
        image->array.CD = (complex_double*) mapv;
        mapv += SIZEOF_DATATYPE_COMPLEX_DOUBLE * image->md[0].nelement;
    }



    //printf("%ld keywords\n", (long) image->md[0].NBkw);
    //fflush(stdout);

    image->kw = (IMAGE_KEYWORD*) (mapv);

    int kw;
    /*for(kw=0; kw<image->md[0].NBkw; kw++)
    {
        if(image->kw[kw].type == 'L')
            printf("%d  %s %ld %s\n", kw, image->kw[kw].name, image->kw[kw].value.numl, image->kw[kw].comment);
        if(image->kw[kw].type == 'D')
            printf("%d  %s %lf %s\n", kw, image->kw[kw].name, image->kw[kw].value.numf, image->kw[kw].comment);
        if(image->kw[kw].type == 'S')
            printf("%d  %s %s %s\n", kw, image->kw[kw].name, image->kw[kw].value.valstr, image->kw[kw].comment);
    }*/




    mapv += sizeof(IMAGE_KEYWORD)*image->md[0].NBkw;
    image->semReadPID = (pid_t*) (mapv);

    mapv += sizeof(pid_t)*image->md[0].sem;
    image->semWritePID = (pid_t*) (mapv);




    // mapv += sizeof(IMAGE_KEYWORD)*image->md[0].NBkw;

    strncpy(image->name, name, strlen(name));


    // looking for semaphores
    while(sOK==1)
    {
        snprintf(sname, sizeof(sname), "%s_sem%02ld", image->md[0].name, snb);
        sem_t *stest;
        if((stest = sem_open(sname, 0, 0644, 0))== SEM_FAILED)
            sOK = 0;
        else
        {
            sem_close(stest);
            snb++;
        }
    }
    //printf("%ld semaphores detected  (image->md[0].sem = %d)\n", snb, (int) image->md[0].sem);




    //        image->md[0].sem = snb;
    image->semptr = (sem_t**) malloc(sizeof(sem_t*) * image->md[0].sem);
    for(s=0; s < image->md[0].sem; s++)
    {
        snprintf(sname, sizeof(sname), "%s_sem%02ld", image->md[0].name, s);
        if ((image->semptr[s] = sem_open(sname, 0, 0644, 0))== SEM_FAILED) {
            printf("ERROR: could not open semaphore %s -> (re-)CREATING semaphore\n", sname);

            if ((image->semptr[s] = sem_open(sname, O_CREAT, 0644, 1)) == SEM_FAILED) {
                perror("semaphore initialization");
            }
            else
                sem_init(image->semptr[s], 1, 0);
        }
    }


    snprintf(sname, sizeof(sname), "%s_semlog", image->md[0].name);
    if ((image->semlog = sem_open(sname, 0, 0644, 0))== SEM_FAILED) {
        printf("ERROR: could not open semaphore %s -> (re-)CREATING semaphore\n", sname);

        if ((image->semlog = sem_open(sname, O_CREAT, 0644, 1)) == SEM_FAILED) {
            perror("semaphore initialization");
        }
        else
            sem_init(image->semlog, 1, 0);
    }




    return(rval);
}






int ImageStreamIO_closeIm(IMAGE * image)
{
    long s;
    for(s=0; s<image->md[0].sem; s++)
    {
        sem_close(image->semptr[s]);
    }

    free(image->semptr);

    sem_close(image->semlog);

    return munmap( image->md, image->memsize);
}





/* =============================================================================================== */
/* =============================================================================================== */
/* @name 2. MANAGE SEMAPHORES
 *  
 */
/* =============================================================================================== */
/* =============================================================================================== */




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
 */

int ImageStreamIO_createsem(
	IMAGE *image, 
	long   NBsem
)
{
    long s;
    int r;
    char command[200];
    int semfile[100];

	
	printf("Creating %ld semaphores\n", NBsem);
	// fprintf(stderr, "%d here", __LINE__);

	// Remove pre-existing semaphores if any
    if((image->md[0].sem>0) && (image->md[0].sem != NBsem))
    {
        // Close existing semaphores ...
        for(s=0; s < image->md[0].sem; s++)
            sem_close(image->semptr[s]);
        image->md[0].sem = 0;

		// ... and remove associated files
		long s1;
        for(s1=NBsem; s1<100; s1++)
        {
			char fname[200];
            snprintf(fname,sizeof(fname), "/dev/shm/sem.%s_sem%02ld", image->md[0].name, s1);
            remove(fname);
        }
        free(image->semptr);
        image->semptr = NULL;
    }

   
    if(image->md[0].sem == 0)
    {
        printf("malloc semptr %ld entries\n", NBsem);
        image->semptr = (sem_t**) malloc(sizeof(sem_t**)*NBsem);


        for(s=0; s<NBsem; s++)
        {
			char sname[200];
            snprintf(sname, sizeof(sname), "%s_sem%02ld", image->md[0].name, s);
            
            //Note that if sem already exists, the initialization to 1 is ignored!
            if ((image->semptr[s] = sem_open(sname, O_CREAT, 0644, 1)) == SEM_FAILED) {
                    perror("semaphore initilization");
                }
        }
        
        image->md[0].sem = NBsem; //Do this last so nobody accesses before init is done.
    }
    
    printf("image->md[0].sem = %ld\n", (long) image->md[0].sem);
    
    return EXIT_SUCCESS;
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
 */
long ImageStreamIO_sempost(
    IMAGE *image,
    long   index
)
{
    pid_t writeProcessPID;

    writeProcessPID = getpid();


    if(index<0)
    {
        long s;

        for(s=0; s<image->md[0].sem; s++)
        {
            int semval;

            sem_getvalue(image->semptr[s], &semval);

            if(semval<SEMAPHORE_MAXVAL)
                sem_post(image->semptr[s]);

            image->semWritePID[s] = writeProcessPID;
        }
    }
    else
    {
        if(index>image->md[0].sem-1)
            printf("ERROR: image %s semaphore # %ld does no exist\n", image->md[0].name, index);
        else
        {
            int semval;

            sem_getvalue(image->semptr[index], &semval);
            if(semval<SEMAPHORE_MAXVAL)
                sem_post(image->semptr[index]);

            image->semWritePID[index] = writeProcessPID;
        }
    }

    if(image->semlog!=NULL)
    {
        int semval;

        sem_getvalue(image->semlog, &semval);
        if(semval<SEMAPHORE_MAXVAL)
            sem_post(image->semlog);
    }

    return EXIT_SUCCESS;
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
 */
long ImageStreamIO_sempost_excl(IMAGE *image, long index)
{
    long s;
    pid_t writeProcessPID;
    
    writeProcessPID = getpid();

    for(s=0; s<image->md[0].sem; s++)
        {
			if(s!=index)
			{
			    int semval;

				sem_getvalue(image->semptr[s], &semval);
				if(semval<SEMAPHORE_MAXVAL)
					sem_post(image->semptr[s]);
				
				image->semWritePID[s] = writeProcessPID;
			}
        }
        
    if(image->semlog!=NULL)
    {
		int semval;
    
        sem_getvalue(image->semlog, &semval);
        if(semval<SEMAPHORE_MAXVAL)
            sem_post(image->semlog);
    }

    return EXIT_SUCCESS;
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
long ImageStreamIO_sempost_loop(IMAGE *image, long index, long dtus)
{
	pid_t writeProcessPID;
	
	writeProcessPID = getpid();
	
    while(1)
    {
        if(index<0)
        {
			long s;
			
            for(s=0; s<image->md[0].sem; s++)
            {
				int semval;
				
                sem_getvalue(image->semptr[s], &semval);
                if(semval<SEMAPHORE_MAXVAL)
                    sem_post(image->semptr[s]);
                
                image->semWritePID[s] = writeProcessPID;
            }
        }
        else
        {
            if(index>image->md[0].sem-1)
                printf("ERROR: image %s semaphore # %ld does no exist\n", image->md[0].name, index);
            else
            {
				int semval;
				
                sem_getvalue(image->semptr[index], &semval);
                if(semval<SEMAPHORE_MAXVAL)
                    sem_post(image->semptr[index]);
                
                image->semWritePID[index] = writeProcessPID;
            }
        }
        
        if(image->semlog!=NULL)
		{
		int semval;
    
        sem_getvalue(image->semlog, &semval);
        if(semval<SEMAPHORE_MAXVAL)
            sem_post(image->semlog);
		}        
        
        sleep(dtus);
    }

    return EXIT_SUCCESS;
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
int ImageStreamIO_getsemwaitindex(IMAGE *image, int semindexdefault)
{
	pid_t readProcessPID;
	int OK = 0; // toggles to 1 when semaphore is found
	int semindex;
	int rval = -1;
	
	readProcessPID = getpid();
	
	// Check if default semindex is available
	semindex = semindexdefault;
	if( (image->semReadPID[semindex]==0) || (getpgid(image->semReadPID[semindex]) < 0))
	{
		OK = 1;
		rval = semindex;
	}
	
	// if not, look for available semindex 
	semindex = 0;
	while( (OK == 0) && (semindex < image->md[0].sem) )
	{
		if( (image->semReadPID[semindex]==0) || (getpgid(image->semReadPID[semindex]) < 0))
		{
			rval = semindex;
			OK = 1;
		}
		semindex++;
	}

	rval = semindexdefault; // remove this line when fully tested
	image->semReadPID[rval] = readProcessPID;
    
    return(rval);
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
 */
int ImageStreamIO_semwait(IMAGE *image, int index)
{
	int rval = -1;
	
    if(index>image->md[0].sem-1)
    {
        printf("ERROR: image %s semaphore # %d does not exist\n", image->md[0].name, index);
    }
    else
        rval = sem_wait(image->semptr[index]);

    return(rval);
}

int ImageStreamIO_semtrywait(IMAGE *image, int index)
{
	int rval = -1;
	
    if(index>image->md[0].sem-1)
    {
        printf("ERROR: image %s semaphore # %d does not exist\n", image->md[0].name, index);
    }
    else
        rval = sem_trywait(image->semptr[index]);

    return(rval);
}

int ImageStreamIO_semtimedwait(IMAGE *image, int index, const struct timespec *semwts)
{
	int rval = -1;
	
    if(index>image->md[0].sem-1)
    {
        printf("ERROR: image %s semaphore # %d does not exist\n", image->md[0].name, index);
		return(-1);
    }
    else
        rval = sem_timedwait(image->semptr[index], semwts);

    return(rval);
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
 */
long ImageStreamIO_semflush(IMAGE *image, long index)
{
    if(index<0)
    {
		long s;
		
        for(s=0; s<image->md[0].sem; s++)
        {
			int semval;
			int i;
			
            sem_getvalue(image->semptr[s], &semval);
            for(i=0; i<semval; i++)
                sem_trywait(image->semptr[s]);
        }
    }
    else
    {
        if(index>image->md[0].sem-1)
            printf("ERROR: image %s semaphore # %ld does not exist\n", image->md[0].name, index);
        else
        {
			long s;
			int semval;
			int i;
			
            s = index;
            sem_getvalue(image->semptr[s], &semval);
            for(i=0; i<semval; i++)
                sem_trywait(image->semptr[s]);

        }
    }
    
    return(EXIT_SUCCESS);
}

