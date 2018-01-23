/**
 * @file    ImageCreate.h
 * @brief   Function prototypes for ImageCreate
 * 
 *  
 * @author  O. Guyon
 * @date    12 Jul 2017
 *
 * 
 * @bug No known bugs.
 * 
 */



#ifndef _IMAGESTREAMIO_H
#define _IMAGESTREAMIO_H
 
#ifdef __cplusplus
extern "C"
{
#endif
   

void __attribute__ ((constructor)) libinit_ImageStreamIO();
int_fast8_t init_ImageStreamIO();



/* =============================================================================================== */
/* =============================================================================================== */
/** @name ImageStreamIO - 1. READ / WRITE STREAM                                                   */                  
/* =============================================================================================== */
/* =============================================================================================== */

/** @brief Create shared memory image stream */
int ImageStreamIO_createIm( IMAGE *image,      ///< [out] IMAGE structure which will have its members allocated and initialized.
                            const char *name,  ///< [in] the name of the shared memory file will be SHAREDMEMDIR/<name>_im.shm
                            long naxis,        ///< [in] number of axes in the image.
                            uint32_t *size,    ///< [in] the size of the image along each axis.  Must have naxis elements.
                            uint8_t atype,     ///< [in] data type code
                            int shared,        ///< [in] if true then a shared memory buffer is allocated.  If false, only local storage is used.
                            int NBkw           ///< [in] the number of keywords to allocate.
                          );

/** @brief Read / connect to existing shared memory image stream */
long ImageStreamIO_read_sharedmem_image_toIMAGE( const char *name, ///< [in] the name of the shared memory file to access, as in SHAREDMEMDIR/<name>_im.shm
                                                 IMAGE *image      ///< [out] the IMAGE structure to connect to the stream
                                               );



/* =============================================================================================== */
/* =============================================================================================== */
/** @name ImageStreamIO - 2. MANAGE SEMAPHORES                                                     */
/* =============================================================================================== */
/* =============================================================================================== */

/** @brief Create shmim semaphores */
int ImageStreamIO_createsem(IMAGE *image, long NBsem);

/** @brief Post all shmim semaphores */
long ImageStreamIO_sempost(IMAGE *image, long index);

/** @brief Post all shmim semaphores except one */
long ImageStreamIO_sempost_excl(IMAGE *image, long index);

/** @brief Post shmim semaphores at regular time interval */
long ImageStreamIO_sempost_loop(IMAGE *image, long index, long dtus);

/** @brief Wait for semaphore */
long ImageStreamIO_semwait(IMAGE *image, long index);

/** @brief Flush all semaphores of a shmim */
long ImageStreamIO_semflush(IMAGE *image, long index);

///@}


#ifdef __cplusplus
} //extern "C"
#endif


#endif


