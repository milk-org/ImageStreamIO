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
 

int_fast8_t init_ImageStreamIO();



/* =============================================================================================== */
/* =============================================================================================== */
/** @name ImageStreamIO - 1. READ / WRITE STREAM                                                   */                  
/* =============================================================================================== */
/* =============================================================================================== */

/** @brief Create shared memory image stream */
int ImageStreamIO_createIm(IMAGE *image, const char *name, long naxis, uint32_t *size, uint8_t atype, int shared, int NBkw);

/** @brief Read / connect to existing shared memory image stream */
long ImageStreamIO_read_sharedmem_image_toIMAGE(const char *name, IMAGE *image);



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




#endif


