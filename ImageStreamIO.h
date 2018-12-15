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
 
#include "ImageStruct.h"

#ifdef __cplusplus
extern "C"
{
#endif
   

void __attribute__ ((constructor)) libinit_ImageStreamIO();
int_fast8_t init_ImageStreamIO();




/* =============================================================================================== */
/* =============================================================================================== */
/** @name ImageStreamIO - 0. Utilities                                                             */                           
/**@{                                                                                              */
/* =============================================================================================== */
/* =============================================================================================== */

/** @brief Get the standard stream filename.
  * 
  * Fills in the \p file_name string with the standard shared memory image path, e.g.
  *  \code
  *    char file_name[64];
  *    ImageStreamIO_filename(file_name, 64, "image00");
  *    printf("%s\n", file_name);
  *  \endcode
  * produces the output:
  *  \verbatim
  *    /tmp/image00.im.shm 
  8  \endverbatim
  *
  * \returns 0 on success
  * \returns -1 on error
  */
int ImageStreamIO_filename( char * file_name,    ///< [out] the file name string to fill in
                            size_t ssz,          ///< [in] the allocated size of file_name
                            const char * im_name ///< [in] the image name
                          );
                            
/** @brief Get the size in bytes from the data type code. 
  */
int ImageStreamIO_typesize( uint8_t atype /**< [in] the type code (see ImageStruct.h*/);

/** @brief Get the FITSIO BITPIX from the data type code. 
  */
int ImageStreamIO_bitpix( uint8_t atype /**< [in] the type code (see ImageStruct.h*/);

///@}

/* =============================================================================================== */
/* =============================================================================================== */
/** @name ImageStreamIO - 1. READ / WRITE STREAM                                                   */               
/**@{                                                                                              */
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

/** @brief Deallocate and remove an IMAGE structure.
  *
  * For a shared image:
  * Closes all semaphores, deallcoates sem pointers,
  * and removes associated files. Unmaps the shared memory
  * segment, and finally removes the file. Sets the metadata and
  * keyword pointers to NULL.
  * 
  * For a non-shred image:
  * Deallocates all arrays and sets pointers to NULL.
  * 
  * \returns 0 on success
  * \returns -1 on an error (currently no checks done)
  * 
  */
int ImageStreamIO_destroyIm( IMAGE *image /**< [in] The IMAGE structure to deallocate and remove from the system.*/);

/** @brief Connect to an existing shared memory image stream 
  * 
  * Wrapper for  \ref ImageStreamIO_read_sharedmem_image_toIMAGE
  */
int ImageStreamIO_openIm( IMAGE *image,    ///< [out] IMAGE structure which will be attached to the existing IMAGE
                          const char *name ///< [in] the name of the shared memory file will be SHAREDMEMDIR/<name>_im.shm
                        );

/** @brief Read / connect to existing shared memory image stream */
int ImageStreamIO_read_sharedmem_image_toIMAGE( const char *name, ///< [in] the name of the shared memory file to access, as in SHAREDMEMDIR/<name>_im.shm
                                                IMAGE *image      ///< [out] the IMAGE structure to connect to the stream
                                              );


/** @brief Close a shared memmory image stream.
  * 
  * For use in clients, detaches and cleans up memory used by non-owner process.
  * 
  * \returns 0 on success
  * \returns -1 on error
  * 
  */ 
int ImageStreamIO_closeIm(IMAGE * image /**< [in] A real-time image structure which contains the image data and meta-data.*/);

///@}

/* =============================================================================================== */
/* =============================================================================================== */
/** @name ImageStreamIO - 2. MANAGE SEMAPHORES                                                     */
/**@{                                                                                              */
/* =============================================================================================== */
/* =============================================================================================== */

/** @brief Create shmim semaphores 
 *
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
int ImageStreamIO_createsem(IMAGE *image, long NBsem);

/** @brief Post all shmim semaphores 
 *
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
long ImageStreamIO_sempost(IMAGE *image, long index);

/** @brief Post all shmim semaphores except one 
 *
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
long ImageStreamIO_sempost_excl(IMAGE *image, long index);



/** @brief Post shmim semaphores at regular time interval 
 *
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
long ImageStreamIO_sempost_loop(IMAGE *image, long index, long dtus);




/** @brief Get available semaphore index
 * 
 * 
 * 
 */
int ImageStreamIO_getsemwaitindex(IMAGE *image, int semindexdefault);


/** @brief Wait for semaphore 
 *
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
long ImageStreamIO_semwait(IMAGE *image, long index);
long ImageStreamIO_semtrywait(IMAGE *image, long index);
long ImageStreamIO_semtimedwait(IMAGE *image, long index, const struct timespec *semwts);



/** @brief Flush all semaphores of a shmim 
 *
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
long ImageStreamIO_semflush(IMAGE *image, long index);

///@}


#ifdef __cplusplus
} //extern "C"
#endif


#endif


