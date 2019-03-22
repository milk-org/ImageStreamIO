#ifndef IMAGE_ERROR_H
#define IMAGE_ERROR_H

#ifndef IMAGESTREAMIO_SUCCESS
#define IMAGESTREAMIO_SUCCESS        0 
#define IMAGESTREAMIO_FAILURE       1   // generic error code
#endif

#ifndef __STDC_LIB_EXT1__
typedef int errno_t;
#endif

#endif // IMAGE_ERROR_H
