/*
 * Example code to write image in shared memory
 *
 * compile with:
 * gcc ImCreate_test.c ImageStreamIO.c -lm -lpthread
 *
 * Required files in compilation directory :
 * ImCreate_test.c   : source code (this file)
 * ImageStreamIO.c   : ImageStreamIO source code
 * ImageStreamIO.h   : ImageCreate function prototypes
 * ImageStruct.h     : Image structure definition
 *
 * EXECUTION:
 * ./a.out
 * (no argument)
 *
 * Creates an image imtest00 in shared memory
 * Updates the image every ~ 10ms, forever...
 * A square is rotating around the center of the image
 *
 */




#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "ImageStruct.h"
#include "ImageStreamIO.h"

#include <cuda_runtime_api.h>



int main()
{
	IMAGE *imarray;    // pointer to array of images
	int NBIMAGES = 1;  // can hold 1 image
	long naxis;        // number of axis
	uint8_t atype;     // data type
	uint32_t *imsize;  // image size
	int shared;        // 1 if image in shared memory
	int location;      // -1 if image in CPU memory, >=0 in GPU mem
	int NBkw;          // number of keywords supported

	// allocate memory for array of images
	imarray = (IMAGE*) malloc(sizeof(IMAGE)*NBIMAGES);


	// image will be 2D
	naxis = 2;

	// image size will be 512 x 512
	imsize = (uint32_t *) malloc(sizeof(uint32_t)*naxis);
	imsize[0] = 512;
	imsize[1] = 512;

	// image will be float type
	// see file ImageStruct.h for list of supported types
	atype = _DATATYPE_FLOAT;

	// image will be in shared memory
	shared = 1;
	location = 0; // on GPU0

	// allocate space for 10 keywords
	NBkw = 1;

	// create an image in shared memory
	ImageStreamIO_createIm_gpu(&imarray[0], "imtest00", naxis, imsize, atype, location, shared, IMAGE_NB_SEMAPHORE, NBkw, CIRCULAR_BUFFER | ZAXIS_TEMPORAL, 0);
	void *d_ptr = ImageStreamIO_get_image_d_ptr(&imarray[0]);

	// cudaMemset(d_ptr, 0, imsize[0]*imsize[1]*sizeof(float));

	float *h_ptr = (float*)malloc(imsize[0]*imsize[1]*sizeof(float));
	for(int i=0; i<imarray[0].md[0].size[0]*imarray[0].md[0].size[1]; i++){
		h_ptr[i]=-i;
	}

	cudaMemcpy(d_ptr, h_ptr, imsize[0]*imsize[1]*sizeof(float),
		cudaMemcpyHostToDevice);

	printf("ImCreate_test_gpuipc wrote in SHM\n");
	for(int i=0; i<10 /*imsize[0]*imsize[1]*/; i++){
		printf("%f ", h_ptr[i]);
	}
	printf("\n");

	printf("ImCreate_test_gpuipc is waiting update\n");
	ImageStreamIO_semwait(&imarray[0], 0);
	while(imarray[0].md[0].write );

	printf("ImCreate_test_gpuipc reads in SHM\n");
	cudaMemcpy(h_ptr, d_ptr, imsize[0]*imsize[1]*sizeof(float),
		cudaMemcpyDeviceToHost);
	for(int i=0; i<10 /*imsize[0]*imsize[1]*/; i++){
		printf("%f ", h_ptr[i]);
	}
	printf("\n");



	free(h_ptr);
	free(imsize);
	free(imarray);

	return 0;
}
