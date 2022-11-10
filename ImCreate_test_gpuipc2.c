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

	// allocate memory for array of images
	imarray = (IMAGE*) malloc(sizeof(IMAGE)*NBIMAGES);
	
	// create an image in shared memory
	ImageStreamIO_read_sharedmem_image_toIMAGE("imtest00", &imarray[0]);
	void *d_ptr = ImageStreamIO_get_image_d_ptr(&imarray[0]);

	float *h_ptr = (float*)malloc(imarray[0].md[0].size[0]*imarray[0].md[0].size[1]*sizeof(float));
	cudaMemcpy(h_ptr, d_ptr, imarray[0].md[0].size[0]*imarray[0].md[0].size[1]*sizeof(float),
		cudaMemcpyDeviceToHost);

	printf("Read in SHM\n");	
	for(int i=0; i<10 /*imsize[0]*imsize[1]*/; i++){
		printf("%f ", h_ptr[i]);
	}
	printf("\n");

	for(int i=0; i<imarray[0].md[0].size[0]*imarray[0].md[0].size[1]; i++){
		h_ptr[i]=i;
	}
	cudaMemcpy(d_ptr, h_ptr, imarray[0].md[0].size[0]*imarray[0].md[0].size[1]*sizeof(float),
		cudaMemcpyHostToDevice);

	// POST ALL SEMAPHORES
	printf("Sending update\n");
	ImageStreamIO_sempost(&imarray[0], -1);
	
	imarray[0].md[0].write = 0; // Done writing data
	imarray[0].md[0].cnt0++;
	imarray[0].md[0].cnt1++;
	
	printf("Wrote in SHM\n");	
	for(int i=0; i<10 /*imsize[0]*imsize[1]*/; i++){
		printf("%f ", h_ptr[i]);
	}
	printf("\n");

	free(h_ptr);
	free(imarray);
	
	return 0;
}
