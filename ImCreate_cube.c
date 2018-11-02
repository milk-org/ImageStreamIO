/*
 * Example code to write image in shared memory
 * 
 * compile with:
 * gcc ImCreate_cube.c ImageStreamIO.c -o ImCreate_cube -lm -lpthread 
 * 
 * Required files in compilation directory :
 * ImCreate_cube.c   : source code (this file)
 * ImageStreamIO.c   : ImageStreamIO source code
 * ImageStreamIO.h   : ImageCreate function prototypes
 * ImageStruct.h     : Image structure definition
 * 
 * EXECUTION:
 * ./ImCreate_cube  
 * (no argument)
 * 
 * Creates a circular buffer imtest00 in shared memory
 * Updates the image every ~ 10ms, forever...
 * A square is rotating around the center of the image
 * 
 */




#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "ImageStruct.h"
#include "ImageStreamIO.h"




int main()
{
	IMAGE *imarray;    // pointer to array of images
	int NBIMAGES = 1;  // can hold 1 image
	long naxis;        // number of axis
	uint8_t atype;     // data type
	uint32_t *imsize;  // image size 
	int shared;        // 1 if image in shared memory
	int NBkw;          // number of keywords supported



	

	// allocate memory for array of images
	imarray = (IMAGE*) malloc(sizeof(IMAGE)*NBIMAGES);

	
	// image will be a cube
	naxis = 3;
	
	// image size will be 512 x 512
	imsize = (uint32_t *) malloc(sizeof(uint32_t)*naxis);
	imsize[0] = 3;
	imsize[1] = 512;
	imsize[2] = 512;
	
	// image will be float type
	// see file ImageStruct.h for list of supported types
	atype = _DATATYPE_FLOAT;
	
	// image will be in shared memory
	shared = 1;
	
	// allocate space for 10 keywords
	NBkw = 10;

	
	// create an image in shared memory
	ImageStreamIO_createIm(&imarray[0], "imtest00", naxis, imsize, atype, shared, NBkw);



	float angle; 
	float r;
	float r1;
	long ii, jj;
	float x, y, x0, y0, xc, yc;
	// float squarerad=20;
	long dtus = 1000; // update every 1ms
	float dangle = 0.02;
	
	int s;
	int semval;
	float *current_image;

	// writes a square in image
	// square location rotates around center
	angle = 0.0;
	r = 100.0;
	x0 = 0.5*imarray[0].md[0].size[1];
	y0 = 0.5*imarray[0].md[0].size[2];
	while (1)
	{
		// disk location
		xc = x0 + r*cos(angle);
		yc = y0 + r*sin(angle);
		
		
		imarray[0].md[0].write = 1; // set this flag to 1 when writing data

		imarray[0].md[0].cnt1++;
		if(imarray[0].md[0].cnt1 == imarray[0].md[0].size[0])
			imarray[0].md[0].cnt1=0;
        printf("%d %d\r", imarray[0].md[0].cnt0, imarray[0].md[0].cnt1);

		current_image = &imarray[0].array.F[imarray[0].md[0].cnt1 * imarray[0].md[0].size[1] * imarray[0].md[0].size[2]];
        // printf("%d, %x \n", imarray[0].md[0].cnt1 * imarray[0].md[0].size[1] * imarray[0].md[0].size[2], current_image);
		for(ii=0; ii<imarray[0].md[0].size[1]; ii++)
			for(jj=0; jj<imarray[0].md[0].size[2]; jj++)
			{
				x = 1.0*ii;
				y = 1.0*jj;
				float dx = x-xc;
				float dy = y-yc;
				current_image[ii*imarray[0].md[0].size[1]+jj] = cos(0.03*dx)*cos(0.03*dy)*exp(-1.0e-4*(dx*dx+dy*dy));
				
				//if( (x-xc<squarerad) && (x-xc>-squarerad) && (y-yc<squarerad) && (y-yc>-squarerad))
				//	imarray[0].array.F[jj*imarray[0].md[0].size[0]+ii] = 1.0;
				//else
				//	imarray[0].array.F[jj*imarray[0].md[0].size[0]+ii] = 0.0;
			}
		// POST ALL SEMAPHORES
		ImageStreamIO_sempost(&imarray[0], -1);
		
		imarray[0].md[0].write = 0; // Done writing data
		imarray[0].md[0].cnt0++;
				
		usleep(dtus);
		angle += dangle;
		if(angle > 2.0*M_PI)
			angle -= 2.0*M_PI;
		//printf("Wrote square at position xc = %16f  yc = %16f\n", xc, yc);
		//fflush(stdout);
	}
	


	free(imsize);
	free(imarray);
	
	return 0;
}
