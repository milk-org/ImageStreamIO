/*
 * Example code to write image in shared memory
 *
 * compile with:
 * gcc ImCreate_img.c ImageStreamIO.c -o ImCreate_img -lm -lpthread
 * gcc ImCreate_img.c ImageStreamIO.c -DHAVE_CUDA -o ImCreate_test -lm -lpthread -I/opt/cuda/include -L/opt/cuda/lib64 -lcudart
 *
 * Required files in compilation directory :
 * ImCreate_img.c   : source code (this file)
 * ImageStreamIO.c   : ImageStreamIO source code
 * ImageStreamIO.h   : ImageCreate function prototypes
 * ImageStruct.h     : Image structure definition
 *
 * EXECUTION:
 * ./ImCreate_img
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
    IMAGE imarray;    // pointer to array of images
    long naxis;        // number of axis
    uint8_t atype;     // data type
    uint32_t *imsize;  // image size
    int shared;        // 1 if image in shared memory
    int NBkw;          // number of keywords supported

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

    // allocate space for 10 keywords
    NBkw = 10;


    // create an image in shared memory
    ImageStreamIO_createIm_gpu(&imarray,
        "imtest00",
        naxis,
        imsize,
        atype,
        -1,
        shared,
        IMAGE_NB_SEMAPHORE,
        NBkw,
        MATH_DATA);

    free(imsize);

    snprintf(imarray.kw[0].name,
             sizeof(imarray.kw[0].name),
             "%s", "keyword_long");
    imarray.kw[0].type = 'L';
    imarray.kw[0].value.numl = 42;

    snprintf(imarray.kw[1].name,
             sizeof(imarray.kw[1].name),
             "%s", "keyword_float");
    imarray.kw[1].type = 'D';
    imarray.kw[1].value.numf = 3.141592;

    snprintf(imarray.kw[2].name,
             sizeof(imarray.kw[2].name),
             "%s", "keyword_string");
    imarray.kw[2].type = 'S';
    snprintf(imarray.kw[2].value.valstr,
             sizeof(imarray.kw[2].value.valstr),
             "%s", "Hello!");

    float angle;
    float r;
    float r1;
    long ii, jj;
    float x, y, x0, y0, xc, yc;
    // float squarerad=20;
    long dtus = 100000; // update every 1ms
    float dangle = 0.02;

    int s;
    int semval;
    float *current_image;

    // writes a square in image
    // square location rotates around center
    angle = 0.0;
    r = 100.0;
    x0 = 0.5*imarray.md->size[0];
    y0 = 0.5*imarray.md->size[1];
    while (1)
    {
        // disk location
        xc = x0 + r*cos(angle);
        yc = y0 + r*sin(angle);


        SHMIM_WRITE_ACQUIRE(imarray.md);

        for(ii=0; ii<imarray.md->size[0]; ii++)
            for(jj=0; jj<imarray.md->size[1]; jj++)
            {
                x = 1.0*ii;
                y = 1.0*jj;
                float dx = x-xc;
                float dy = y-yc;
                imarray.array.F[ii*imarray.md->size[1]+jj] = cosf(0.03f*dx)*cosf(0.03f*dy)*expf(-1.0e-4f*(dx*dx+dy*dy));

                //if( (x-xc<squarerad) && (x-xc>-squarerad) && (y-yc<squarerad) && (y-yc>-squarerad))
                //	imarray.array.F[jj*imarray.md->size[0]+ii] = 1.0f;
                //else
                //	imarray.array.F[jj*imarray.md->size[0]+ii] = 0.0f;
            }
        imarray.md->cnt1 = 0;
        SHMIM_CNT0_INCREMENT(imarray.md);
        // POST ALL SEMAPHORES
        ImageStreamIO_sempost(&imarray, -1);

        SHMIM_WRITE_RELEASE(imarray.md);

        usleep(dtus);
        angle += dangle;
        if(angle > 2.0*3.141592)
            angle -= 2.0*3.141592;
        //printf("Wrote square at position xc = %16f  yc = %16f\n", xc, yc);
        //fflush(stdout);
    }

    return 0;
}
