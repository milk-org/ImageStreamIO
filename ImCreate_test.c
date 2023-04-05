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
 * Updates the image every ~ 1ms, forever...
 * A disk is rotating around the center of the image
 *
 */

#include <math.h>
#include <stdio.h>
//#include <stdlib.h>
#include "ImageStreamIO.h"

int main()
{
    IMAGE imarray[1];              // pointer to array of images
    uint32_t imsize[2] = { 512, 512 }; // image size is 512 x 512
    long naxis = sizeof(imsize) / (sizeof *imsize);  // # of axes

    // Data type; see file ImageStruct.h for list of supported types
    uint8_t atype = _DATATYPE_FLOAT;

    int shared = 1;                // 1 if image in shared mem
    int NBkw = 10;                 // number of keywords allowed

    // create an image in shared memory
    ImageStreamIO_createIm(imarray, "imtest00", naxis, imsize, atype, shared, NBkw, 1);

    long ii, jj;              // Image column and row indices
    float x0, y0, xc, yc;     // Image center; disk center
    long dtus = 1000;         // Wait 1ms = 1000 microseconds
    float dangle = 1/64.0;    // Angle step size
    float angle = 0.0;        // Angle of disc center, CW from right
    float r = 100.0;          // Radius of disk center from image center

    x0 = 0.5*imarray->md->size[0];           // Calculate enter of image
    y0 = 0.5*imarray->md->size[1];

    // Writes a disk in image
    // - Location of disk center rotates around image center
    while (1)
    {
        // Calculate disk center location based on angle,
        // then increment angle for the next loop pass
        xc = x0 + r*cos(angle);
        yc = y0 + r*sin(angle);
        angle += dangle;
        if(angle > 2.0*M_PI) { angle -= 2.0 * M_PI; }

        imarray->md->write = 1;         // Poor-man's mutex when writing

        // ->array is union; ->array.F is float pointer to image
        float* dotF = imarray->array.F;
        for(jj=0; jj<imarray->md->size[1]; jj++)            // loop rows
        {
            float dy = jj-yc;
            float dy2 = dy*dy;
            for(ii=0; ii<imarray->md->size[0]; ii++)     // loop columns
            {
                float dx = ii-xc;
                // Brightness is highest at the center
                // of the disk and fades radially
                *(dotF++) = cos(0.003*dx)*cos(0.003*dy)*exp(-1.0e-4*(dx*dx+dy2));

            }
        }
        // Post all semaphores (index = -1)
        ImageStreamIO_sempost(imarray, -1);

        imarray->md->write = 0; // Done writing; release mutex
        imarray->md->cnt0++;
        imarray->md->cnt1++;

        usleep(dtus);           // Wait 1ms
    }
    return 0;
}
