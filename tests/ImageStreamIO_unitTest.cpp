#include "ImageStreamIO.h"
#include "gtest/gtest.h"
#include <limits.h>

/* A prefix to all names indicating ImageStreamIO Unit Tests */
#define SHM_NAME_PREFIX    "__ISIOUTs__"
#define SHM_NAME_ImageTest SHM_NAME_PREFIX "ImageTest"
#define SHM_NAME_CubeTest  SHM_NAME_PREFIX "CubeTest"
#define SHM_NAME_LocnTest  SHM_NAME_PREFIX "LocationTest"

namespace {

  uint32_t dims2[2] = {32, 32};
  uint32_t dims3[3] = {16, 16, 13};
  IMAGE imageTest;
  IMAGE circularbufferTest;

  int8_t cpuLocn = -1;   // Location of -1 => CPU-based shmim
  int8_t gpuLocn =  0;   // Location of  0 => Pretend GPU-based shmim
  int8_t badLocn = -2;   // Location of -2 => bad location

////////////////////////////////////////////////////////////////////////
// ImageStreamIO_creatIM_gpu - create  a shmim file 
////////////////////////////////////////////////////////////////////////
TEST(ImageStreamIOTestCreation, ImageCPUCreation) {

  char SM_fname[200];
  ImageStreamIO_filename(SM_fname, 200, SHM_NAME_ImageTest);

  fprintf(stderr,"[%s=>%s]=SM_fname\n",SHM_NAME_ImageTest, SM_fname);

  EXPECT_EQ(IMAGESTREAMIO_SUCCESS
           ,ImageStreamIO_createIm_gpu(&imageTest, SHM_NAME_ImageTest
                                      ,2, dims2, _DATATYPE_FLOAT
                                      ,cpuLocn, 0, 10, 10, MATH_DATA,0)
           );
}

TEST(ImageStreamIOTestCreation, ImageCPUSharedCreation) {

  EXPECT_EQ(IMAGESTREAMIO_SUCCESS
           ,ImageStreamIO_createIm_gpu(&imageTest, SHM_NAME_ImageTest
                                      ,2, dims2, _DATATYPE_FLOAT
                                      ,cpuLocn, 1, 10, 10, MATH_DATA,0)
           );
}

TEST(ImageStreamIOTestCreation, CubeCPUSharedCreationFailureDimension) {

  EXPECT_EQ(IMAGESTREAMIO_INVALIDARG
           ,ImageStreamIO_createIm_gpu(&circularbufferTest, SHM_NAME_CubeTest
                                      ,2, dims2, _DATATYPE_FLOAT
                                      ,cpuLocn, 1, 10, 10, CIRCULAR_BUFFER,1)
           );
}

TEST(ImageStreamIOTestCreation, CubeCPUSharedCreation) {

  EXPECT_EQ(IMAGESTREAMIO_SUCCESS
           ,ImageStreamIO_createIm_gpu(&circularbufferTest, SHM_NAME_CubeTest
                                      ,3, dims3,_DATATYPE_FLOAT
                                      ,cpuLocn, 1, 10, 10, CIRCULAR_BUFFER,1)
           );
}

////////////////////////////////////////////////////////////////////////
// ImageStreamIO_OpenIm - opening an existing shmim file 
// - Use ImageTest from above; assume it has not been destroyed
////////////////////////////////////////////////////////////////////////
TEST(ImageStreamIOTestOpen, ImageCPUSharedOpen) {

  EXPECT_EQ(IMAGESTREAMIO_SUCCESS
           ,ImageStreamIO_openIm(&imageTest, SHM_NAME_ImageTest)
           );
}

TEST(ImageStreamIOTestOpen, ImageCPUSharedOpenNotExist) {

  EXPECT_EQ(IMAGESTREAMIO_FILEOPEN
           ,ImageStreamIO_openIm(&imageTest
                                ,SHM_NAME_ImageTest "DoesNotExist")
           );
}

TEST(ImageStreamIOTestOpen, CubeCPUSharedOpen) {

  EXPECT_EQ(IMAGESTREAMIO_SUCCESS
           ,ImageStreamIO_openIm(&circularbufferTest, SHM_NAME_CubeTest)
           );
}

TEST(ImageStreamIOTestRead, ImageCPUSharedNbSlices) {

  EXPECT_EQ(IMAGESTREAMIO_SUCCESS
           ,ImageStreamIO_read_sharedmem_image_toIMAGE(SHM_NAME_ImageTest
                                                      ,&imageTest)
           );
  EXPECT_EQ(1, ImageStreamIO_nbSlices(&imageTest));
}

TEST(ImageStreamIOTestRead, CubeCPUSharedNbSlices) {

  EXPECT_EQ(IMAGESTREAMIO_SUCCESS
           ,ImageStreamIO_read_sharedmem_image_toIMAGE(SHM_NAME_CubeTest
                                                      ,&circularbufferTest)
           );
  EXPECT_EQ(13, ImageStreamIO_nbSlices(&circularbufferTest));
}

////////////////////////////////////////////////////////////////////////
// Location-related tests
////////////////////////////////////////////////////////////////////////
TEST(ImageStreamIOTestLocation, BadLocation) {

  EXPECT_EQ(IMAGESTREAMIO_FAILURE
           ,ImageStreamIO_createIm_gpu(&imageTest, SHM_NAME_LocnTest
                                      ,2, dims2,_DATATYPE_FLOAT
                                      ,badLocn, 1, 10, 10, MATH_DATA,0)
           );
}

#ifdef HAVE_CUDA
TEST(ImageStreamIOTestCreation, ImageGPUSharedCreation) {

  EXPECT_EQ(IMAGESTREAMIO_SUCCESS
           ,ImageStreamIO_createIm_gpu(&imageTest, SHM_NAME_LocnTest
                                      ,2, dims2, _DATATYPE_FLOAT
                                      ,gpuLocn, 1, 10, 10, MATH_DATA,0)
           );
}
#endif

// For GPU-based shmim, existing file (ImageTest from above) is an error
TEST(ImageStreamIOTestLocation, InitCpuLocation) {

  ASSERT_EQ(IMAGESTREAMIO_SUCCESS
           ,ImageStreamIO_createIm_gpu(&imageTest, SHM_NAME_LocnTest
                                      ,2, dims2, _DATATYPE_FLOAT
                                      ,cpuLocn, 1, 10, 10, MATH_DATA,0)
           );

  EXPECT_EQ(IMAGESTREAMIO_FILEEXISTS
           ,ImageStreamIO_createIm_gpu(&imageTest, SHM_NAME_LocnTest
                                      ,2, dims2, _DATATYPE_FLOAT
                                      ,gpuLocn, 1, 10, 10, MATH_DATA,0)
           );
}

} // namespace
