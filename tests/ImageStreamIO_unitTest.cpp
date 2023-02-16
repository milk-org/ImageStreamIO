#include "ImageStreamIO.h"
#include "gtest/gtest.h"
#include <limits.h>

/* A prefix to all names indicating ImageStreamIO Unit Tests */
#define SHM_NAME_PREFIX "__ISIOUTs__"

namespace {

TEST(ImageStreamIOTestCreation, ImageCPUCreation) {
  IMAGE imageTest;
  uint32_t dims[2] = {512, 512};

  EXPECT_EQ(IMAGESTREAMIO_SUCCESS, ImageStreamIO_createIm_gpu(&imageTest, SHM_NAME_PREFIX "ImageTest", 2,
                                                     dims, _DATATYPE_FLOAT, -1,
                                                     0, 10, 10, MATH_DATA,0));
}

TEST(ImageStreamIOTestCreation, ImageCPUSharedCreation) {
  IMAGE imageTest;
  uint32_t dims[2] = {512, 512};

  EXPECT_EQ(IMAGESTREAMIO_SUCCESS, ImageStreamIO_createIm_gpu(&imageTest, SHM_NAME_PREFIX "ImageTest", 2,
                                                     dims, _DATATYPE_FLOAT, -1,
                                                     1, 10, 10, MATH_DATA,0));
}

#ifdef HAVE_CUDA
TEST(ImageStreamIOTestCreation, ImageGPUSharedCreation) {
  IMAGE imageTest;
  uint32_t dims[2] = {512, 512};

  EXPECT_EQ(IMAGESTREAMIO_SUCCESS, ImageStreamIO_createIm_gpu(&imageTest, SHM_NAME_PREFIX "ImageTest", 2,
                                                     dims, _DATATYPE_FLOAT, 0,
                                                     1, 10, 10, MATH_DATA,0));
}
#endif

TEST(ImageStreamIOTestCreation, CubeCPUSharedCreationFailureDimension) {
  IMAGE circularbufferTest;
  uint32_t dims[2] = {512, 512};

  EXPECT_EQ(IMAGESTREAMIO_INVALIDARG, ImageStreamIO_createIm_gpu(
                              &circularbufferTest, SHM_NAME_PREFIX "CubeTest", 2, dims,
                              _DATATYPE_FLOAT, -1, 1, 10, 10, CIRCULAR_BUFFER,1));
}

TEST(ImageStreamIOTestCreation, CubeCPUSharedCreation) {
  IMAGE circularbufferTest;
  uint32_t dims[3] = {512, 512, 13};

  EXPECT_EQ(IMAGESTREAMIO_SUCCESS, ImageStreamIO_createIm_gpu(
                              &circularbufferTest, SHM_NAME_PREFIX "CubeTest", 3, dims,
                              _DATATYPE_FLOAT, -1, 1, 10, 10, CIRCULAR_BUFFER,1));
  EXPECT_EQ(circularbufferTest.array.raw, (void*)circularbufferTest.array.UI32);
}

TEST(ImageStreamIOTestOpen, ImageCPUSharedOpen) {
  IMAGE imageTest;

  EXPECT_EQ(IMAGESTREAMIO_SUCCESS, ImageStreamIO_openIm(&imageTest, SHM_NAME_PREFIX "ImageTest"));
  EXPECT_EQ(imageTest.array.raw, (void*)imageTest.array.UI16);
}

TEST(ImageStreamIOTestOpen, ImageCPUSharedOpenNotExist) {
  IMAGE imageTest;

  EXPECT_EQ(IMAGESTREAMIO_FILEOPEN, ImageStreamIO_openIm(&imageTest, SHM_NAME_PREFIX "ImageTestNo"));
  EXPECT_EQ(imageTest.array.raw, (void*)imageTest.array.SI16);
}

TEST(ImageStreamIOTestOpen, CubeCPUSharedOpen) {
  IMAGE circularbufferTest;

  EXPECT_EQ(IMAGESTREAMIO_SUCCESS,
            ImageStreamIO_openIm(&circularbufferTest, SHM_NAME_PREFIX "CubeTest"));
  EXPECT_EQ(circularbufferTest.array.raw, (void*)circularbufferTest.array.UI16);
}

TEST(ImageStreamIOTestRead, ImageCPUSharedNbSlices) {
  IMAGE imageTest;
  ImageStreamIO_read_sharedmem_image_toIMAGE(SHM_NAME_PREFIX "ImageTest", &imageTest);
  EXPECT_EQ(1, ImageStreamIO_nbSlices(&imageTest));
  EXPECT_EQ(imageTest.array.raw, (void*)imageTest.array.UI8);
}

TEST(ImageStreamIOTestRead, CubeCPUSharedNbSlices) {
  IMAGE circularbufferTest;
  ImageStreamIO_read_sharedmem_image_toIMAGE(SHM_NAME_PREFIX "CubeTest", &circularbufferTest);
  EXPECT_EQ(13, ImageStreamIO_nbSlices(&circularbufferTest));
  EXPECT_EQ(circularbufferTest.array.raw, (void*)circularbufferTest.array.SI8);
}

TEST(ImageStreamIOTestLocationBad, BadLocation) {
  IMAGE imageTest;
  uint32_t dims[2] = {64, 64};
  int8_t location = -2;

  EXPECT_EQ(IMAGESTREAMIO_FAILURE, ImageStreamIO_createIm_gpu(
                              &imageTest, SHM_NAME_PREFIX "LocationTest", 2, dims,
                              _DATATYPE_FLOAT, location, 1, 10, 10, MATH_DATA,0));
}

TEST(ImageStreamIOTestLocationInit, InitLocation) {
  IMAGE imageTest;
  uint32_t dims[2] = {64, 64};
  int8_t location = -1;

  EXPECT_EQ(IMAGESTREAMIO_SUCCESS, ImageStreamIO_createIm_gpu(
                              &imageTest, SHM_NAME_PREFIX "LocationTest", 2, dims,
                              _DATATYPE_FLOAT, location, 1, 10, 10, MATH_DATA,0));
}

TEST(ImageStreamIOTestLocationExists, ExistsLocation) {
  IMAGE imageTest;
  uint32_t dims[2] = {64, 64};
  int8_t location = 0;       // Location of 0 => Pretend GPU-based shmim

  EXPECT_EQ(IMAGESTREAMIO_FILEEXISTS, ImageStreamIO_createIm_gpu(
                              &imageTest, SHM_NAME_PREFIX "LocationTest", 2, dims,
                              _DATATYPE_FLOAT, location, 1, 10, 10, MATH_DATA,0));
}

} // namespace
