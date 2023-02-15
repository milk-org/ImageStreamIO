#include "ImageStreamIO.h"
#include "gtest/gtest.h"
#include <limits.h>

namespace {

TEST(ImageStreamIOTestCreation, ImageCPUCreation) {
  IMAGE imageTest;
  uint32_t dims[2] = {512, 512};

  EXPECT_EQ(EXIT_SUCCESS, ImageStreamIO_createIm_gpu(&imageTest, "ImageTest", 2,
                                                     dims, _DATATYPE_FLOAT, -1,
                                                     0, 10, 10, MATH_DATA,0));
}

TEST(ImageStreamIOTestCreation, ImageCPUSharedCreation) {
  IMAGE imageTest;
  uint32_t dims[2] = {512, 512};

  EXPECT_EQ(EXIT_SUCCESS, ImageStreamIO_createIm_gpu(&imageTest, "ImageTest", 2,
                                                     dims, _DATATYPE_FLOAT, -1,
                                                     1, 10, 10, MATH_DATA,0));
}

#ifdef HAVE_CUDA
TEST(ImageStreamIOTestCreation, ImageGPUSharedCreation) {
  IMAGE imageTest;
  uint32_t dims[2] = {512, 512};

  EXPECT_EQ(EXIT_SUCCESS, ImageStreamIO_createIm_gpu(&imageTest, "ImageTest", 2,
                                                     dims, _DATATYPE_FLOAT, 0,
                                                     1, 10, 10, MATH_DATA,0));
}
#endif

TEST(ImageStreamIOTestCreation, CubeCPUSharedCreationFailureDimension) {
  IMAGE circularbufferTest;
  uint32_t dims[2] = {512, 512};

  EXPECT_EQ(IMAGESTREAMIO_INVALIDARG, ImageStreamIO_createIm_gpu(
                              &circularbufferTest, "CubeTest", 2, dims,
                              _DATATYPE_FLOAT, -1, 1, 10, 10, CIRCULAR_BUFFER,1));
}

TEST(ImageStreamIOTestCreation, CubeCPUSharedCreation) {
  IMAGE circularbufferTest;
  uint32_t dims[3] = {512, 512, 13};

  EXPECT_EQ(EXIT_SUCCESS, ImageStreamIO_createIm_gpu(
                              &circularbufferTest, "CubeTest", 3, dims,
                              _DATATYPE_FLOAT, -1, 1, 10, 10, CIRCULAR_BUFFER,1));
}

TEST(ImageStreamIOTestOpen, ImageCPUSharedOpen) {
  IMAGE imageTest;

  EXPECT_EQ(EXIT_SUCCESS, ImageStreamIO_openIm(&imageTest, "ImageTest"));
}

TEST(ImageStreamIOTestOpen, ImageCPUSharedOpenNotExist) {
  IMAGE imageTest;

  EXPECT_EQ(IMAGESTREAMIO_FILEOPEN, ImageStreamIO_openIm(&imageTest, "ImageTestNo"));
}

TEST(ImageStreamIOTestOpen, CubeCPUSharedOpen) {
  IMAGE circularbufferTest;

  EXPECT_EQ(EXIT_SUCCESS,
            ImageStreamIO_openIm(&circularbufferTest, "CubeTest"));
}

TEST(ImageStreamIOTestRead, ImageCPUSharedNbSlices) {
  IMAGE imageTest;
  ImageStreamIO_read_sharedmem_image_toIMAGE("ImageTest", &imageTest);
  EXPECT_EQ(1, ImageStreamIO_nbSlices(&imageTest));
}

TEST(ImageStreamIOTestRead, CubeCPUSharedNbSlices) {
  IMAGE circularbufferTest;
  ImageStreamIO_read_sharedmem_image_toIMAGE("CubeTest", &circularbufferTest);
  EXPECT_EQ(13, ImageStreamIO_nbSlices(&circularbufferTest));
  EXPECT_EQ(circularbufferTest.array.raw, (void*)circularbufferTest.array.UI8);
}

} // namespace
