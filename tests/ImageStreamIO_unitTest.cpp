//#include <limits.h>
# ifdef USE_CFITSIO
#include <fitsio.h>
#endif//USE_CFITSIO
#include "ImageStreamIO.h"
#include "ImageStreamIO_subTest_Operations.hpp"
#include "gtest/gtest.h"

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
// ImageStreamIO utilities
////////////////////////////////////////////////////////////////////////
TEST(ImageStreamIOUtilities, SlicesAndIndices) {

  // Calculations related to the number of slices
  // - Use local memory for IMAGE and IMAGE_METADATA structures
  IMAGE image { 0 };
  IMAGE_METADATA md { 0 };

  // - Make IMAGE metadata pointer point to METADATA structure
  image.md = &md;

  // - Put dummy values in width and height sizes; 300 in slice size
  md.size[0] = -100;
  md.size[1] = -200;
  md.size[2] = 300;

  // - Assume last-written slice was slice 5
  md.cnt1 = 5;

  // - 1 axis:  md.size[2] and .cnt1 are ignored; number of slices is 1
  md.naxis = 1;
  EXPECT_EQ(1, ImageStreamIO_nbSlices(&image));
  EXPECT_EQ(0, ImageStreamIO_readLastWroteIndex(&image));
  EXPECT_EQ(0, ImageStreamIO_writeIndex(&image));

  // - 2 axes:  md.size[2] and .cnt1 are ignored; number of slices is 1
  md.naxis = 2;
  EXPECT_EQ(1, ImageStreamIO_nbSlices(&image));
  EXPECT_EQ(0, ImageStreamIO_readLastWroteIndex(&image));
  EXPECT_EQ(0, ImageStreamIO_writeIndex(&image));

  // - 3 axes:  md.size[2] and .cnt1(5) are used in slice calculations
  md.naxis = 3;
  EXPECT_EQ(300, ImageStreamIO_nbSlices(&image));
  EXPECT_EQ(5, ImageStreamIO_readLastWroteIndex(&image));
  EXPECT_EQ(6, ImageStreamIO_writeIndex(&image));

  // - 3 axes with md.size[2]=300, and .cnt1 == 299:  299 is last slice;
  //   299+1 = 300 is the next slice, but it rolls over to 0
  md.cnt1 = 299;
  EXPECT_EQ(300, ImageStreamIO_nbSlices(&image));
  EXPECT_EQ(299, ImageStreamIO_readLastWroteIndex(&image));
  EXPECT_EQ(0, ImageStreamIO_writeIndex(&image));
}

TEST(ImageStreamIOUtilities, Typesize) {
# ifdef UTSEE
# undef UTSEE
# endif
# define UTSEE(A,B,N) \
         EXPECT_EQ(A, ImageStreamIO_typesize(B)); \
         EXPECT_EQ(N, ImageStreamIO_typesize(B))
  UTSEE(SIZEOF_DATATYPE_UINT8,           _DATATYPE_UINT8,           1);
  UTSEE(SIZEOF_DATATYPE_INT8,            _DATATYPE_INT8,            1);
  UTSEE(SIZEOF_DATATYPE_UINT16,          _DATATYPE_UINT16,          2);
  UTSEE(SIZEOF_DATATYPE_INT16,           _DATATYPE_INT16,           2);
  UTSEE(SIZEOF_DATATYPE_UINT32,          _DATATYPE_UINT32,          4);
  UTSEE(SIZEOF_DATATYPE_INT32,           _DATATYPE_INT32,           4);
  UTSEE(SIZEOF_DATATYPE_UINT64,          _DATATYPE_UINT64,          8);
  UTSEE(SIZEOF_DATATYPE_INT64,           _DATATYPE_INT64,           8);
  UTSEE(SIZEOF_DATATYPE_HALF,            _DATATYPE_HALF,            2);
  UTSEE(SIZEOF_DATATYPE_FLOAT,           _DATATYPE_FLOAT,           4);
  UTSEE(SIZEOF_DATATYPE_DOUBLE,          _DATATYPE_DOUBLE,          8);
  UTSEE(SIZEOF_DATATYPE_COMPLEX_FLOAT,   _DATATYPE_COMPLEX_FLOAT,   8);
  UTSEE(SIZEOF_DATATYPE_COMPLEX_DOUBLE,  _DATATYPE_COMPLEX_DOUBLE, 16);
  UTSEE(-1,                              _DATATYPE_UNINITIALIZED,  -1);
  UTSEE(-1,                              255,                      -1);
# undef UTSEE
}

TEST(ImageStreamIOUtilities, Typename) {
# ifdef UTNEE
# undef UTNEE
# endif
# define UTNEE(A,B) \
         EXPECT_STREQ(A, ImageStreamIO_typename(B))
  UTNEE("UINT8",   _DATATYPE_UINT8);
  UTNEE("INT8",    _DATATYPE_INT8);
  UTNEE("UINT16",  _DATATYPE_UINT16);
  UTNEE("INT16",   _DATATYPE_INT16);
  UTNEE("UINT32",  _DATATYPE_UINT32);
  UTNEE("INT32",   _DATATYPE_INT32);
  UTNEE("UINT64",  _DATATYPE_UINT64);
  UTNEE("INT64",   _DATATYPE_INT64);
  UTNEE("FLT16",   _DATATYPE_HALF);
  UTNEE("FLT32",   _DATATYPE_FLOAT);
  UTNEE("FLT64",   _DATATYPE_DOUBLE);
  UTNEE("CPLX32",  _DATATYPE_COMPLEX_FLOAT);
  UTNEE("CPLX64",  _DATATYPE_COMPLEX_DOUBLE);
  UTNEE("unknown", _DATATYPE_UNINITIALIZED);
  UTNEE("unknown", 255);
# undef UTNEE
}

TEST(ImageStreamIOUtilities, Typename_7) {
# ifdef UT7EE
# undef UT7EE
# endif
# define UT7EE(A,B) \
         EXPECT_STREQ(A, ImageStreamIO_typename_7(B))
  UT7EE("UINT8  ",  _DATATYPE_UINT8);
  UT7EE("INT8   ",  _DATATYPE_INT8);
  UT7EE("UINT16 ",  _DATATYPE_UINT16);
  UT7EE("INT16  ",  _DATATYPE_INT16);
  UT7EE("UINT32 ",  _DATATYPE_UINT32);
  UT7EE("INT32  ",  _DATATYPE_INT32);
  UT7EE("UINT64 ",  _DATATYPE_UINT64);
  UT7EE("INT64  ",  _DATATYPE_INT64);
  UT7EE("FLT16  ",  _DATATYPE_HALF);
  UT7EE("FLOAT  ",  _DATATYPE_FLOAT);
  UT7EE("DOUBLE ",  _DATATYPE_DOUBLE);
  UT7EE("CFLOAT ",  _DATATYPE_COMPLEX_FLOAT);
  UT7EE("CDOUBLE",  _DATATYPE_COMPLEX_DOUBLE);
  UT7EE("unknown",  _DATATYPE_UNINITIALIZED);
  UT7EE("unknown",  255);
# undef UT7EE
}

TEST(ImageStreamIOUtilities, TypenameShort) {
# ifdef UTSEE
# undef UTSEE
# endif
# define UTSEE(A,B) \
         EXPECT_STREQ(A, ImageStreamIO_typename_short(B))
  UTSEE(" UI8",  _DATATYPE_UINT8);
  UTSEE("  I8",  _DATATYPE_INT8);
  UTSEE("UI16",  _DATATYPE_UINT16);
  UTSEE(" I16",  _DATATYPE_INT16);
  UTSEE("UI32",  _DATATYPE_UINT32);
  UTSEE(" I32",  _DATATYPE_INT32);
  //////////////////////////////////////////////////////////////////////
  /// Bug in existing code; remove these comments when it's resolved ///
  /// BTCarcich drbitbyu@github.com ca. 2023-04-18                   ///
  //////////////////////////////////////////////////////////////////////
  //Uncomment this line; remove next:UTSEE("UI64",  _DATATYPE_UINT64);
  UTSEE(" UI64",  _DATATYPE_UINT64);  /// REMOVE THIS LINE AFTER FIX ///
  //////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////
  UTSEE(" I64",  _DATATYPE_INT64);
  UTSEE(" F16",  _DATATYPE_HALF);
  UTSEE(" FLT",  _DATATYPE_FLOAT);
  UTSEE(" DBL",  _DATATYPE_DOUBLE);
  UTSEE("CFLT",  _DATATYPE_COMPLEX_FLOAT);
  UTSEE("CDBL",  _DATATYPE_COMPLEX_DOUBLE);
  UTSEE(" ???",  _DATATYPE_UNINITIALIZED);
  UTSEE(" ???",  255);
# undef UTSEE
}

TEST(ImageStreamIOUtilities, Checktype) {
# ifdef UCTEE
# undef UCTEE
# endif
# define UCTEE(A,B,C) \
         EXPECT_EQ(A, ImageStreamIO_checktype(B,0)); \
         EXPECT_EQ(C, ImageStreamIO_checktype(B,1))
  UCTEE( 0,  _DATATYPE_UINT8,           0);
  UCTEE( 0,  _DATATYPE_INT8,            0);
  UCTEE( 0,  _DATATYPE_UINT16,          0);
  UCTEE( 0,  _DATATYPE_INT16,           0);
  UCTEE( 0,  _DATATYPE_UINT32,          0);
  UCTEE( 0,  _DATATYPE_INT32,           0);
  UCTEE( 0,  _DATATYPE_UINT64,          0);
  UCTEE( 0,  _DATATYPE_INT64,           0);
  UCTEE( 0,  _DATATYPE_HALF,            0);
  UCTEE( 0,  _DATATYPE_FLOAT,           0);
  UCTEE( 0,  _DATATYPE_DOUBLE,          0);
  UCTEE(-1,  _DATATYPE_COMPLEX_FLOAT,   0);
  UCTEE(-1,  _DATATYPE_COMPLEX_DOUBLE,  0);
  UCTEE(-1,  _DATATYPE_UNINITIALIZED,  -1);
  UCTEE(-1,  255,                      -1);
# undef UCTEE
}

TEST(ImageStreamIOUtilities, Floattype) {
# ifdef UFTEE
# undef UFTEE
# endif
# define UFTEE(A,B) \
         EXPECT_EQ(A, ImageStreamIO_floattype(B))
  UFTEE(_DATATYPE_FLOAT,          _DATATYPE_UINT8);
  UFTEE(_DATATYPE_FLOAT,          _DATATYPE_INT8);
  UFTEE(_DATATYPE_FLOAT,          _DATATYPE_UINT16);
  UFTEE(_DATATYPE_FLOAT,          _DATATYPE_INT16);
  UFTEE(_DATATYPE_FLOAT,          _DATATYPE_UINT32);
  UFTEE(_DATATYPE_FLOAT,          _DATATYPE_INT32);
  UFTEE(_DATATYPE_DOUBLE,         _DATATYPE_UINT64);
  UFTEE(_DATATYPE_DOUBLE,         _DATATYPE_INT64);
  UFTEE(_DATATYPE_HALF,           _DATATYPE_HALF);
  UFTEE(_DATATYPE_FLOAT,          _DATATYPE_FLOAT);
  UFTEE(_DATATYPE_DOUBLE,         _DATATYPE_DOUBLE);
  UFTEE(_DATATYPE_COMPLEX_FLOAT,  _DATATYPE_COMPLEX_FLOAT);
  UFTEE(_DATATYPE_COMPLEX_DOUBLE, _DATATYPE_COMPLEX_DOUBLE);
  UFTEE(-1,                       _DATATYPE_UNINITIALIZED);
  UFTEE(-1,                       255);
# undef UFTEE
}

TEST(ImageStreamIOUtilities, FITSIOdatatype) {
# ifdef UFDEE
# undef UFDEE
# endif
# define UFDEE(A,B) \
         EXPECT_EQ(A, ImageStreamIO_FITSIOdatatype(B))
# ifdef USE_CFITSIO
  UFDEE(TBYTE,   _DATATYPE_UINT8);
  UFDEE(TSBYTE,  _DATATYPE_INT8);
  UFDEE(TUSHORT, _DATATYPE_UINT16);
  UFDEE(TSHORT,  _DATATYPE_INT16);
  UFDEE(TUINT,   _DATATYPE_UINT32);
  UFDEE(TINT,    _DATATYPE_INT32);
  UFDEE(TULONG,  _DATATYPE_UINT64);
  UFDEE(TLONG,   _DATATYPE_INT64);
  UFDEE(TFLOAT,  _DATATYPE_FLOAT);
  UFDEE(TDOUBLE, _DATATYPE_DOUBLE);
# else//USE_CFITSIO
  UFDEE(-1,      _DATATYPE_UINT8);
  UFDEE(-1,      _DATATYPE_INT8);
  UFDEE(-1,      _DATATYPE_UINT16);
  UFDEE(-1,      _DATATYPE_INT16);
  UFDEE(-1,      _DATATYPE_UINT32);
  UFDEE(-1,      _DATATYPE_INT32);
  UFDEE(-1,      _DATATYPE_UINT64);
  UFDEE(-1,      _DATATYPE_INT64);
  UFDEE(-1,      _DATATYPE_FLOAT);
  UFDEE(-1,      _DATATYPE_DOUBLE);
# endif//USE_CFITSIO
  UFDEE(-1,      _DATATYPE_HALF);
  UFDEE(-1,      _DATATYPE_COMPLEX_FLOAT);
  UFDEE(-1,      _DATATYPE_COMPLEX_DOUBLE);
  UFDEE(-1,      _DATATYPE_UNINITIALIZED);
  UFDEE(-1,      255);
# undef UFDEE
}

TEST(ImageStreamIOUtilities, FITSIObitpix) {
# ifdef UFBEE
# undef UFBEE
# endif
# define UFBEE(A,B) \
         EXPECT_EQ(A, ImageStreamIO_FITSIObitpix(B))
# ifdef USE_CFITSIO
  UFBEE(BYTE_IMG,      _DATATYPE_UINT8);
  UFBEE(SBYTE_IMG,     _DATATYPE_INT8);
  UFBEE(USHORT_IMG,    _DATATYPE_UINT16);
  UFBEE(SHORT_IMG,     _DATATYPE_INT16);
  UFBEE(ULONG_IMG,     _DATATYPE_UINT32);
  UFBEE(LONG_IMG,      _DATATYPE_INT32);
  UFBEE(ULONGLONG_IMG, _DATATYPE_UINT64);
  UFBEE(LONGLONG_IMG,  _DATATYPE_INT64);
  UFBEE(FLOAT_IMG,     _DATATYPE_FLOAT);
  UFBEE(DOUBLE_IMG,    _DATATYPE_DOUBLE);
# else//USE_CFITSIO
  UFBEE(-1,            _DATATYPE_UINT8);
  UFBEE(-1,            _DATATYPE_INT8);
  UFBEE(-1,            _DATATYPE_UINT16);
  UFBEE(-1,            _DATATYPE_INT16);
  UFBEE(-1,            _DATATYPE_UINT32);
  UFBEE(-1,            _DATATYPE_INT32);
  UFBEE(-1,            _DATATYPE_UINT64);
  UFBEE(-1,            _DATATYPE_INT64);
  UFBEE(-1,            _DATATYPE_FLOAT);
  UFBEE(-1,            _DATATYPE_DOUBLE);
# endif//USE_CFITSIO
  UFBEE(-1,            _DATATYPE_HALF);
  UFBEE(-1,            _DATATYPE_COMPLEX_FLOAT);
  UFBEE(-1,            _DATATYPE_COMPLEX_DOUBLE);
  UFBEE(-1,            _DATATYPE_UNINITIALIZED);
  UFBEE(-1,            255);
# undef UFBEE
}

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

TEST(ImageStreamIOTestCreation, CubeCPUSharedCreationDimensionFailure) {

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
// - Use imageTest from above; assume it has not been destroyed
////////////////////////////////////////////////////////////////////////
TEST(ImageStreamIOTestOpen, ImageCPUSharedOpen) {

  EXPECT_EQ(IMAGESTREAMIO_SUCCESS
           ,ImageStreamIO_openIm(&imageTest, SHM_NAME_ImageTest)
           );
}

TEST(ImageStreamIOTestOpen, ImageCPUSharedOpenNotExistFailure) {

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
// Location-related tests:  location is in CPU or GPU; not in filesystem
////////////////////////////////////////////////////////////////////////
TEST(ImageStreamIOTestLocation, BadLocationFailure) {

  EXPECT_EQ(IMAGESTREAMIO_FAILURE
           ,ImageStreamIO_createIm_gpu(&imageTest, SHM_NAME_LocnTest
                                      ,2, dims2,_DATATYPE_FLOAT
                                      ,badLocn, 1, 10, 10, MATH_DATA,0)
           );
}

TEST(ImageStreamIOTestCreation, ImageGPUSharedCreation) {

# ifndef HAVE_CUDA
  GTEST_SKIP_("Skipped GPU Shared Creation; HAVE_CUDA is undefined");
# endif

  EXPECT_EQ(IMAGESTREAMIO_SUCCESS
           ,ImageStreamIO_createIm_gpu(&imageTest, SHM_NAME_LocnTest
                                      ,2, dims2, _DATATYPE_FLOAT
                                      ,gpuLocn, 1, 10, 10, MATH_DATA,0)
           );
}

// For GPU-located shmim, creating an existing file is an error
TEST(ImageStreamIOTestLocation, InitCpuLocationFailure) {

  // Ensure file exists by using CPU Location
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

// Operational test:  child process writes to shmim; parent reads
TEST(ImageStreamIOTestOperations, OperationsTest) {

  int success_count;
  int test_count;
  ImageStreamIO_subTest_Operations(test_count, success_count);
  ASSERT_EQ(success_count, test_count);
}

} // namespace
