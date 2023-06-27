#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
# ifdef USE_CFITSIO
#include <fitsio.h>
#endif//USE_CFITSIO
#include "ImageStreamIO.h"
#include "ImageStreamIO_subTest_Operations.hpp"
#include "ImageStreamIO_cleanupTest.hpp"

// Duplicate ImageStreamIO search for directory to contain shmim file
char* gtest_shmdirname()
{
  const char shmdir_envvar_name[] = { "MILK_SHM_DIR" };
  static char shmdir_macro[] = { SHAREDMEMDIR };
  static char slash_tmp[] = { "/tmp" };
  char* pshmdirname = getenv("MILK_SHM_DIR");

  // If envvar check returned a null pointer, advance to macro
  if (!pshmdirname) { pshmdirname = shmdir_macro; }

  while (pshmdirname)
  {
    struct stat statbuf;
    if (!lstat(pshmdirname,&statbuf))
    {
      if ((statbuf.st_mode & S_IFMT) == S_IFDIR) { return pshmdirname; }
    }
    else
    {
      errno = 0;
    }

    // Advance shmdir name pointer to the next string to test
    // - tmp => NULL (to terminate the search)
    // - macro => /tmp
    // - (else) envvar => macro
    if (pshmdirname==slash_tmp)          { pshmdirname = (char*) NULL; }
    else if (pshmdirname==shmdir_macro ) { pshmdirname = slash_tmp; }
    else                                 { pshmdirname = shmdir_macro; }
  }
  return pshmdirname;
}

#include "catch.hpp"

// A prefix to all names indicating ImageStreamIO Unit Tests
#define SHM_NAME_PREFIX    "__ISIOUTs__"
#define SHM_NAME_ImageTest SHM_NAME_PREFIX "ImageTest"
#define SHM_NAME_CubeTest  SHM_NAME_PREFIX "CubeTest"
#define SHM_NAME_LocnTest  SHM_NAME_PREFIX "LocationTest"

TEST_CASE("ImageStreamIO") {

  uint32_t dims2[2] = {32, 32};
  uint32_t dims3[3] = {16, 16, 13};
  IMAGE imageTest;
  IMAGE circularbufferTest;

  int8_t cpuLocn = -1;   // Location of -1 => CPU-based shmim
  int8_t gpuLocn =  0;   // Location of  0 => Pretend GPU-based shmim
  int8_t badLocn = -2;   // Location of -2 => bad location

////////////////////////////////////////////////////////////////////////
// ImageStreamIO utilities
// - Finding the address of the start of data of interest
//    - SlicesAndIndices
//    - NonCircularReadBufferAddresses
//    - CircularReadBufferAddresses
//    - NonCircularWroteBufferAddresses
//    - CircularWroteBufferAddresses
//    - NonCircularWriteBufferAddresses
//    - CircularWriteBufferAddresses
// - Building the shmim filename
//    - FilenameFailure
//    - FilenameSuccess
// - Data type information (convert to size, name, CFITSIO type, etc.)
//    - Typesize
//    - Typename
//    - Typename_7
//    - TypenameShort
//    - Checktype
//    - Floattype
//    - FITSIOdatatype
//    - FITSIObitpix
////////////////////////////////////////////////////////////////////////
SECTION("Utilities.SlicesAndIndices") {

  // Calculations related to the number of slices
  // - Use local memory for IMAGE and IMAGE_METADATA structures
  IMAGE image { 0 };
  IMAGE_METADATA md { 0 };

  // - Make IMAGE metadata pointer point to METADATA structure
  image.md = &md;

  // - Put values in width and height sizes; 30 as slice count
  md.size[0] = 10;
  md.size[1] = 20;
  md.size[2] = 30;

  // - Assume last-written slice was slice 5
  md.cnt1 = 5;

  // - 1 axis:  md.size[2] and .cnt1 are ignored; number of slices is 1
  md.imagetype &= ~CIRCULAR_BUFFER;
  md.naxis = 1;
  CHECK(1 == ImageStreamIO_nbSlices(&image));
  CHECK(0 == ImageStreamIO_readLastWroteIndex(&image));
  CHECK(0 == ImageStreamIO_writeIndex(&image));

  // - 2 axes:  md.size[2] and .cnt1 are ignored; number of slices is 1
  md.naxis = 2;
  CHECK(1 == ImageStreamIO_nbSlices(&image));
  CHECK(0 == ImageStreamIO_readLastWroteIndex(&image));
  CHECK(0 == ImageStreamIO_writeIndex(&image));

  // - 3 axes:  md.size[2] and .cnt1(5) are used in slice calculations
  md.imagetype |= CIRCULAR_BUFFER;
  md.naxis = 3;
  CHECK(30 == ImageStreamIO_nbSlices(&image));
  CHECK(5 == ImageStreamIO_readLastWroteIndex(&image));
  CHECK(6 == ImageStreamIO_writeIndex(&image));

  // - 3 axes with md.size[2]=300, and .cnt1 == 299:  299 is last slice;
  //   299+1 = 300 is the next slice, but it rolls over to 0
  md.cnt1 = 29;
  CHECK(30 == ImageStreamIO_nbSlices(&image));
  CHECK(29 == ImageStreamIO_readLastWroteIndex(&image));
  CHECK(0 == ImageStreamIO_writeIndex(&image));
}

SECTION("Utilities.NonCircularReadBufferAddresses") {

  // Calculations related to the number of slices
  // - Use local memory for IMAGE and IMAGE_METADATA structures
  IMAGE image { 0 };
  IMAGE_METADATA md { 0 };
  uint8_t* pui8 { 0 };
  union { void* raw; uint8_t* UI8; } p;

  // - Make IMAGE metadata pointer point to METADATA structure
  image.md = &md;

  // - Put values in width and height sizes; 30 as slice count
  // - Choose 16-byte data elements
  md.size[0] = 10;
  md.size[1] = 20;
  md.size[2] = 30;
  md.datatype = _DATATYPE_COMPLEX_DOUBLE;

  // - Place data buffer at end of md, configure buffer as non-circular
  image.array.UI8 = pui8 = ((uint8_t*)image.md) + (sizeof md);
  md.naxis = 1;
  md.imagetype &= ~CIRCULAR_BUFFER;

  // - Result from ImageStreamIO_readBufferAt will be constant
  p.raw = 0;
  CHECK(p.UI8 ==(uint8_t*)NULL);
  CHECK(IMAGESTREAMIO_SUCCESS
           == ImageStreamIO_readBufferAt(&image,0,&p.raw));
  CHECK(image.array.UI8 ==p.UI8);

  p.raw = 0;
  CHECK(IMAGESTREAMIO_SUCCESS
           == ImageStreamIO_readBufferAt(&image,29,&p.raw));
  CHECK(image.array.UI8 ==p.UI8);

  p.raw = 0;
  CHECK(IMAGESTREAMIO_SUCCESS
           == ImageStreamIO_readBufferAt(&image,30,&p.raw));
  CHECK(image.array.UI8 ==p.UI8);
}
SECTION("Utilities.CircularReadBufferAddresses") {

  // Calculations related to the number of slices
  // - Use local memory for IMAGE and IMAGE_METADATA structures
  IMAGE image { 0 };
  IMAGE_METADATA md { 0 };
  uint8_t* pui8 { 0 };
  union { void* raw; uint8_t* UI8; } p;
  uint64_t slice_size {0 };

  // - Make IMAGE metadata pointer point to METADATA structure
  image.md = &md;

  // - Put values in width and height sizes; 30 as slice count
  // - Choose 16-byte data elements, calculate slice size
  md.size[0] = 10;
  md.size[1] = 20;
  md.size[2] = 30;
  md.datatype = _DATATYPE_COMPLEX_DOUBLE;
  slice_size = md.size[0];
  slice_size *= md.size[1];
  slice_size *= 16;

  // - Place data buffer at end of md, configure buffer as circular
  image.array.UI8 = pui8 = ((uint8_t*)image.md) + (sizeof md);
  md.naxis = 3;
  md.imagetype |= CIRCULAR_BUFFER;

  // - Test at start of circular buffer
  p.raw = 0;
  CHECK(IMAGESTREAMIO_SUCCESS
           == ImageStreamIO_readBufferAt(&image,0,&p.raw));
  CHECK(image.array.UI8 ==p.UI8);

  // - Test at end of circular buffer
  p.raw = 0;
  CHECK(IMAGESTREAMIO_SUCCESS
           == ImageStreamIO_readBufferAt(&image,29,&p.raw));
  CHECK(image.array.UI8+(29*slice_size) ==p.UI8);

  // - Test past end of circular buffer (failure)
  CHECK(IMAGESTREAMIO_FAILURE
           == ImageStreamIO_readBufferAt(&image,30,&p.raw));
  CHECK(p.UI8 ==(uint8_t*)NULL);
}

SECTION("Utilities.NonCircularWroteBufferAddresses") {

  // Calculations related to the number of slices
  // - Use local memory for IMAGE and IMAGE_METADATA structures
  IMAGE image { 0 };
  IMAGE_METADATA md { 0 };
  uint8_t* pui8 { 0 };
  union { void* raw; uint8_t* UI8; } p;

  // - Make IMAGE metadata pointer point to METADATA structure
  image.md = &md;

  // - Put values in width and height sizes; 30 as slice count
  // - Choose 16-byte data elements
  md.size[0] = 10;
  md.size[1] = 20;
  md.size[2] = 30;
  md.datatype = _DATATYPE_COMPLEX_DOUBLE;

  // - Place data buffer at end of md, configure buffer as non-circular
  image.array.UI8 = pui8 = ((uint8_t*)image.md) + (sizeof md);
  md.naxis = 1;
  md.imagetype &= ~CIRCULAR_BUFFER;

  // - Result from ImageStreamIO_readLastWroteBuffer will be constant
  p.raw = 0;
  md.cnt1 = 0;
  CHECK(IMAGESTREAMIO_SUCCESS
           == ImageStreamIO_readLastWroteBuffer(&image,&p.raw));
  CHECK(image.array.UI8 ==p.UI8);

  p.raw = 0;
  md.cnt1 = 29;
  CHECK(IMAGESTREAMIO_SUCCESS
           == ImageStreamIO_readLastWroteBuffer(&image,&p.raw));
  CHECK(image.array.UI8 ==p.UI8);

  p.raw = 0;
  md.cnt1 = 30;
  CHECK(IMAGESTREAMIO_SUCCESS
           == ImageStreamIO_readLastWroteBuffer(&image,&p.raw));
  CHECK(image.array.UI8 ==p.UI8);
}

SECTION("Utilities.CircularWroteBufferAddresses") {

  // Calculations related to the number of slices
  // - Use local memory for IMAGE and IMAGE_METADATA structures
  IMAGE image { 0 };
  IMAGE_METADATA md { 0 };
  uint8_t* pui8 { 0 };
  union { void* raw; uint8_t* UI8; } p;
  uint64_t slice_size {0 };

  // - Make IMAGE metadata pointer point to METADATA structure
  image.md = &md;

  // - Put values in width and height sizes; 30 as slice count
  // - Choose 16-byte data elements, calculate slice size
  md.size[0] = 10;
  md.size[1] = 20;
  md.size[2] = 30;
  md.datatype = _DATATYPE_COMPLEX_DOUBLE;
  slice_size = md.size[0];
  slice_size *= md.size[1];
  slice_size *= 16;

  // - Place data buffer at end of md, configure buffer as circular
  image.array.UI8 = pui8 = ((uint8_t*)image.md) + (sizeof md);
  md.naxis = 3;
  md.imagetype |= CIRCULAR_BUFFER;

  // - Test at start of circular buffer
  p.raw = 0;
  md.cnt1 = 0;
  CHECK(IMAGESTREAMIO_SUCCESS
           == ImageStreamIO_readLastWroteBuffer(&image,&p.raw));
  CHECK(image.array.UI8 ==p.UI8);

  // - Test at end of circular buffer
  p.raw = 0;
  md.cnt1 = 29;
  CHECK(IMAGESTREAMIO_SUCCESS
           == ImageStreamIO_readLastWroteBuffer(&image,&p.raw));
  CHECK(image.array.UI8+(29*slice_size) ==p.UI8);

  // - Test past end of circular buffer (failure)
  p.raw = 0;
  md.cnt1 = 30;
  CHECK(IMAGESTREAMIO_FAILURE
           == ImageStreamIO_readLastWroteBuffer(&image,&p.raw));
  CHECK(p.UI8 ==(uint8_t*)NULL);
}

SECTION("Utilities.NonCircularWriteBufferAddresses") {

  // Calculations related to the number of slices
  // - Use local memory for IMAGE and IMAGE_METADATA structures
  IMAGE image { 0 };
  IMAGE_METADATA md { 0 };
  uint8_t* pui8 { 0 };
  union { void* raw; uint8_t* UI8; } p;

  // - Make IMAGE metadata pointer point to METADATA structure
  image.md = &md;

  // - Put values in width and height sizes; 30 as slice count
  // - Choose 16-byte data elements
  md.size[0] = 10;
  md.size[1] = 20;
  md.size[2] = 30;
  md.datatype = _DATATYPE_COMPLEX_DOUBLE;

  // - Place data buffer at end of md, configure buffer as non-circular
  image.array.UI8 = pui8 = ((uint8_t*)image.md) + (sizeof md);
  md.naxis = 1;
  md.imagetype &= ~CIRCULAR_BUFFER;

  // - Result from ImageStreamIO_writeBuffer will be constant
  p.raw = 0;
  md.cnt1 = 0;
  CHECK(IMAGESTREAMIO_SUCCESS
           == ImageStreamIO_writeBuffer(&image,&p.raw));
  CHECK(image.array.UI8 ==p.UI8);

  p.raw = 0;
  md.cnt1 = 29;
  CHECK(IMAGESTREAMIO_SUCCESS
           == ImageStreamIO_writeBuffer(&image,&p.raw));
  CHECK(image.array.UI8 ==p.UI8);

  p.raw = 0;
  md.cnt1 = 30;
  CHECK(IMAGESTREAMIO_SUCCESS
           == ImageStreamIO_writeBuffer(&image,&p.raw));
  CHECK(image.array.UI8 ==p.UI8);
}

SECTION("Utilities.CircularWriteBufferAddresses") {

  // Calculations related to the number of slices
  // - Use local memory for IMAGE and IMAGE_METADATA structures
  IMAGE image { 0 };
  IMAGE_METADATA md { 0 };
  uint8_t* pui8 { 0 };
  union { void* raw; uint8_t* UI8; } p;
  uint64_t slice_size {0 };

  // - Make IMAGE metadata pointer point to METADATA structure
  image.md = &md;

  // - Put values in width and height sizes; 30 as slice count
  // - Choose 16-byte data elements, calculate slice size
  md.size[0] = 10;
  md.size[1] = 20;
  md.size[2] = 30;
  md.datatype = _DATATYPE_COMPLEX_DOUBLE;
  slice_size = md.size[0];
  slice_size *= md.size[1];
  slice_size *= 16;

  // - Place data buffer at end of md, configure buffer as circular
  image.array.UI8 = pui8 = ((uint8_t*)image.md) + (sizeof md);
  md.naxis = 3;
  md.imagetype |= CIRCULAR_BUFFER;

  // - Test at start of circular buffer
  p.raw = 0;
  md.cnt1 = 0;
  CHECK(IMAGESTREAMIO_SUCCESS
           == ImageStreamIO_writeBuffer(&image,&p.raw));
  CHECK(image.array.UI8+slice_size ==p.UI8);

  // - Test at end of circular buffer
  p.raw = 0;
  md.cnt1 = 29;
  CHECK(IMAGESTREAMIO_SUCCESS
           == ImageStreamIO_writeBuffer(&image,&p.raw));
  CHECK(image.array.UI8 ==p.UI8);

  // - Test past end of circular buffer; modulo prevents failure
  p.raw = 0;
  md.cnt1 = 30;
  CHECK(IMAGESTREAMIO_SUCCESS
           == ImageStreamIO_writeBuffer(&image,&p.raw));
  CHECK(image.array.UI8+slice_size ==p.UI8);
}

SECTION("Utilities.FilenameFailure") {
  char file_name[256];
  char* gtest_shmdirname(void);
  char* pshmdirname = gtest_shmdirname();
  const char gtest_name[] { "g" };

  // Get minimum length of shmim file path (/dir/name.im.shm)
  size_t toosmall{ (pshmdirname ? strlen(pshmdirname) : 0)
                 + strlen("/")
                 + strlen(gtest_name)
                 + strlen(".im.shm")
                 };

  REQUIRE(pshmdirname);
  //if(!pshmdirname)
  //{
  //  GTEST_SKIP_("Skipped filename tests; no directory is available");
  //}

  CHECK(toosmall < sizeof file_name);

  // One character too small (inadeqquat space for terminating null)
  CHECK(IMAGESTREAMIO_FAILURE
           == ImageStreamIO_filename(file_name,toosmall,gtest_name));
}

SECTION("Utilities.FilenameSuccess") {
  char file_name[256];
  char* pfile_name{0};
  char* gtest_shmdirname(void);
  char* pshmdirname = gtest_shmdirname();
  const char gtest_name[] { "g" };

  // Get minimum length of shmim file path (/dir/name.im.shm)
  size_t toosmall{ (pshmdirname ? strlen(pshmdirname) : 0)
                 + strlen("/")
                 + strlen(gtest_name)
                 + strlen(".im.shm")
                 };

  REQUIRE(pshmdirname);
  //if(!pshmdirname)
  //{
  //  GTEST_SKIP_("Skipped filename tests; no directory is available");
  //}

  CHECK(toosmall < sizeof file_name);

  // Barely enough
  CHECK(IMAGESTREAMIO_SUCCESS
           == ImageStreamIO_filename(file_name,toosmall+1,gtest_name));

  CHECK(strlen(file_name) ==toosmall);

  pfile_name = file_name;

  CHECK(0 ==strncmp(pfile_name,pshmdirname,strlen(pshmdirname)));
  pfile_name += strlen(pshmdirname);

  CHECK('/' ==*pfile_name);
  ++pfile_name;

  CHECK(0 ==strncmp(pfile_name,gtest_name,strlen(gtest_name)));
  pfile_name += strlen(gtest_name);

  CHECK(0 ==strcmp(pfile_name,".im.shm"));
}

SECTION("Utilities.Typesize") {
# ifdef UTSEE
# undef UTSEE
# endif
# define UTSEE(A,B,N) \
         CHECK(A == ImageStreamIO_typesize(B)); \
         CHECK(N == ImageStreamIO_typesize(B))
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

SECTION("Utilities.Typename") {
# ifdef UTNEE
# undef UTNEE
# endif
# define UTNEE(A,B) \
         CHECK(std::string(A) == ImageStreamIO_typename(B))
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

SECTION("Utilities.Typename_7") {
# ifdef UT7EE
# undef UT7EE
# endif
# define UT7EE(A,B) \
         CHECK(std::string(A) == ImageStreamIO_typename_7(B))
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

SECTION("Utilities.TypenameShort") {
# ifdef UTSEE
# undef UTSEE
# endif
# define UTSEE(A,B) \
         CHECK(std::string(A) == ImageStreamIO_typename_short(B))
  UTSEE(" UI8",  _DATATYPE_UINT8);
  UTSEE("  I8",  _DATATYPE_INT8);
  UTSEE("UI16",  _DATATYPE_UINT16);
  UTSEE(" I16",  _DATATYPE_INT16);
  UTSEE("UI32",  _DATATYPE_UINT32);
  UTSEE(" I32",  _DATATYPE_INT32);
  UTSEE("UI64",  _DATATYPE_UINT64);
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

SECTION("Utilities.Checktype") {
# ifdef UCTEE
# undef UCTEE
# endif
# define UCTEE(A,B,C) \
         CHECK(A == ImageStreamIO_checktype(B,0)); \
         CHECK(C == ImageStreamIO_checktype(B,1))
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

SECTION("Utilities.Floattype") {
# ifdef UFTEE
# undef UFTEE
# endif
# define UFTEE(A,B) \
         CHECK(A == ImageStreamIO_floattype(B))
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

SECTION("Utilities.FITSIOdatatype") {
# ifdef UFDEE
# undef UFDEE
# endif
# define UFDEE(A,B) \
         CHECK(A == ImageStreamIO_FITSIOdatatype(B))
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

SECTION("Utilities.FITSIObitpix") {
# ifdef UFBEE
# undef UFBEE
# endif
# define UFBEE(A,B) \
         CHECK(A == ImageStreamIO_FITSIObitpix(B))
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
SECTION("TestCreation.ImageCPUCreation") {

  char SM_fname[200];
  ImageStreamIO_filename(SM_fname, 200, SHM_NAME_ImageTest);

  fprintf(stderr,"[%s=>%s]=SM_fname\n",SHM_NAME_ImageTest, SM_fname);

  CHECK(IMAGESTREAMIO_SUCCESS
           == ImageStreamIO_createIm_gpu(&imageTest, SHM_NAME_ImageTest
                                      ,2, dims2, _DATATYPE_FLOAT
                                      ,cpuLocn, 0, 10, 10, MATH_DATA,0)
           );
}

SECTION("TestCreation.ImageCPUSharedCreation") {

  CHECK(IMAGESTREAMIO_SUCCESS
           == ImageStreamIO_createIm_gpu(&imageTest, SHM_NAME_ImageTest
                                      ,2, dims2, _DATATYPE_FLOAT
                                      ,cpuLocn, 1, 10, 10, MATH_DATA,0)
           );
}

SECTION("TestCreation.CubeCPUSharedCreationDimensionFailure") {

  CHECK(IMAGESTREAMIO_INVALIDARG
           == ImageStreamIO_createIm_gpu(&circularbufferTest, SHM_NAME_CubeTest
                                      ,2, dims2, _DATATYPE_FLOAT
                                      ,cpuLocn, 1, 10, 10, CIRCULAR_BUFFER,1)
           );
}

SECTION("TestCreation.CubeCPUSharedCreation") {

  CHECK(IMAGESTREAMIO_SUCCESS
           == ImageStreamIO_createIm_gpu(&circularbufferTest, SHM_NAME_CubeTest
                                      ,3, dims3,_DATATYPE_FLOAT
                                      ,cpuLocn, 1, 10, 10, CIRCULAR_BUFFER,1)
           );
}

////////////////////////////////////////////////////////////////////////
// ImageStreamIO_OpenIm - opening an existing shmim file 
// - Use imageTest from above; assume it has not been destroyed
////////////////////////////////////////////////////////////////////////
SECTION("TestOpen.ImageCPUSharedOpen") {

  CHECK(IMAGESTREAMIO_SUCCESS
           == ImageStreamIO_openIm(&imageTest, SHM_NAME_ImageTest)
           );
}

SECTION("TestOpen.ImageCPUSharedOpenNotExistFailure") {

  CHECK(IMAGESTREAMIO_FILEOPEN
           == ImageStreamIO_openIm(&imageTest
                                ,SHM_NAME_ImageTest "DoesNotExist")
           );
}

SECTION("TestOpen.CubeCPUSharedOpen") {

  CHECK(IMAGESTREAMIO_SUCCESS
           == ImageStreamIO_openIm(&circularbufferTest, SHM_NAME_CubeTest)
           );
}

SECTION("TestRead.ImageCPUSharedNbSlices") {

  CHECK(IMAGESTREAMIO_SUCCESS
           == ImageStreamIO_read_sharedmem_image_toIMAGE(SHM_NAME_ImageTest
                                                      ,&imageTest)
           );
  CHECK(1 == ImageStreamIO_nbSlices(&imageTest));
}

SECTION("TestRead.CubeCPUSharedNbSlices") {

  CHECK(IMAGESTREAMIO_SUCCESS
           == ImageStreamIO_read_sharedmem_image_toIMAGE(SHM_NAME_CubeTest
                                                      ,&circularbufferTest)
           );
  CHECK(13 == ImageStreamIO_nbSlices(&circularbufferTest));
}

////////////////////////////////////////////////////////////////////////
// Location-related tests:  location is in CPU or GPU; not in filesystem
////////////////////////////////////////////////////////////////////////
SECTION("TestLocation.BadLocationFailure") {

  CHECK(IMAGESTREAMIO_INVALIDARG
           == ImageStreamIO_createIm_gpu(&imageTest, SHM_NAME_LocnTest
                                      ,2, dims2,_DATATYPE_FLOAT
                                      ,badLocn, 1, 10, 10, MATH_DATA,0)
           );
}

#ifdef HAVE_CUDA
SECTION("TestCreation.ImageGPUSharedCreation") {
  CHECK(IMAGESTREAMIO_SUCCESS
           == ImageStreamIO_createIm_gpu(&imageTest, SHM_NAME_LocnTest
                                      ,2, dims2, _DATATYPE_FLOAT
                                      ,gpuLocn, 1, 10, 10, MATH_DATA,0)
           );
}
#else
SECTION("###SKIPPED:  TestCreation.ImageGPUSharedCreation") {
  REQUIRE( true == true);
  //GTEST_SKIP_("Skipped GPU Shared Creation; HAVE_CUDA is undefined");
}
#endif

// For GPU-located shmim, creating an existing file is an error
SECTION("TestLocation.InitCpuLocationFailure") {

  // Ensure file exists by using CPU Location
  REQUIRE(IMAGESTREAMIO_SUCCESS
           == ImageStreamIO_createIm_gpu(&imageTest, SHM_NAME_LocnTest
                                      ,2, dims2, _DATATYPE_FLOAT
                                      ,cpuLocn, 1, 10, 10, MATH_DATA,0)
           );

  CHECK(IMAGESTREAMIO_FILEEXISTS
           == ImageStreamIO_createIm_gpu(&imageTest, SHM_NAME_LocnTest
                                      ,2, dims2, _DATATYPE_FLOAT
                                      ,gpuLocn, 1, 10, 10, MATH_DATA,0)
           );
}

// Operational test:  child process writes to shmim; parent reads
SECTION("TestOperations.OperationsTest") {

  int success_count;
  int test_count;
  ImageStreamIO_subTest_Operations(test_count, success_count);
  REQUIRE(success_count == test_count);
}

// Operational test:  child process writes to shmim; parent reads
SECTION("TestOperations.CleanupTest") {

  bool kill_child{false};
  std::string sOK{"OK"};

  ISIO_CLEANUP isio_cleanup{ISIO_CLEANUP()};

  CHECK(sOK == isio_cleanup.rm_shmim_filepath_01());
  CHECK(sOK == isio_cleanup.block_SIGUSR2_02(true));
  CHECK(sOK == isio_cleanup.fork_child_03());
  CHECK(sOK == isio_cleanup.wait_for_SIGUSR2_04());
  CHECK(sOK == isio_cleanup.open_shmim_05());
  CHECK(sOK == isio_cleanup.check_for_semfiles_06());
  CHECK(sOK == isio_cleanup.release_the_child_07());
  CHECK(sOK == isio_cleanup.wait_for_sem_08(kill_child));
  CHECK(sOK == isio_cleanup.close_shmim_09());
  CHECK(sOK == isio_cleanup.wait_for_child_10(kill_child));
  CHECK(sOK == isio_cleanup.file_cleanup_11(kill_child));

  isio_cleanup._destructor();
  isio_cleanup._constructor();

  kill_child = true;

  CHECK(sOK == isio_cleanup.rm_shmim_filepath_01());
  CHECK(sOK == isio_cleanup.block_SIGUSR2_02(true));
  CHECK(sOK == isio_cleanup.fork_child_03());
  CHECK(sOK == isio_cleanup.wait_for_SIGUSR2_04());
  CHECK(sOK == isio_cleanup.open_shmim_05());
  CHECK(sOK == isio_cleanup.check_for_semfiles_06());
  CHECK(sOK == isio_cleanup.release_the_child_07());
  CHECK(sOK == isio_cleanup.wait_for_sem_08(kill_child));
  CHECK(sOK == isio_cleanup.close_shmim_09());
  CHECK(sOK == isio_cleanup.wait_for_child_10(kill_child));
  CHECK(sOK == isio_cleanup.file_cleanup_11(kill_child));
}

// Ops test:  child makes non-shared, process-local shmim; parent waits.
SECTION("TestOperations.LocalShmimCleanupTest") {

  std::string sOK{"OK"};

  ISIO_CLEANUP isio_cleanup{ISIO_CLEANUP()};

  // Arg=1:  child creates non-shared, process-local calloc'ed shmim
  CHECK(sOK == isio_cleanup.fork_child_03(1));
  CHECK(sOK == isio_cleanup.wait_for_child_10(false));
}

} // SECTION("")
