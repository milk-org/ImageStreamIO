cmake_minimum_required(VERSION 3.8)

set(LIBNAME "ImageStreamIO-backport")
set(SRCNAME "ImageStreamIO")

option(python_build "Compile Python Wrappers" OFF)
option(gtest_build_tests "Build Unit Tests" OFF)

project(${LIBNAME} LANGUAGES C)

add_compile_options(-Ofast)

add_library(${LIBNAME} SHARED ${SRCNAME}.c)

find_package(PkgConfig REQUIRED)
pkg_check_modules(CFITSIO cfitsio)
if(${CFITSIO_FOUND})
  link_directories(${CFITSIO_LIBRARY_DIRS})
  target_compile_definitions(${LIBNAME} PRIVATE USE_CFITSIO=1)
  target_include_directories(${LIBNAME} PRIVATE ${CFITSIO_INCLUDE_DIRS})
endif()

target_include_directories(${LIBNAME}
                           PUBLIC ${PROJECT_SOURCE_DIR}
                                  ${PROJECT_SOURCE_DIR}/src
                                  ${PROJECT_SOURCE_DIR}/..)

if(USE_CUDA)
  find_package(CUDA REQUIRED)
  target_include_directories(${LIBNAME} PUBLIC ${CUDA_INCLUDE_DIRS})
  target_link_libraries(${LIBNAME} PUBLIC ${CUDA_LIBRARIES})
  target_compile_options(${LIBNAME} PUBLIC -DHAVE_CUDA)
endif(USE_CUDA)

#
# Python wrap.
#
# The python wrapper is not built by default.  To build it, set the python_build
# option to ON.  You can do it by running ccmake or specifying the
# -Dpython_build=ON flag when running cmake.

if(python_build)
  add_subdirectory(python_module)
endif()

########################################################################
#
# Google Test.
#
# The tests are not built by default.  To build them, set the
# gtest_build_tests option to ON.  You can do it by running ccmake
# or specifying the -Dgtest_build_tests=ON flag when running cmake.

if (gtest_build_tests)
  # This must be set in the root directory for the tests to be run by
  # 'make test' or ctest.
  enable_testing()
  add_subdirectory(tests)
endif()


install(TARGETS ${LIBNAME} DESTINATION lib)
install(FILES ${SRCNAME}.h DESTINATION include/${SRCNAME})
install(FILES ImageStruct.h ImageStreamIOError.h DESTINATION include/${SRCNAME})
