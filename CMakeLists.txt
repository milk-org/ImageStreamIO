cmake_minimum_required(VERSION 3.8)

option(python_build "Compile Python Wrappers" ON)

if(python_build)
  include("cmake/HunterGate.cmake")
  huntergate(URL
             "https://github.com/ruslo/hunter/archive/v0.23.126.tar.gz"
             SHA1
             "ccaf37e820cb54d6a5c02993c7f389b9861fc38e")
endif()

project(ImageStreamIO LANGUAGES C)

add_compile_options(-Ofast)

add_library(ImageStreamIO SHARED ImageStreamIO.c)

find_package(PkgConfig REQUIRED)
pkg_check_modules(CFITSIO REQUIRED cfitsio)
link_directories(${CFITSIO_LIBRARY_DIRS})

target_include_directories(ImageStreamIO
                           PUBLIC ${PROJECT_SOURCE_DIR}
                                  ${PROJECT_SOURCE_DIR}/src
                                  ${PROJECT_SOURCE_DIR}/..
                                  ${CFITSIO_INCLUDE_DIRS})

if(USE_CUDA)
  find_package(CUDA REQUIRED)
  target_include_directories(ImageStreamIO PUBLIC ${CUDA_INCLUDE_DIRS})
  target_link_libraries(ImageStreamIO PUBLIC ${CUDA_LIBRARIES})
  target_compile_options(ImageStreamIO PUBLIC -DHAVE_CUDA)
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

install(TARGETS ImageStreamIO DESTINATION lib)
install(FILES ImageStreamIO.h DESTINATION include)
install(FILES ImageStruct.h DESTINATION include)
