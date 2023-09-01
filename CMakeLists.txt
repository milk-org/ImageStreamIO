cmake_minimum_required(VERSION 3.8)

set(LIBNAME "ImageStreamIO")
set(SRCNAME "ImageStreamIO")

message("")
message(" SRCNAME = ${SRCNAME} -> LIBNAME = ${LIBNAME}")

option(build_python_module "Compile Python Wrappers" OFF)
option(gtest_build_tests "Build Unit Tests" OFF)

project(${LIBNAME} LANGUAGES C)

add_compile_options(-Ofast)

add_library(${LIBNAME} SHARED ${SRCNAME}.c)

# set -C99 flag for 'for' loop initial declartaions
set_property(TARGET ${LIBNAME} PROPERTY C_STANDARD 99)

find_package(PkgConfig REQUIRED)
pkg_check_modules(CFITSIO cfitsio)
if(${CFITSIO_FOUND})
  link_directories(${CFITSIO_LIBRARY_DIRS})
  target_compile_definitions(${LIBNAME} PUBLIC USE_CFITSIO=1)
  target_include_directories(${LIBNAME} PUBLIC ${CFITSIO_INCLUDE_DIRS})
  target_link_directories(${LIBNAME} PUBLIC ${CFITSIO_INCLUDE_DIRS})
endif()

target_include_directories(${LIBNAME} PUBLIC ${PROJECT_SOURCE_DIR})

if(USE_CUDA)
  find_package(CUDA REQUIRED)
  target_include_directories(${LIBNAME} PUBLIC ${CUDA_INCLUDE_DIRS})
  target_link_libraries(${LIBNAME} PUBLIC ${CUDA_LIBRARIES})
  target_compile_options(${LIBNAME} PUBLIC -DHAVE_CUDA)
endif(USE_CUDA)

#
# Python wrap.
#
# The python wrapper is not built by default.  To build it, set the build_python_module
# option to ON.  You can do it by running ccmake or specifying the
# -Dbuild_python_module=ON flag when running cmake.

if(build_python_module)
  add_subdirectory(python_module)
endif()

########################################################################
#
# pkg-config
#
string(APPEND LINKSTRING "-l${LIBNAME} ")
set(INSTALL_PKGCONFIG_DIR "lib/pkgconfig"
    CACHE PATH "Installation directory for pkgconfig (.pc) files")

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${LIBNAME}.pc.in
               ${CMAKE_CURRENT_BINARY_DIR}/${LIBNAME}.pc @ONLY)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${LIBNAME}.pc
        DESTINATION "${INSTALL_PKGCONFIG_DIR}")



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
