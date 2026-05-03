cmake_minimum_required(VERSION 3.17)

set(LIBNAME "ImageStreamIOpyMilk") # FIXME !!
set(LIBNAME_PCFILES "ImageStreamIO")
set(SRCNAME "ImageStreamIO")

message("")
message(" SRCNAME = ${SRCNAME} -> LIBNAME = ${LIBNAME}")

option(build_python_module "Compile Python Wrappers" OFF)
option(gtest_build_tests "Build Unit Tests" OFF)

project(${LIBNAME} LANGUAGES C)

# Version number
set ( VERSION_MAJOR 2 )
set ( VERSION_MINOR 01 )
set ( VERSION_PATCH 00 )
set ( VERSION_OPTION "" )
set ( PROJECT_VERSION "${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}" )
configure_file (
  "${PROJECT_SOURCE_DIR}/${LIBNAME_PCFILES}_config.h.in"
  "${PROJECT_SOURCE_DIR}//${LIBNAME_PCFILES}_config.h"
  )
message(" VERSION = ${PROJECT_VERSION}")

if(USE_CUDA)
  set(IPC_REGISTRY_SRC ImageStreamIO_ipc_registry.c)
else()
  set(IPC_REGISTRY_SRC "")
endif()

add_library(${LIBNAME} SHARED ${SRCNAME}.c timeutils.c ${IPC_REGISTRY_SRC})

# set -C99 flag for 'for' loop initial declartaions
set_property(TARGET ${LIBNAME} PROPERTY C_STANDARD 99)

# =======================================
# BUILD TYPE
# =======================================
# cmake -DCMAKE_BUILD_TYPE=Debug
# option: add -fanalyzer if gcc10+ (warning: this will take a while to run)
set(CMAKE_C_FLAGS_DEBUG     "-O0 -g -Wall -Wextra -std=gnu17 -Wno-sign-compare")
# cmake -DCMAKE_BUILD_TYPE=Coverage
set(CMAKE_C_FLAGS_COVERAGE     "-O0 -g --coverage -fprofile-arcs -ftest-coverage")
set(CMAKE_CXX_FLAGS_COVERAGE     "-O0 -g --coverage -fprofile-arcs -ftest-coverage")
# cmake -DCMAKE_BUILD_TYPE=Release
set(CMAKE_C_FLAGS_RELEASE     "-Ofast -DNDEBUG -std=gnu17")


# Set a default build type if none was specified
set(default_build_type "Release")
if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}")
endif()

message("CMAKE_BUILD_TYPE : ${CMAKE_BUILD_TYPE}")


# Setting these PRIVATE/PUBLIC tags means that sibling folders
# do NOT inherit this folder as a valid -I<> target.
# So a sibling project MUST #include "ImageStreamIO/ImageStruct.h"
# instead of just #include "ImageStruct.h"
target_include_directories(${LIBNAME} PRIVATE ${PROJECT_SOURCE_DIR})
target_include_directories(${LIBNAME} PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
                                             $<INSTALL_INTERFACE:include/${SRCNAME}>
                                             $<INSTALL_INTERFACE:include>)

find_package(PkgConfig REQUIRED)

pkg_check_modules(CFITSIO cfitsio)
if(${CFITSIO_FOUND})
  link_directories(${CFITSIO_LIBRARY_DIRS})
  target_compile_definitions(${LIBNAME} PUBLIC USE_CFITSIO=1)
  target_include_directories(${LIBNAME} PUBLIC ${CFITSIO_INCLUDE_DIRS})
  target_link_directories(${LIBNAME} PUBLIC ${CFITSIO_INCLUDE_DIRS})
endif()

if(USE_CUDA)
  find_package(CUDAToolkit 9.0 REQUIRED)
  message(STATUS "Found CUDA ${CUDAToolkit_VERSION} at ${CUDAToolkit_LIBRARY_DIR}")
  message("---- CUDA_INCLUDE_DIRS = ${CUDAToolkit_INCLUDE_DIRS}")
  target_include_directories(${LIBNAME} PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
  target_link_libraries(${LIBNAME} PRIVATE CUDA::toolkit CUDA::cudart)
  target_compile_options(${LIBNAME} PUBLIC -DHAVE_CUDA)

  set(BUILD_FLAGS "${BUILD_FLAGS} -DHAVE_CUDA" )
endif(USE_CUDA)

# =======================================
# COVERAGE BUILD
# =======================================
if (CMAKE_BUILD_TYPE STREQUAL "Coverage")
  target_compile_definitions(${LIBNAME} PRIVATE COVERAGE_BUILD=1)
  target_link_options(${LIBNAME} PRIVATE --coverage)
  target_link_libraries(${LIBNAME} PRIVATE gcov)
endif()

if(DAO_COMPAT)
  target_compile_options(${LIBNAME} PUBLIC -DDAO_COMPAT)
endif(DAO_COMPAT)

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
# CONFIG.CMAKE
#

set(INCLUDE_INSTALL_DIR ${CMAKE_INSTALL_PREFIX}/include
    CACHE PATH "Location of header files" )
set(LIBRARY_INSTALL_DIR ${CMAKE_INSTALL_PREFIX}/lib
    CACHE PATH "Location of libraries" )

# Export the targets to a file for use in the Config.cmake file
install(EXPORT ${LIBNAME}Targets
    FILE "${LIBNAME}Targets.cmake"
    NAMESPACE ${LIBNAME}::
    DESTINATION lib/cmake
)

include(CMakePackageConfigHelpers)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/${LIBNAME}ConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/${LIBNAME}Config.cmake"
    INSTALL_DESTINATION lib/cmake
    PATH_VARS INCLUDE_INSTALL_DIR LIBRARY_INSTALL_DIR
)

# Install the Config.cmake and ConfigVersion.cmake files
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/${LIBNAME}Config.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/${LIBNAME}ConfigVersion.cmake"
    DESTINATION lib/cmake
)

########################################################################
#
# pkg-config
#
string(APPEND LINKSTRING "-l${LIBNAME} ")
set(INSTALL_PKGCONFIG_DIR "lib/pkgconfig"
    CACHE PATH "Installation directory for pkgconfig (.pc) files")

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${LIBNAME_PCFILES}.pc.in
               ${CMAKE_CURRENT_BINARY_DIR}/${LIBNAME_PCFILES}.pc @ONLY)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${LIBNAME_PCFILES}.pc
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


install(TARGETS ${LIBNAME} EXPORT ${LIBNAME}Targets DESTINATION lib)
install(FILES
  ${SRCNAME}.h
  ImageStruct.h
  ImageStreamIOError.h
  DESTINATION include/${SRCNAME})