cmake_minimum_required(VERSION 3.17)

set(LIBNAME "ImageStreamIOWrap")
project(${LIBNAME} LANGUAGES CXX)

set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# =======================================
# BUILD TYPE
# =======================================
# Set a default build type if none was specified
set(default_build_type "Release")
if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}")
endif()

message("CMAKE_BUILD_TYPE : ${CMAKE_BUILD_TYPE}")

# =======================================
# NANOBIND
# =======================================
find_package(Python 3.9 COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)
nanobind_add_module(${LIBNAME} ImageStreamIOWrap.cpp)

# =======================================
# COVERAGE BUILD
# =======================================
if (CMAKE_BUILD_TYPE STREQUAL "Coverage")
  target_compile_definitions(${LIBNAME} PRIVATE COVERAGE_BUILD=1)
  target_link_options(${LIBNAME} PRIVATE --coverage)
  target_link_libraries(${LIBNAME} PRIVATE gcov)
endif()



target_link_libraries(${LIBNAME} PUBLIC ImageStreamIOpyMilk)

install(TARGETS ${LIBNAME}
        EXPORT ${LIBNAME}Config
        ARCHIVE DESTINATION python
        LIBRARY DESTINATION python)
