cmake_minimum_required(VERSION 3.8)

project(ImageStreamIOWrap LANGUAGES CXX)

if(use_hunter)
  hunter_add_package(pybind11)
  find_package(pybind11 CONFIG REQUIRED)
else()
  if(NOT TARGET pybind11)
    if(NOT DEFINED WYRM_ROOT)
      set(WYRM_ROOT $ENV{WYRM_ROOT})
    endif()
    if(IS_DIRECTORY ${WYRM_ROOT})
      add_subdirectory(${WYRM_ROOT} wyrm_dir)
    else()
      message(
        FATAL_ERROR
          "wyrm not found, specify it by set WYRM_ROOT to the right path")
    endif()
  endif()
endif()

add_library(ImageStreamIOWrap MODULE src/ImageStreamIOWrap.cpp)

target_compile_options(ImageStreamIOWrap PRIVATE -Wno-deprecated-declarations)

target_include_directories(ImageStreamIOWrap PRIVATE inc)

target_link_libraries(ImageStreamIOWrap PRIVATE pybind11::module ImageStreamIO)

target_compile_features(ImageStreamIOWrap PRIVATE cxx_std_14)

set_target_properties(ImageStreamIOWrap
                      PROPERTIES PREFIX
                                 "${PYTHON_MODULE_PREFIX}"
                                 SUFFIX
                                 "${PYTHON_MODULE_EXTENSION}")
install(TARGETS ImageStreamIOWrap
        EXPORT ImageStreamIOWrapConfig
        ARCHIVE DESTINATION python
        LIBRARY DESTINATION python)
