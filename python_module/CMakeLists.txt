cmake_minimum_required(VERSION 3.8)

project(ImageStreamIOWrap LANGUAGES CXX)

set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

find_package(pybind11)

if(NOT TARGET pybind11)
  execute_process(COMMAND bash -c "${PYTHON_EXECUTABLE} -m pybind11 --includes"
                  OUTPUT_VARIABLE pybind11_inc)
  execute_process(COMMAND bash -c "${PYTHON_EXECUTABLE} -c 'import sysconfig; print(sysconfig.get_config_var(\"EXT_SUFFIX\"))'"
                  OUTPUT_VARIABLE PYTHON_MODULE_EXTENSION)
  string(REPLACE "-I" "" pybind11_inc ${pybind11_inc})
  string(REPLACE " " ";" pybind11_inc ${pybind11_inc})
  string(REGEX REPLACE "\n$" "" pybind11_inc "${pybind11_inc}")
  string(REGEX REPLACE "\n$" "" PYTHON_MODULE_EXTENSION "${PYTHON_MODULE_EXTENSION}")

  add_library(ImageStreamIOWrap MODULE ImageStreamIOWrap.cpp)

  target_compile_features(ImageStreamIOWrap PUBLIC cxx_std_14)
  target_include_directories(ImageStreamIOWrap PUBLIC "${pybind11_inc}")
  target_compile_options(ImageStreamIOWrap PUBLIC "-Wno-deprecated-declarations")
  set_target_properties(ImageStreamIOWrap
          PROPERTIES PREFIX
              "${PYTHON_MODULE_PREFIX}"
              SUFFIX
              "${PYTHON_MODULE_EXTENSION}")
else()
  pybind11_add_module(ImageStreamIOWrap ImageStreamIOWrap.cpp)
endif()

target_link_libraries(ImageStreamIOWrap PUBLIC ImageStreamIO)

install(TARGETS ImageStreamIOWrap
        EXPORT ImageStreamIOWrapConfig
        ARCHIVE DESTINATION python
        LIBRARY DESTINATION python)
