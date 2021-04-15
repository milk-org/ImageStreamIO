cmake_minimum_required(VERSION 3.8)

project(ImageStreamIOWrap LANGUAGES CXX)

set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

if(NOT TARGET pybind11)
  if(NOT DEFINED WYRM_ROOT)
    set(WYRM_ROOT $ENV{WYRM_ROOT})
  endif()
  if(IS_DIRECTORY ${WYRM_ROOT})
    add_subdirectory(${WYRM_ROOT}/pybind pybind11_dir)
    pybind11_add_module(ImageStreamIOWrap-backport ImageStreamIOWrap.cpp)
  else()
    execute_process(COMMAND bash -c "python -m pybind11 --includes"
                    OUTPUT_VARIABLE pybind11_inc)
    execute_process(COMMAND bash -c "python3-config --extension-suffix"
                    OUTPUT_VARIABLE PYTHON_MODULE_EXTENSION)
    string(REPLACE "-I" "" pybind11_inc ${pybind11_inc})
    string(REPLACE " " ";" pybind11_inc ${pybind11_inc})
    string(REGEX REPLACE "\n$" "" pybind11_inc "${pybind11_inc}")
    string(REGEX REPLACE "\n$" "" PYTHON_MODULE_EXTENSION "${PYTHON_MODULE_EXTENSION}")

    add_library(ImageStreamIOWrap-backport MODULE ImageStreamIOWrap.cpp)

    target_compile_features(ImageStreamIOWrap-backport PRIVATE cxx_std_14)
    target_include_directories(ImageStreamIOWrap-backport PRIVATE "${pybind11_inc}")
    target_compile_options(ImageStreamIOWrap-backport PRIVATE "-Wno-deprecated-declarations")
    set_target_properties(ImageStreamIOWrap-backport
            PROPERTIES PREFIX
               "${PYTHON_MODULE_PREFIX}"
               SUFFIX
               "${PYTHON_MODULE_EXTENSION}")
  
    endif()
else()
  pybind11_add_module(ImageStreamIOWrap-backport ImageStreamIOWrap.cpp)
endif()  

target_link_libraries(ImageStreamIOWrap-backport PRIVATE ImageStreamIO-backport)

install(TARGETS ImageStreamIOWrap-backport
        EXPORT ImageStreamIOWrapConfig
        ARCHIVE DESTINATION python
        LIBRARY DESTINATION python)
