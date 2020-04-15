cmake_minimum_required(VERSION 3.8)

project(ImageStreamIOWrap LANGUAGES CXX)

if(NOT TARGET pybind11)
  if(NOT DEFINED WYRM_ROOT)
    set(WYRM_ROOT $ENV{WYRM_ROOT})
  endif()
  if(IS_DIRECTORY ${WYRM_ROOT})
    add_subdirectory(${WYRM_ROOT}/pybind pybind11_dir)
    pybind11_add_module(ImageStreamIOWrap src/ImageStreamIOWrap.cpp)
  else()
    execute_process(COMMAND bash -c "python -m pybind11 --includes"
                    OUTPUT_VARIABLE pybind11_inc)
    execute_process(COMMAND bash -c "python3-config --extension-suffix"
                    OUTPUT_VARIABLE PYTHON_MODULE_EXTENSION)
    string(REPLACE "-I" "" pybind11_inc ${pybind11_inc})
    string(REPLACE " " ";" pybind11_inc ${pybind11_inc})
    string(REGEX REPLACE "\n$" "" pybind11_inc "${pybind11_inc}")
    string(REGEX REPLACE "\n$" "" PYTHON_MODULE_EXTENSION "${PYTHON_MODULE_EXTENSION}")

    add_library(ImageStreamIOWrap MODULE src/ImageStreamIOWrap.cpp)

    target_compile_features(ImageStreamIOWrap PRIVATE cxx_std_14)
    target_include_directories(ImageStreamIOWrap PRIVATE "${pybind11_inc}")
    target_compile_options(ImageStreamIOWrap PRIVATE "-Wno-deprecated-declarations")
    set_target_properties(ImageStreamIOWrap
            PROPERTIES PREFIX
               "${PYTHON_MODULE_PREFIX}"
               SUFFIX
               "${PYTHON_MODULE_EXTENSION}")
  
    endif()
else()
  pybind11_add_module(ImageStreamIOWrap src/ImageStreamIOWrap.cpp)
endif()  

target_link_libraries(ImageStreamIOWrap PRIVATE ImageStreamIO)

install(TARGETS ImageStreamIOWrap
        EXPORT ImageStreamIOWrapConfig
        ARCHIVE DESTINATION python
        LIBRARY DESTINATION python)
