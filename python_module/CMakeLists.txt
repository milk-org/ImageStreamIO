cmake_minimum_required(VERSION 3.8)

project(ImageStreamIOWrap LANGUAGES CXX)

execute_process(COMMAND bash -c "python -m pybind11 --includes"
                OUTPUT_VARIABLE pybind11_inc)
execute_process(COMMAND bash -c "python3-config --extension-suffix"
                OUTPUT_VARIABLE PYTHON_MODULE_EXTENSION)

string(REPLACE "-I" "" pybind11_inc ${pybind11_inc})
string(REPLACE " " ";" pybind11_inc ${pybind11_inc})
string(REGEX REPLACE "\n$" "" pybind11_inc "${pybind11_inc}")
string(REGEX REPLACE "\n$" "" PYTHON_MODULE_EXTENSION "${PYTHON_MODULE_EXTENSION}")

# message("toto:"${pybind11_inc})
# message("toto:"${PYTHON_MODULE_PREFIX})
# message("toto:"${PYTHON_MODULE_EXTENSION})

add_library(ImageStreamIOWrap MODULE src/ImageStreamIOWrap.cpp)

target_compile_options(ImageStreamIOWrap PRIVATE -Wno-deprecated-declarations)

target_include_directories(ImageStreamIOWrap PRIVATE inc ${pybind11_inc})

target_link_libraries(ImageStreamIOWrap PRIVATE ImageStreamIO)

# if(NOT TARGET pybind11)
#   if(NOT DEFINED WYRM_ROOT)
#     set(WYRM_ROOT $ENV{WYRM_ROOT})
#   endif()
#   if(IS_DIRECTORY ${WYRM_ROOT})
#     add_subdirectory(${WYRM_ROOT} wyrm_dir)
#     target_link_libraries(ImageStreamIOWrap PRIVATE pybind11::module)
#   endif()
# endif()

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
