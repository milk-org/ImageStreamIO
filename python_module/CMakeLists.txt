cmake_minimum_required(VERSION 3.8)

project(ImageStreamIOWrap LANGUAGES CXX)

hunter_add_package(pybind11)
find_package(pybind11 CONFIG REQUIRED)

find_package(pybind11 CONFIG REQUIRED)

add_library(ImageStreamIOWrap MODULE src/ImageStreamIOWrap.cpp)

target_compile_options(ImageStreamIOWrap PRIVATE -Wno-deprecated-declarations)

target_include_directories(ImageStreamIOWrap PRIVATE inc)

target_link_libraries(ImageStreamIOWrap
                      PRIVATE pybind11::pybind11
                              pybind11::embed
                              pybind11::module
                              ImageStreamIO)

set_target_properties(ImageStreamIOWrap
                      PROPERTIES PREFIX
                                 "${PYTHON_MODULE_PREFIX}"
                                 SUFFIX
                                 "${PYTHON_MODULE_EXTENSION}")
install(TARGETS ImageStreamIOWrap
        EXPORT ImageStreamIOWrapConfig
        ARCHIVE DESTINATION python
        LIBRARY DESTINATION python)
