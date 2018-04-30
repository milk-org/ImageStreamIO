project(lib_ImageStreamIO_project)

include_directories ("${PROJECT_SOURCE_DIR}/src")
include_directories ("${PROJECT_SOURCE_DIR}/..")
add_library(ImageStreamIO SHARED ImageStreamIO.c)

install(TARGETS ImageStreamIO DESTINATION lib)
install(FILES ImageStreamIO.h DESTINATION include)
install(FILES ImageStruct.h DESTINATION include)

