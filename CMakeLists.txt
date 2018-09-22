project(ImageStreamIO LANGUAGES C)

add_compile_options(-Ofast)

add_library(ImageStreamIO SHARED ImageStreamIO.c)
target_include_directories (ImageStreamIO PUBLIC
"${PROJECT_SOURCE_DIR}/src")
target_include_directories (ImageStreamIO PUBLIC
"${PROJECT_SOURCE_DIR}/..")

install(TARGETS ImageStreamIO DESTINATION lib)
install(FILES ImageStreamIO.h DESTINATION include)
install(FILES ImageStruct.h DESTINATION include)

