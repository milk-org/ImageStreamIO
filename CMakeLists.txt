project(ImageStreamIO LANGUAGES C)

add_compile_options(-Ofast)

add_library(ImageStreamIO SHARED ImageStreamIO.c)
target_include_directories (ImageStreamIO PUBLIC
"${PROJECT_SOURCE_DIR}/src")
target_include_directories (ImageStreamIO PUBLIC
"${PROJECT_SOURCE_DIR}/..")

if(USE_CUDA)
find_package( CUDA REQUIRED )
target_include_directories(ImageStreamIO PUBLIC ${CUDA_INCLUDE_DIRS})
target_link_libraries(ImageStreamIO PUBLIC  ${CUDA_LIBRARIES})
target_compile_options(ImageStreamIO PUBLIC -DHAVE_CUDA)
endif(USE_CUDA)

install(TARGETS ImageStreamIO DESTINATION lib)
install(FILES ImageStreamIO.h DESTINATION include)
install(FILES ImageStruct.h DESTINATION include)

