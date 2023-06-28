project(ISIO_Catch2)

add_executable(ISIO_Catch2
  ISIO_C2_main.cpp
  ISIO_C2_unitTests.cpp
  ImageStreamIO_subTest_Operations.cpp
)

target_link_libraries(ISIO_Catch2 ImageStreamIO pthread
                     )

add_test(NAME ISIO_Catch2
         COMMAND ISIO_Catch2
        )
