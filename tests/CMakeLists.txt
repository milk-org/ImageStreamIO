project(ImageStreamIOTest)

find_package(GTest REQUIRED)

add_executable(ImageStreamIOTest
ImageStreamIO_unitTest.cpp
    ImageStreamIO_subTest_Operations.cpp
)

target_link_libraries(ImageStreamIOTest
    PUBLIC GTest::GTest GTest::Main ImageStreamIO
)

gtest_add_tests(TARGET ImageStreamIOTest
TEST_SUFFIX .noArgs
TEST_LIST   noArgsTests
)

set_tests_properties(${noArgsTests}   PROPERTIES TIMEOUT 10)
