add_executable(HelloWorldTest
  HelloWorldTest.cpp
)

target_link_libraries(HelloWorldTest
  PUBLIC LLZK::AllDialects ${GTEST_LIB_TARGETS} ${GTEST_EXE_TARGET} ${LLZK_DEP_DIALECT_LIBS} MLIRParser
)

llzk_gtest_suite(HelloWorldTest)