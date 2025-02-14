file(GLOB_RECURSE LLZK_UTIL_TEST_SOURCES "*.cpp")
add_executable(LLZK_Util_Tests ${LLZK_UTIL_TEST_SOURCES})

target_link_libraries(
  LLZK_Util_Tests
  PUBLIC LLZK::AllDialects ${GTEST_LIB_TARGETS} ${GTEST_EXE_TARGET}
  ${LLZK_DEP_DIALECT_LIBS} MLIRParser)

llzk_gtest_suite(LLZK_Util_Tests)
