file(GLOB_RECURSE LLZK_CAPI_TEST_SOURCES "*.cpp")
add_executable(LLZK_CAPI_Tests ${LLZK_CAPI_TEST_SOURCES})

add_subdirectory(Dialect)
target_link_libraries(
  LLZK_CAPI_Tests
  PUBLIC LLZK::DialectRegistration LLZK::AllDialects ${GTEST_LIB_TARGETS}
  ${GTEST_EXE_TARGET} ${LLZK_DEP_DIALECT_LIBS} MLIRParser
  LLZK::LLZKAnalysis LLZK::LLZKCAPI LLZK::LLZKTransforms LLZK::LLZKValidators)

llzk_gtest_suite(LLZK_CAPI_Tests)
