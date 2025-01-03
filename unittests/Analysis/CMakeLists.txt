add_executable(LLZKCallGraphTests CallGraphTests.cpp)

target_link_libraries(
  LLZKCallGraphTests
  PUBLIC LLZK::DialectRegistration LLZK::AllDialects ${GTEST_LIB_TARGETS}
         ${GTEST_EXE_TARGET} ${LLZK_DEP_DIALECT_LIBS} MLIRParser)

llzk_gtest_suite(LLZKCallGraphTests)

