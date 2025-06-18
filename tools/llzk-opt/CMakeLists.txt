add_executable(llzk-opt llzk-opt.cpp)
target_link_libraries(llzk-opt PUBLIC ${LLZK_DEP_DIALECT_LIBS} MLIROptLib
                      LLZK::DialectRegistration LLZK::LLZKAnalysis
                      LLZK::LLZKTransforms LLZK::LLZKUtil LLZK::LLZKValidators
                      LLZKDialectHeaders MLIRIR R1CSDialect)
llzk_target_add_mlir_link_settings(llzk-opt)
include_directories(${MLIR_INCLUDE_DIRS} ${LLVM_INCLUDE_DIRS} ${R1CS_INCLUDE_DIR} ${CMAKE_BINARY_DIR}/r1cs/include/)

set_target_properties(llzk-opt PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                          ${CMAKE_BINARY_DIR}/bin)
install(
  TARGETS llzk-opt
  EXPORT LLZKTargets
  COMPONENT devtools)
