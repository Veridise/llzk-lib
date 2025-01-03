add_executable(llzk-opt llzk-opt.cpp)
target_link_libraries(llzk-opt PUBLIC ${LLZK_DEP_DIALECT_LIBS} MLIROptLib
                                      LLZK::DialectRegistration)
llzk_target_add_mlir_link_settings(llzk-opt)
target_link_libraries(llzk-opt PUBLIC LLZKDialectHeaders MLIRIR)
include_directories(${MLIR_INCLUDE_DIRS} ${LLVM_INCLUDE_DIRS})

set_target_properties(llzk-opt PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                          ${CMAKE_BINARY_DIR}/bin)
install(
  TARGETS llzk-opt
  EXPORT LLZKTargets
  COMPONENT devtools)
