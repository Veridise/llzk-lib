add_executable(llzk-lsp-server llzk-lsp-server.cpp)
target_link_libraries(
  llzk-lsp-server PRIVATE ${LLZK_DEP_DIALECT_LIBS} MLIRLspServerLib
                         LLZK::DialectRegistration LLZKAnalysis
                         LLZKTransforms LLZKUtil LLZKValidators
                         LLZKDialectHeaders MLIRIR)
llzk_target_add_mlir_link_settings(llzk-lsp-server)
include_directories(${MLIR_INCLUDE_DIRS} ${LLVM_INCLUDE_DIRS})

set_target_properties(llzk-lsp-server PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                                 ${CMAKE_BINARY_DIR}/bin)
install(
  TARGETS llzk-lsp-server
  EXPORT LLZKTargets
  COMPONENT devtools)
