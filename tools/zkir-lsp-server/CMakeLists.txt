add_executable(zkir-lsp-server zkir-lsp-server.cpp)
target_link_libraries(zkir-lsp-server PUBLIC
  ${ZKIR_DEP_DIALECT_LIBS}
  MLIRLspServerLib
  ZKIR::AllDialects
)
zkir_target_add_mlir_link_settings(zkir-lsp-server)
target_link_libraries(zkir-lsp-server
  PUBLIC ZKIRDialectHeaders MLIRIR
)
include_directories(${MLIR_INCLUDE_DIRS} ${LLVM_INCLUDE_DIRS})

set_target_properties(zkir-lsp-server PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
install(TARGETS zkir-lsp-server EXPORT ZKIRTargets COMPONENT devtools)
