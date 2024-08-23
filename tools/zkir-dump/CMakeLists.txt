add_executable(zkir-dump zkir-dump.cpp)
target_link_libraries(zkir-dump PUBLIC
  ${ZKIR_DEP_DIALECT_LIBS}
  MLIRParser
  ZKIR::AllDialects
)
zkir_target_add_mlir_link_settings(zkir-dump)

set_target_properties(zkir-dump PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
install(TARGETS zkir-dump EXPORT ZKIRTargets)
