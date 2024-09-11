add_executable(zkir-opt zkir-opt.cpp)
target_link_libraries(zkir-opt PUBLIC
  ${ZKIR_DEP_DIALECT_LIBS}
  MLIROptLib
  ZKIR::AllDialects
)
zkir_target_add_mlir_link_settings(zkir-opt)
target_link_libraries(zkir-opt
  PUBLIC ZKIRDialectHeaders MLIRIR
)
include_directories(${MLIR_INCLUDE_DIRS} ${LLVM_INCLUDE_DIRS})

set_target_properties(zkir-opt PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
install(TARGETS zkir-opt EXPORT ZKIRTargets COMPONENT devtools)
