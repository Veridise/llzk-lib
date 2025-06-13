add_executable(r1cs-opt
  r1cs-opt.cpp
)

include_directories(${MLIR_INCLUDE_DIRS} 
${LLVM_INCLUDE_DIRS}
${R1CS_INCLUDE_DIR}
${CMAKE_BINARY_DIR}/r1cs/include
)

llzk_target_add_mlir_link_settings(r1cs-opt)
set_target_properties(r1cs-opt PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                          ${CMAKE_BINARY_DIR}/bin)
target_link_libraries(r1cs-opt
  PRIVATE
  LLVMSupport
  MLIRSupport
  MLIROptLib
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRTransforms
  MLIRTransforms
  R1CSDialect  # <- Your dialect library
)
