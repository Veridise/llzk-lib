include_directories(${MLIR_INCLUDE_DIRS} ${ZKIR_INCLUDE_DIR})

set(LLVM_TARGET_DEFINITIONS "ZKIRPasses.td")
mlir_tablegen(ZKIRPasses.h.inc -gen-pass-decls)
zkir_add_mlir_doc(ZKIRPassesDocGen ZKIRPasses.md -gen-pass-doc)

add_public_tablegen_target(ZKIRTransformIncGen)
add_dependencies(mlir-headers ZKIRTransformIncGen)
add_dependencies(ZKIRDialectHeaders ZKIRTransformIncGen)
