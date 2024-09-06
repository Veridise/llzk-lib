include_directories(${MLIR_INCLUDE_DIRS})

set(LLVM_TARGET_DEFINITIONS "ZKIRPasses.td")
mlir_tablegen(ZKIRPasses.h.inc -gen-pass-decls -dialect=zkir)
zkir_add_mlir_doc(ZKIRPassesDocGen ZKIRPasses.md -gen-pass-doc -dialect=zkir)

add_public_tablegen_target(ZKIRTransformIncGen)
add_dependencies(mlir-headers ZKIRTransformIncGen)
add_dependencies(ZKIRDialectHeaders ZKIRTransformIncGen)
