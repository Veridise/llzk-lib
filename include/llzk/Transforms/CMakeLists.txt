include_directories(${MLIR_INCLUDE_DIRS} ${LLZK_INCLUDE_DIR})

set(LLVM_TARGET_DEFINITIONS "LLZKTransformationPasses.td")
mlir_tablegen(LLZKTransformationPasses.h.inc -gen-pass-decls -name=Transformation)
llzk_add_mlir_doc(LLZKTransformationPassesDocGen passes/LLZKTransformationPasses.md -gen-pass-doc)

add_public_tablegen_target(LLZKTransformationIncGen)
add_dependencies(mlir-headers LLZKTransformationIncGen)
add_dependencies(LLZKDialectHeaders LLZKTransformationIncGen)
