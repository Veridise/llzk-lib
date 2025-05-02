include_directories(${MLIR_INCLUDE_DIRS} ${LLZK_INCLUDE_DIR})

set(LLVM_TARGET_DEFINITIONS "TransformationPasses.td")
mlir_tablegen(TransformationPasses.h.inc -gen-pass-decls -name=Transformation)
llzk_add_mlir_doc(PolymorphicTransformationPassesDocGen passes/polymorphic/TransformationPasses.md -gen-pass-doc)

add_public_tablegen_target(PolymorphicTransformationIncGen)
add_dependencies(mlir-headers PolymorphicTransformationIncGen)
add_dependencies(LLZKDialectHeaders PolymorphicTransformationIncGen)
