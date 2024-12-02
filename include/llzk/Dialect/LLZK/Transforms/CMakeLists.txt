include_directories(${MLIR_INCLUDE_DIRS} ${LLZK_INCLUDE_DIR})

set(LLVM_TARGET_DEFINITIONS "LLZKPasses.td")
mlir_tablegen(LLZKPasses.h.inc -gen-pass-decls)
llzk_add_mlir_doc(LLZKPassesDocGen LLZKPasses.md -gen-pass-doc)

add_public_tablegen_target(LLZKTransformIncGen)
add_dependencies(mlir-headers LLZKTransformIncGen)
add_dependencies(LLZKDialectHeaders LLZKTransformIncGen)
