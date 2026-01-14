include_directories(${MLIR_INCLUDE_DIRS} ${LLZK_INCLUDE_DIR})

set(LLVM_TARGET_DEFINITIONS "Dialect.td")
# New file: Attrs.td defines the overflow semantics attribute
set(LLVM_TARGET_DEFINITIONS "${LLVM_TARGET_DEFINITIONS};Attrs.td")
mlir_tablegen(Dialect.h.inc -gen-dialect-decls -dialect=cast)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs -dialect=cast)

set(LLVM_TARGET_DEFINITIONS "Ops.td")
# Ops.td depends on Attrs.td, ensure it is present for tablegen
set(LLVM_TARGET_DEFINITIONS "${LLVM_TARGET_DEFINITIONS};Ops.td")
mlir_tablegen(Ops.h.inc -gen-op-decls -dialect=cast)
mlir_tablegen(Ops.cpp.inc -gen-op-defs -dialect=cast)

llzk_add_mlir_doc(CastOpsDocGen dialect/CastDialect.md -gen-dialect-doc -dialect=cast)

add_public_tablegen_target(CastDialectIncGen)
add_dependencies(mlir-headers CastDialectIncGen)
add_dependencies(LLZKDialectHeaders CastDialectIncGen)