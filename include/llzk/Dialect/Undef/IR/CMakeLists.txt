include_directories(${MLIR_INCLUDE_DIRS} ${LLZK_INCLUDE_DIR})

set(LLVM_TARGET_DEFINITIONS "Dialect.td")
mlir_tablegen(Dialect.h.inc -gen-dialect-decls -dialect=undef)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs -dialect=undef)

set(LLVM_TARGET_DEFINITIONS "Ops.td")
mlir_tablegen(Ops.h.inc -gen-op-decls -dialect=undef)
mlir_tablegen(Ops.cpp.inc -gen-op-defs -dialect=undef)

llzk_add_mlir_doc(UndefOpsDocGen UndefDialect.md -gen-dialect-doc -dialect=undef)

add_public_tablegen_target(UndefDialectIncGen)
add_dependencies(mlir-headers UndefDialectIncGen)
add_dependencies(LLZKDialectHeaders UndefDialectIncGen)
