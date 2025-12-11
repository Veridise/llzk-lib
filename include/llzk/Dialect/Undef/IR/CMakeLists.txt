include_directories(${MLIR_INCLUDE_DIRS} ${LLZK_INCLUDE_DIR})

set(LLVM_TARGET_DEFINITIONS "Dialect.td")
mlir_tablegen(Dialect.h.inc -gen-dialect-decls -dialect=undef)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs -dialect=undef)
llzk_tablegen(Dialect.capi.test.cpp.inc -gen-dialect-capi-tests -prefix=llzk -dialect=undef)

set(LLVM_TARGET_DEFINITIONS "Ops.td")
mlir_tablegen(Ops.h.inc -gen-op-decls -dialect=undef)
mlir_tablegen(Ops.cpp.inc -gen-op-defs -dialect=undef)
llzk_tablegen(Ops.capi.h.inc -gen-op-capi-header -prefix=llzk -dialect=undef)
llzk_tablegen(Ops.capi.cpp.inc -gen-op-capi-impl -prefix=llzk -dialect=undef)
llzk_tablegen(Ops.capi.test.cpp.inc -gen-op-capi-tests -prefix=llzk -dialect=undef)

llzk_add_mlir_doc(UndefOpsDocGen dialect/UndefDialect.md -gen-dialect-doc -dialect=undef)

add_public_tablegen_target(UndefDialectIncGen)
add_dependencies(mlir-headers UndefDialectIncGen)
add_dependencies(LLZKDialectHeaders UndefDialectIncGen)
