include_directories(${MLIR_INCLUDE_DIRS} ${LLZK_INCLUDE_DIR})

set(LLVM_TARGET_DEFINITIONS "Dialect.td")
mlir_tablegen(Dialect.h.inc -gen-dialect-decls -dialect=llzk_include)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs -dialect=llzk_include)

set(LLVM_TARGET_DEFINITIONS "Ops.td")
mlir_tablegen(Ops.h.inc -gen-op-decls -dialect=llzk_include)
mlir_tablegen(Ops.cpp.inc -gen-op-defs -dialect=llzk_include)

llzk_add_mlir_doc(IncludeOpsDocGen IncludeDialect.md -gen-dialect-doc -dialect=llzk_include)

add_public_tablegen_target(IncludeDialectIncGen)
add_dependencies(mlir-headers IncludeDialectIncGen)
add_dependencies(LLZKDialectHeaders IncludeDialectIncGen)
