include_directories(${MLIR_INCLUDE_DIRS} ${LLZK_INCLUDE_DIR})

set(LLVM_TARGET_DEFINITIONS "Dialect.td")
mlir_tablegen(Dialect.h.inc -gen-dialect-decls -dialect=poly)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs -dialect=poly)

set(LLVM_TARGET_DEFINITIONS "Types.td")
mlir_tablegen(Types.h.inc -gen-typedef-decls -typedefs-dialect=poly)
mlir_tablegen(Types.cpp.inc -gen-typedef-defs -typedefs-dialect=poly)

set(LLVM_TARGET_DEFINITIONS "Ops.td")
mlir_tablegen(Ops.h.inc -gen-op-decls -dialect=poly)
mlir_tablegen(Ops.cpp.inc -gen-op-defs -dialect=poly)

llzk_add_mlir_doc(PolymorphicOpsDocGen PolymorphicDialect.md -gen-dialect-doc -dialect=poly)

add_public_tablegen_target(PolymorphicDialectIncGen)
add_dependencies(mlir-headers PolymorphicDialectIncGen)
add_dependencies(LLZKDialectHeaders PolymorphicDialectIncGen)
