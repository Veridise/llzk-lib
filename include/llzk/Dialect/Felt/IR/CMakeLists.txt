include_directories(${MLIR_INCLUDE_DIRS} ${LLZK_INCLUDE_DIR})

set(LLVM_TARGET_DEFINITIONS "Dialect.td")
mlir_tablegen(Dialect.h.inc -gen-dialect-decls -dialect=felt)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs -dialect=felt)

set(LLVM_TARGET_DEFINITIONS "Attrs.td")
mlir_tablegen(Attrs.h.inc -gen-attrdef-decls -attrdefs-dialect=felt)
mlir_tablegen(Attrs.cpp.inc -gen-attrdef-defs -attrdefs-dialect=felt)

set(LLVM_TARGET_DEFINITIONS "Types.td")
mlir_tablegen(Types.h.inc -gen-typedef-decls -typedefs-dialect=felt)
mlir_tablegen(Types.cpp.inc -gen-typedef-defs -typedefs-dialect=felt)

set(LLVM_TARGET_DEFINITIONS "Ops.td")
mlir_tablegen(Ops.h.inc -gen-op-decls -dialect=felt)
mlir_tablegen(Ops.cpp.inc -gen-op-defs -dialect=felt)

llzk_add_mlir_doc(FeltOpsDocGen dialect/FeltDialect.md -gen-dialect-doc -dialect=felt)

add_public_tablegen_target(FeltDialectIncGen)
add_dependencies(mlir-headers FeltDialectIncGen)
add_dependencies(LLZKDialectHeaders FeltDialectIncGen)
