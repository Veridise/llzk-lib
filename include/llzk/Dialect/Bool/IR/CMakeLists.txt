include_directories(${MLIR_INCLUDE_DIRS} ${LLZK_INCLUDE_DIR})

set(LLVM_TARGET_DEFINITIONS "Dialect.td")
mlir_tablegen(Dialect.h.inc -gen-dialect-decls -dialect=bool)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs -dialect=bool)

set(LLVM_TARGET_DEFINITIONS "Attrs.td")
mlir_tablegen(Attrs.h.inc -gen-attrdef-decls -attrdefs-dialect=bool)
mlir_tablegen(Attrs.cpp.inc -gen-attrdef-defs -attrdefs-dialect=bool)

set(LLVM_TARGET_DEFINITIONS "Enums.td")
mlir_tablegen(Enums.h.inc -gen-enum-decls)
mlir_tablegen(Enums.cpp.inc -gen-enum-defs)

set(LLVM_TARGET_DEFINITIONS "Ops.td")
mlir_tablegen(Ops.h.inc -gen-op-decls -dialect=bool)
mlir_tablegen(Ops.cpp.inc -gen-op-defs -dialect=bool)

llzk_add_mlir_doc(BooleanOpsDocGen dialect/BoolDialect.md -gen-dialect-doc -dialect=bool)

add_public_tablegen_target(BoolDialectIncGen)
add_dependencies(mlir-headers BoolDialectIncGen)
add_dependencies(LLZKDialectHeaders BoolDialectIncGen)
