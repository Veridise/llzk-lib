include_directories(${MLIR_INCLUDE_DIRS} ${ZKIR_INCLUDE_DIR})

set(LLVM_TARGET_DEFINITIONS "Dialect.td")
mlir_tablegen(Dialect.h.inc -gen-dialect-decls -dialect=zkir)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs -dialect=zkir)

set(LLVM_TARGET_DEFINITIONS "Ops.td")
mlir_tablegen(Ops.h.inc -gen-op-decls)
mlir_tablegen(Ops.cpp.inc -gen-op-defs)
zkir_add_mlir_doc(ZKIROpsDocGen Dialect.md -gen-dialect-doc -dialect=zkir)

set(LLVM_TARGET_DEFINITIONS "Types.td")
mlir_tablegen(Types.h.inc -gen-typedef-decls -typedefs-dialect=zkir)
mlir_tablegen(Types.cpp.inc -gen-typedef-defs -typedefs-dialect=zkir)

set(LLVM_TARGET_DEFINITIONS "Attrs.td")
mlir_tablegen(Attrs.h.inc -gen-attrdef-decls -attrdefs-dialect=zkir)
mlir_tablegen(Attrs.cpp.inc -gen-attrdef-defs -attrdefs-dialect=zkir)

set(LLVM_TARGET_DEFINITIONS "Enums.td")
mlir_tablegen(Enums.h.inc -gen-enum-decls)
mlir_tablegen(Enums.cpp.inc -gen-enum-defs)

add_public_tablegen_target(ZKIRDialectIncGen)
add_dependencies(mlir-headers ZKIRDialectIncGen)
add_dependencies(ZKIRDialectHeaders ZKIRDialectIncGen)
