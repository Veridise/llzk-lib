include_directories(${MLIR_INCLUDE_DIRS} ${LLZK_INCLUDE_DIR})

set(LLVM_TARGET_DEFINITIONS "Dialect.td")
mlir_tablegen(Dialect.h.inc -gen-dialect-decls -dialect=llzk)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs -dialect=llzk)

set(LLVM_TARGET_DEFINITIONS "Attrs.td")
mlir_tablegen(Attrs.h.inc -gen-attrdef-decls -attrdefs-dialect=llzk)
mlir_tablegen(Attrs.cpp.inc -gen-attrdef-defs -attrdefs-dialect=llzk)

set(LLVM_TARGET_DEFINITIONS "Enums.td")
mlir_tablegen(Enums.h.inc -gen-enum-decls)
mlir_tablegen(Enums.cpp.inc -gen-enum-defs)

set(LLVM_TARGET_DEFINITIONS "Types.td")
mlir_tablegen(Types.h.inc -gen-typedef-decls -typedefs-dialect=llzk)
mlir_tablegen(Types.cpp.inc -gen-typedef-defs -typedefs-dialect=llzk)

set(LLVM_TARGET_DEFINITIONS "OpInterfaces.td")
mlir_tablegen(OpInterfaces.h.inc --gen-op-interface-decls)
mlir_tablegen(OpInterfaces.cpp.inc --gen-op-interface-defs)

set(LLVM_TARGET_DEFINITIONS "Ops.td")
mlir_tablegen(Ops.h.inc -gen-op-decls)
mlir_tablegen(Ops.cpp.inc -gen-op-defs)

llzk_add_mlir_doc(LLZKOpsDocGen Dialect.md -gen-dialect-doc -dialect=llzk)

add_public_tablegen_target(LLZKDialectIncGen)
add_dependencies(mlir-headers LLZKDialectIncGen)
add_dependencies(LLZKDialectHeaders LLZKDialectIncGen)
