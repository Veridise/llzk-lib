include_directories(${MLIR_INCLUDE_DIRS} ${LLZK_INCLUDE_DIR})

set(LLVM_TARGET_DEFINITIONS "Dialect.td")
mlir_tablegen(Dialect.h.inc -gen-dialect-decls -dialect=global)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs -dialect=global)

set(LLVM_TARGET_DEFINITIONS "OpInterfaces.td")
mlir_tablegen(OpInterfaces.h.inc --gen-op-interface-decls)
mlir_tablegen(OpInterfaces.cpp.inc --gen-op-interface-defs)

set(LLVM_TARGET_DEFINITIONS "Ops.td")
mlir_tablegen(Ops.h.inc -gen-op-decls -dialect=global)
mlir_tablegen(Ops.cpp.inc -gen-op-defs -dialect=global)

llzk_add_mlir_doc(GlobalOpsDocGen dialect/GlobalDialect.md -gen-dialect-doc -dialect=global)

add_public_tablegen_target(GlobalDialectIncGen)
add_dependencies(mlir-headers GlobalDialectIncGen)
add_dependencies(LLZKDialectHeaders GlobalDialectIncGen)
