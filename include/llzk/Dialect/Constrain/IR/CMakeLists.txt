include_directories(${MLIR_INCLUDE_DIRS} ${LLZK_INCLUDE_DIR})

set(LLVM_TARGET_DEFINITIONS "Dialect.td")
mlir_tablegen(Dialect.h.inc -gen-dialect-decls -dialect=constrain)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs -dialect=constrain)

set(LLVM_TARGET_DEFINITIONS "OpInterfaces.td")
mlir_tablegen(OpInterfaces.h.inc --gen-op-interface-decls)
mlir_tablegen(OpInterfaces.cpp.inc --gen-op-interface-defs)

set(LLVM_TARGET_DEFINITIONS "Ops.td")
mlir_tablegen(Ops.h.inc -gen-op-decls -dialect=constrain)
mlir_tablegen(Ops.cpp.inc -gen-op-defs -dialect=constrain)

llzk_add_mlir_doc(ConstrainOpsDocGen dialect/ConstrainDialect.md -gen-dialect-doc -dialect=constrain)

add_public_tablegen_target(ConstrainDialectIncGen)
add_dependencies(mlir-headers ConstrainDialectIncGen)
add_dependencies(LLZKDialectHeaders ConstrainDialectIncGen)
