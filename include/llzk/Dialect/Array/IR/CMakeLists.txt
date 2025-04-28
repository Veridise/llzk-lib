include_directories(${MLIR_INCLUDE_DIRS} ${LLZK_INCLUDE_DIR})

set(LLVM_TARGET_DEFINITIONS "Dialect.td")
mlir_tablegen(Dialect.h.inc -gen-dialect-decls -dialect=array)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs -dialect=array)

set(LLVM_TARGET_DEFINITIONS "Ops.td")
mlir_tablegen(Ops.h.inc -gen-op-decls -dialect=array)
mlir_tablegen(Ops.cpp.inc -gen-op-defs -dialect=array)

set(LLVM_TARGET_DEFINITIONS "OpInterfaces.td")
mlir_tablegen(OpInterfaces.h.inc --gen-op-interface-decls)
mlir_tablegen(OpInterfaces.cpp.inc --gen-op-interface-defs)

llzk_add_mlir_doc(ArrayOpsDocGen ArrayDialect.md -gen-dialect-doc -dialect=array)

add_public_tablegen_target(ArrayDialectIncGen)
add_dependencies(mlir-headers ArrayDialectIncGen)
add_dependencies(LLZKDialectHeaders ArrayDialectIncGen)
