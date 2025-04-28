include_directories(${MLIR_INCLUDE_DIRS} ${LLZK_INCLUDE_DIR})

set(LLVM_TARGET_DEFINITIONS "Dialect.td")
mlir_tablegen(Dialect.h.inc -gen-dialect-decls -dialect=function)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs -dialect=function)

set(LLVM_TARGET_DEFINITIONS "Attrs.td")
mlir_tablegen(Attrs.h.inc -gen-attrdef-decls -attrdefs-dialect=function)
mlir_tablegen(Attrs.cpp.inc -gen-attrdef-defs -attrdefs-dialect=function)

set(LLVM_TARGET_DEFINITIONS "Ops.td")
mlir_tablegen(Ops.h.inc -gen-op-decls -dialect=function)
mlir_tablegen(Ops.cpp.inc -gen-op-defs -dialect=function)

llzk_add_mlir_doc(FunctionOpsDocGen FunctionDialect.md -gen-dialect-doc -dialect=function)

add_public_tablegen_target(FunctionDialectIncGen)
add_dependencies(mlir-headers FunctionDialectIncGen)
add_dependencies(LLZKDialectHeaders FunctionDialectIncGen)
