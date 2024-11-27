include_directories(${MLIR_INCLUDE_DIRS} ${ZKIR_INCLUDE_DIR})

set(LLVM_TARGET_DEFINITIONS "Constants.td")

mlir_tablegen(Constants.h.inc -gen-enum-decls)
mlir_tablegen(Constants.cpp.inc -gen-enum-defs)

add_public_tablegen_target(ZKIRUtilIncGen)
add_dependencies(mlir-headers ZKIRUtilIncGen)
add_dependencies(ZKIRDialectHeaders ZKIRUtilIncGen)
