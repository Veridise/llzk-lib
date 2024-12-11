include_directories(${MLIR_INCLUDE_DIRS} ${LLZK_INCLUDE_DIR})

set(LLVM_TARGET_DEFINITIONS "CallGraphPasses.td")
mlir_tablegen(CallGraphPasses.h.inc -gen-pass-decls -name=Analysis)
llzk_add_mlir_doc(CallGraphPassesDocGen CallGraphPasses.md -gen-pass-doc)

add_public_tablegen_target(LLZKAnalysisIncGen)
add_dependencies(mlir-headers LLZKAnalysisIncGen)
add_dependencies(LLZKDialectHeaders LLZKAnalysisIncGen)
