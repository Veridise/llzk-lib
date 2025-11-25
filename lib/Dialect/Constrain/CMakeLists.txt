add_library(LLZKConstrainDialect)
add_library(LLZK::LLZKConstrainDialect ALIAS LLZKConstrainDialect)
target_link_libraries(LLZKAllDialects INTERFACE LLZKConstrainDialect)

file(GLOB_RECURSE LLZKConstrainDialect_SOURCES "**/*.cpp")
target_sources(LLZKConstrainDialect PRIVATE ${LLZKConstrainDialect_SOURCES})
target_link_libraries(
  LLZKConstrainDialect PUBLIC LLZKDialectHeaders ${LLZK_DEP_DIALECT_LIBS} MLIRIR
                     MLIRPass MLIRParser MLIRTransformUtils
                     LLVMHeaders MLIRHeaders)
llzk_target_add_mlir_link_settings(LLZKConstrainDialect)

install(TARGETS LLZKConstrainDialect EXPORT LLZKTargets)
