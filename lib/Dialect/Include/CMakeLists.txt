add_library(LLZKIncludeDialect)
add_library(LLZK::LLZKIncludeDialect ALIAS LLZKIncludeDialect)
target_link_libraries(LLZKAllDialects INTERFACE LLZKIncludeDialect)

file(GLOB_RECURSE LLZKIncludeDialect_SOURCES "**/*.cpp")
target_sources(LLZKIncludeDialect PRIVATE ${LLZKIncludeDialect_SOURCES})
target_link_libraries(
  LLZKIncludeDialect PUBLIC LLZKDialectHeaders ${LLZK_DEP_DIALECT_LIBS} MLIRIR
                     MLIRPass MLIRParser MLIRTransformUtils MLIRSCFTransforms
                     LLVMHeaders MLIRHeaders)
llzk_target_add_mlir_link_settings(LLZKIncludeDialect)

install(TARGETS LLZKIncludeDialect EXPORT LLZKTargets)
