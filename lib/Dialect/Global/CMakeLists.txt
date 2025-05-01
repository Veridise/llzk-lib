add_library(LLZKGlobalDialect)
add_library(LLZK::LLZKGlobalDialect ALIAS LLZKGlobalDialect)
target_link_libraries(LLZKAllDialects INTERFACE LLZKGlobalDialect)

file(GLOB_RECURSE LLZKGlobalDialect_SOURCES "**/*.cpp")
target_sources(LLZKGlobalDialect PRIVATE ${LLZKGlobalDialect_SOURCES})
target_link_libraries(
  LLZKGlobalDialect PUBLIC LLZKDialectHeaders ${LLZK_DEP_DIALECT_LIBS} MLIRIR
                     MLIRParser LLVMHeaders MLIRHeaders)
add_dependencies(LLZKGlobalDialect LLZKDialectHeaders)
llzk_target_add_mlir_link_settings(LLZKGlobalDialect)

install(TARGETS LLZKGlobalDialect EXPORT LLZKTargets)
