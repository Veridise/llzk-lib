add_library(LLZKDialect)
add_library(LLZK::LLZKDialect ALIAS LLZKDialect)
target_link_libraries(LLZKAllDialects INTERFACE LLZK::LLZKDialect)

file(GLOB_RECURSE LLZKDialect_SOURCES "**/*.cpp")
target_sources(LLZKDialect PRIVATE ${LLZKDialect_SOURCES})
target_link_libraries(
  LLZKDialect PUBLIC LLZKDialectHeaders ${LLZK_DEP_DIALECT_LIBS} MLIRIR
                     MLIRPass MLIRParser MLIRTransformUtils MLIRSCFTransforms
                     LLVMHeaders MLIRHeaders)
add_dependencies(LLZKDialect LLZKDialectHeaders)
llzk_target_add_mlir_link_settings(LLZKDialect)

install(TARGETS LLZKDialect EXPORT LLZKTargets)
