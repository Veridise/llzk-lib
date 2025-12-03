add_library(LLZKCastDialect)
add_library(LLZK::LLZKCastDialect ALIAS LLZKCastDialect)
target_link_libraries(LLZKAllDialects INTERFACE LLZKCastDialect)

file(GLOB_RECURSE LLZKCastDialect_SOURCES "**/*.cpp")
target_sources(LLZKCastDialect PRIVATE ${LLZKCastDialect_SOURCES})
target_link_libraries(
  LLZKCastDialect PUBLIC LLZKDialectHeaders ${LLZK_DEP_DIALECT_LIBS} MLIRIR
                     MLIRPass MLIRParser MLIRTransformUtils
                     LLVMHeaders MLIRHeaders)
llzk_target_add_mlir_link_settings(LLZKCastDialect)

install(TARGETS LLZKCastDialect EXPORT LLZKTargets)
