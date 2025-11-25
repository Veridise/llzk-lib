add_library(LLZKUndefDialect)
add_library(LLZK::LLZKUndefDialect ALIAS LLZKUndefDialect)
target_link_libraries(LLZKAllDialects INTERFACE LLZKUndefDialect)

file(GLOB_RECURSE LLZKUndefDialect_SOURCES "**/*.cpp")
target_sources(LLZKUndefDialect PRIVATE ${LLZKUndefDialect_SOURCES})
target_link_libraries(
  LLZKUndefDialect PUBLIC LLZKDialectHeaders ${LLZK_DEP_DIALECT_LIBS} MLIRIR
                     MLIRParser LLVMHeaders MLIRHeaders)
llzk_target_add_mlir_link_settings(LLZKUndefDialect)

install(TARGETS LLZKUndefDialect EXPORT LLZKTargets)
