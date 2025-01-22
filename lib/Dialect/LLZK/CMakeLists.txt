add_library(LLZKDialect)
add_library(LLZK::LLZKDialect ALIAS LLZKDialect)
target_link_libraries(LLZKAllDialects INTERFACE LLZK::LLZKDialect)

set(CMAKE_CXX_VISIBILITY_PRESET "hidden")

file(GLOB_RECURSE LLZKDialect_SOURCES "**/*.cpp")
target_sources(LLZKDialect PRIVATE ${LLZKDialect_SOURCES})
target_link_libraries(
  LLZKDialect PUBLIC LLZKDialectHeaders ${LLZK_DEP_DIALECT_LIBS} MLIRIR
                     MLIRPass MLIRParser)
target_link_libraries(LLZKDialect PRIVATE LLVMHeaders MLIRHeaders)
add_dependencies(LLZKDialect LLZKDialectHeaders)
llzk_target_add_mlir_link_settings(LLZKDialect)

install(TARGETS LLZKDialect EXPORT LLZKTargets)
