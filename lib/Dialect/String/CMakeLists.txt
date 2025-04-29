add_library(LLZKStringDialect)
add_library(LLZK::LLZKStringDialect ALIAS LLZKStringDialect)
target_link_libraries(LLZKAllDialects INTERFACE LLZKStringDialect)

file(GLOB_RECURSE LLZKStringDialect_SOURCES "**/*.cpp")
target_sources(LLZKStringDialect PRIVATE ${LLZKStringDialect_SOURCES})
target_link_libraries(
  LLZKStringDialect PUBLIC LLZKDialectHeaders ${LLZK_DEP_DIALECT_LIBS} MLIRIR
                     MLIRPass MLIRParser MLIRTransformUtils MLIRSCFTransforms
                     LLVMHeaders MLIRHeaders)
add_dependencies(LLZKStringDialect LLZKDialectHeaders)
llzk_target_add_mlir_link_settings(LLZKStringDialect)

install(TARGETS LLZKStringDialect EXPORT LLZKTargets)
