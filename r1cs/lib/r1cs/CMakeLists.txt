include_directories(
  ${LLVM_INCLUDE_DIRS}
  ${MLIR_INCLUDE_DIRS}
  ${R1CS_INCLUDE_DIR}
)

add_library(R1CSDialect)
add_library(LLZK::R1CSDialect ALIAS R1CSDialect)

file(GLOB_RECURSE R1CSDialect_SOURCES "**/*.cpp")
target_sources(R1CSDialect PRIVATE ${R1CSDialect_SOURCES})
target_link_libraries(R1CSDialect PUBLIC LLVMHeaders MLIRHeaders MLIRIR
                      ${LLZK_DEP_DIALECT_LIBS} LLZKDialectHeaders R1CSDialectHeaders)
add_dependencies(R1CSDialect LLZKDialectHeaders R1CSDialectHeaders)
llzk_target_add_mlir_link_settings(R1CSDialect)

install(TARGETS R1CSDialect EXPORT LLZKTargets)
