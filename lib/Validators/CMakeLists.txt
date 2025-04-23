add_library(LLZKValidators)
add_library(LLZK::LLZKValidators ALIAS LLZKValidators)

file(GLOB LLZKValidators_SOURCES "*.cpp")
target_sources(LLZKValidators PRIVATE ${LLZKValidators_SOURCES})
target_link_libraries(
  LLZKValidators PUBLIC LLZKDialectHeaders ${LLZK_DEP_DIALECT_LIBS} MLIRIR
                     MLIRPass MLIRParser MLIRTransformUtils MLIRSCFTransforms
                     LLVMHeaders MLIRHeaders)
add_dependencies(LLZKValidators LLZKDialect LLZKDialectHeaders)
llzk_target_add_mlir_link_settings(LLZKValidators)

install(TARGETS LLZKValidators EXPORT LLZKTargets)
