add_library(LLZKDialectShared)
# add_library(LLZK::LLZKDialectShared ALIAS LLZKDialectShared)
target_link_libraries(LLZKAllDialects INTERFACE LLZKDialectShared)

file(GLOB_RECURSE LLZKDialectShared_SOURCES "*.cpp")
target_sources(LLZKDialectShared PRIVATE ${LLZKDialectShared_SOURCES})
target_link_libraries(
  LLZKDialectShared PUBLIC LLZKDialectHeaders ${LLZK_DEP_DIALECT_LIBS} MLIRIR
                     MLIRParser LLVMHeaders MLIRHeaders LLZKUtil)
add_dependencies(LLZKDialectShared LLZKDialectHeaders)
llzk_target_add_mlir_link_settings(LLZKDialectShared)

install(TARGETS LLZKDialectShared EXPORT LLZKTargets)
