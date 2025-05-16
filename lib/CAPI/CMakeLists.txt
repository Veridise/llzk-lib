add_library(LLZKCAPI)
add_library(LLZK::LLZKCAPI ALIAS LLZKCAPI)
target_link_libraries(LLZKAllDialects INTERFACE LLZKCAPI)

file(GLOB_RECURSE LLZKCAPI_SOURCES "**/*.cpp")
target_sources(LLZKCAPI PRIVATE ${LLZKCAPI_SOURCES})
target_link_libraries(
  LLZKCAPI PUBLIC LLZKDialectHeaders ${LLZK_DEP_DIALECT_LIBS} MLIRIR
                     MLIRParser LLVMHeaders MLIRHeaders)
add_dependencies(LLZKCAPI LLZKDialectHeaders)
llzk_target_add_mlir_link_settings(LLZKCAPI)

install(TARGETS LLZKCAPI EXPORT LLZKTargets)
