add_library(LLZKTransforms)
add_library(LLZK::LLZKTransforms ALIAS LLZKTransforms)

file(GLOB LLZKTransforms_SOURCES "*.cpp")
target_sources(LLZKTransforms PRIVATE ${LLZKTransforms_SOURCES})
target_link_libraries(
   LLZKTransforms PUBLIC LLZKDialectHeaders ${LLZK_DEP_DIALECT_LIBS} MLIRIR
                     MLIRPass MLIRParser MLIRTransformUtils MLIRSCFTransforms
                     LLVMHeaders MLIRHeaders
                  PRIVATE
                     LLZKDialect
                     R1CSDialect R1CSDialectHeaders)
add_dependencies(LLZKTransforms
   LLZKDialect LLZKDialectHeaders
   R1CSDialect R1CSDialectHeaders)

llzk_target_add_mlir_link_settings(LLZKTransforms)

install(TARGETS LLZKTransforms EXPORT LLZKTargets)
