include_directories(${R1CS_INCLUDE_DIR} ${CMAKE_BINARY_DIR}/r1cs/include/)
add_library(LLZKTransforms)
add_library(LLZK::LLZKTransforms ALIAS LLZKTransforms)

file(GLOB LLZKTransforms_SOURCES "*.cpp")
target_sources(LLZKTransforms PRIVATE ${LLZKTransforms_SOURCES})
target_link_libraries(
  LLZKTransforms PUBLIC LLZKDialectHeaders ${LLZK_DEP_DIALECT_LIBS} MLIRIR
                     MLIRPass MLIRParser MLIRTransformUtils MLIRSCFTransforms
                     LLVMHeaders MLIRHeaders
                     LLZKDialect
                     R1CSDialect)
add_dependencies(LLZKTransforms LLZKDialect LLZKDialectHeaders)
llzk_target_add_mlir_link_settings(LLZKTransforms)

install(TARGETS LLZKTransforms EXPORT LLZKTargets)
