add_library(LLZKTransforms)
add_library(LLZK::LLZKTransforms ALIAS LLZKTransforms)

# Grab all .cpp, but EXCLUDE the PCL file from the base set.
file(GLOB LLZKTransforms_SOURCES "*.cpp")
list(FILTER LLZKTransforms_SOURCES EXCLUDE REGEX "LLZKPCLLoweringPass\\.cpp$")

target_sources(LLZKTransforms PRIVATE ${LLZKTransforms_SOURCES})

# Add the PCL lowering .cpp only when enabled.
if(LLZK_WITH_PCL)
  target_sources(LLZKTransforms PRIVATE LLZKPCLLoweringPass.cpp)
  target_link_libraries(LLZKTransforms PRIVATE PCL::PCLDialect)
endif()

target_link_libraries(
  LLZKTransforms
  PUBLIC
    LLZKDialectHeaders
    ${LLZK_DEP_DIALECT_LIBS}
    MLIRIR MLIRPass MLIRParser MLIRTransformUtils MLIRSCFTransforms
    LLVMHeaders MLIRHeaders
  PRIVATE
    LLZKDialect
    R1CSDialect
)

add_dependencies(LLZKTransforms LLZKDialect LLZKDialectHeaders)
llzk_target_add_mlir_link_settings(LLZKTransforms)
install(TARGETS LLZKTransforms EXPORT LLZKTargets)

