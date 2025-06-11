include_directories(
  ${LLVM_INCLUDE_DIRS}
  ${MLIR_INCLUDE_DIRS}
  ${R1CS_INCLUDE_DIR}
  ${CMAKE_BINARY_DIR}/r1cs/include
)
add_mlir_dialect_library(R1CSDialect
  Dialect.cpp
  Types.cpp
  Ops.cpp
  Attrs.cpp
  Registration.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport

  DEPENDS
  R1CSDialectIncGen
)

add_dependencies(LLZKAllDialects R1CSDialect)