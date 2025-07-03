include_directories(
  ${R1CS_INCLUDE_DIR}
  ${CMAKE_BINARY_DIR}/r1cs/include
)

add_mlir_dialect_library(R1CSDialect
  Dialect.cpp
  Types.cpp
  Ops.cpp
  Attrs.cpp
  Registration.cpp

  LINK_LIBS PRIVATE
  MLIRIR
  MLIRSupport
  LLVMHeaders
  MLIRHeaders

  DEPENDS
  R1CSDialectIncGen
)
llzk_target_add_mlir_link_settings(R1CSDialect)

install(TARGETS R1CSDialect EXPORT LLZKTargets)
