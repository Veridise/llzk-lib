set(ZKIR_DEP_DIALECT_LIBS
  MLIRFuncDialect
  MLIRIndexDialect
  MLIRSCFDialect
  PARENT_SCOPE
)
add_subdirectory(Dialect/ZKIR)

if(ZKIR_ENABLE_BINDINGS_PYTHON)
  add_subdirectory(Bindings/Python)
endif()

add_library(ZKIRDialectRegistration)
add_library(ZKIR::DialectRegistration ALIAS ZKIRDialectRegistration)
target_sources(ZKIRDialectRegistration PRIVATE InitDialects.cpp)
target_link_libraries(ZKIRDialectRegistration
  PUBLIC ZKIRAllDialects ${ZKIR_DEP_DIALECT_LIBS} ZKIRDialectHeaders)
add_dependencies(ZKIRDialectRegistration ZKIRDialectHeaders)
zkir_target_add_mlir_link_settings(ZKIRDialectRegistration)

install(TARGETS ZKIRDialectRegistration
  EXPORT ZKIRTargets
)