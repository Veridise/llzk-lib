set(LLZK_DEP_DIALECT_LIBS MLIRFuncDialect MLIRSCFDialect)
set(LLZK_DEP_DIALECT_LIBS
    ${LLZK_DEP_DIALECT_LIBS}
    PARENT_SCOPE)
add_subdirectory(Dialect/LLZK)
add_subdirectory(Target)

if(LLZK_ENABLE_BINDINGS_PYTHON)
  add_subdirectory(Bindings/Python)
endif()

add_library(LLZKDialectRegistration)
add_library(LLZK::DialectRegistration ALIAS LLZKDialectRegistration)
# Ensures that the target can be imported as "LLZK::DialectRegistration" and not
# "LLZK::LLZKDialectRegistration"
set_target_properties(LLZKDialectRegistration PROPERTIES EXPORT_NAME
                                                         "DialectRegistration")
target_sources(LLZKDialectRegistration PRIVATE InitDialects.cpp)
target_link_libraries(
  LLZKDialectRegistration PUBLIC LLZKAllDialects ${LLZK_DEP_DIALECT_LIBS}
                                 LLZKDialectHeaders)
add_dependencies(LLZKDialectRegistration LLZKDialectHeaders)
llzk_target_add_mlir_link_settings(LLZKDialectRegistration)

install(TARGETS LLZKDialectRegistration EXPORT LLZKTargets)

