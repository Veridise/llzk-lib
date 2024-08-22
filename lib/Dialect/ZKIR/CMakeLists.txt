add_library(ZKIRDialect)
add_library(ZKIR::ZKIRDialect ALIAS ZKIRDialect)
target_link_libraries(ZKIRAllDialects INTERFACE ZKIR::ZKIRDialect)

file(GLOB_RECURSE ZKIRDialect_SOURCES "**/*.cpp")
target_sources(ZKIRDialect PRIVATE ${ZKIRDialect_SOURCES})
target_link_libraries(ZKIRDialect
  PUBLIC ZKIRDialectHeaders Constraint MLIRIR MLIRPass LLVMHeaders MLIRHeaders
)
add_dependencies(ZKIRDialect ZKIRDialectHeaders)
zkir_target_add_mlir_link_settings(ZKIRDialect)

install(TARGETS ZKIRDialect
  EXPORT ZKIRTargets
)
