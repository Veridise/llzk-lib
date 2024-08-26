# llvm-lit is named lit in external builds
find_program(LLVM_EXTERNAL_LIT NAMES lit llvm-lit ${LLVM_EXTERNAL_LIT} REQUIRED)
message(STATUS "Using lit: ${LLVM_EXTERNAL_LIT}")

set(ZKIR_TEST_DEPENDS
  zkir-dump
#   zkir-driver
#   zkir-query
)

# if(ZKIR_BUILD_DEVTOOLS)
#   set(zkir_lit_extra_suffixes "\".querytxt\"")
#   list(APPEND ZKIR_TEST_DEPENDS zkir-query zkir-driver)
# endif()

set(ZKIR_ENABLE_BINDINGS_PYTHON_STRING "False")
if(ZKIR_ENABLE_BINDINGS_PYTHON)
    set(ZKIR_ENABLE_BINDINGS_PYTHON_STRING "True")
endif()

set(ZKIR_LIT_COMPATIBLE_COVERAGE_STRING "False")
if(ZKIR_LIT_COMPATIBLE_COVERAGE)
  message(STATUS "Lit will generate coverage information for every test")
  set(ZKIR_LIT_COMPATIBLE_COVERAGE_STRING "True")
endif()

set(LIT_ARGS "--xunit-xml-output report.xml")

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

add_lit_testsuite(check-lit "Running the lit-based regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${ZKIR_TEST_DEPENDS}
  ARGS ${LIT_ARGS}
)
