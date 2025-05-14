set(FEATURE_TESTS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/cmake/feature_tests")

function(llzk_check_cxx_feature target feature)
  try_compile(COMPILE_SUCCEEDED "${CMAKE_BINARY_DIR}/compile_tests" "${FEATURE_TESTS_DIR}/${feature}.cpp")
  if(NOT COMPILE_SUCCEEDED)
    message(FATAL_ERROR "Required feature '${feature}' required by ${target} is not supported by the compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
  endif()
endfunction()
