cmake_minimum_required(VERSION 3.18)

project(ZKIR
  DESCRIPTION "Veridise's intermediate representation for zero knowledge languages"
  VERSION 0.1.0
  HOMEPAGE_URL https://github.com/Veridise/zkir-lib
)

include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 20 CACHE STRING "C++ standard")
set(CMAKE_CXX_STANDARD_REQUIRED ON CACHE STRING "")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

# Use same policy as LLVM to suppress warnings
if(POLICY CMP0116)
  cmake_policy(SET CMP0116 OLD)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE INTERNAL "")

### Dependency setup

find_package(LLVM 18.1 REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using LLVM in: ${LLVM_DIR}")
message(STATUS "Using MLIR in: ${MLIR_DIR}")

# LLVM & MLIR do not propagate their include dirs correctly
include_directories(${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})

list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
add_definitions(${LLVM_DEFINITIONS_LIST})
add_compile_options(-Werror=switch -Wall -Wextra -Wno-unused-parameter -Wshadow -Wnon-virtual-dtor -pedantic)

if(CMAKE_CONFIGURATION_TYPES)
  list(APPEND CMAKE_CONFIGURATION_TYPES DebugWithASAN)
  list(REMOVE_DUPLICATES CMAKE_CONFIGURATION_TYPES)
endif()

function(enable_coverage _project_name)
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES ".*Clang")
    target_compile_options(${_project_name} INTERFACE --coverage -O0 -g)
    target_link_libraries(${_project_name} INTERFACE --coverage)
  endif()
endfunction()

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  enable_coverage(ZKIRAllDialects)
endif()

if (CMAKE_BUILD_TYPE STREQUAL "DebugWithASAN")
  enable_coverage(ZKIRAllDialects)
  add_compile_options(-fsanitize=address)
  add_link_options(-fsanitize=address)
endif()

# MAYBE FIXME: If we want to hide inline functions we need to setup macros
# See https://github.com/Veridise/zkir-lib/pull/3#discussion_r1726140891

### Python bindings

include(CMakeDependentOption)
cmake_dependent_option(ZKIR_ENABLE_BINDINGS_PYTHON
  "Enable Python bindings"
  ON
  MLIR_ENABLE_BINDINGS_PYTHON
  OFF
)

message(CHECK_START "Checking ZKIR Python bindings")
list(APPEND CMAKE_MESSAGE_INDENT "  ")
message(CHECK_START "Checking MLIR Python bindings")
if(MLIR_ENABLE_BINDINGS_PYTHON AND ZKIR_ENABLE_BINDINGS_PYTHON)
  include(MLIRDetectPythonEnv)
  include(AddMLIRPython)
  mlir_configure_python_dev_packages()
  message(CHECK_PASS "found")

  list(POP_BACK CMAKE_MESSAGE_INDENT)
  message(CHECK_PASS "enabling ZKIR Python bindings")
elseif(MLIR_ENABLE_BINDINGS_PYTHON)
  message(CHECK_PASS "found")
  list(POP_BACK CMAKE_MESSAGE_INDENT)
  message(CHECK_FAIL "ZKIR_ENABLE_BINDINGS_PYTHON is set to off")
else()
  message(CHECK_FAIL "not found")  # mlir bindings
  list(POP_BACK CMAKE_MESSAGE_INDENT)
  message(CHECK_FAIL "missing dependencies")  # overall status
endif()

### Project files

set(DOXYGEN_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/doc")
set(ZKIR_MLIR_DOC_OUTPUT_DIR "${DOXYGEN_OUTPUT_DIRECTORY}/mlir")

# We need the include dir so we can pass it to mlir-tblgen
set(ZKIR_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/ZKIRMacros.cmake)

# Create an empty list for the dialect libraries, to be populated in the lib
# subdirectory.
add_library(ZKIRAllDialects INTERFACE)
add_library(ZKIR::AllDialects ALIAS ZKIRAllDialects)

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)

if(ZKIR_ENABLE_BINDINGS_PYTHON)
  add_subdirectory(python)
endif()

### Documentation
add_custom_target(doc)
find_package(Doxygen OPTIONAL_COMPONENTS dot)
if (Doxygen_FOUND)
  message(STATUS "Doxygen found, enabling documentation...")
  set(DOXYGEN_EXTRACT_ALL YES)
  set(DOXYGEN_INCLUDE_PATH "${CMAKE_CURRENT_BINARY_DIR}/include/")
  set(DOXYGEN_EXCLUDE_PATTERNS ${CMAKE_CURRENT_BINARY_DIR}/include/*/*.md)
  set(DOXYGEN_USE_MDFILE_AS_MAINPAGE ${CMAKE_CURRENT_SOURCE_DIR}/doc/doxygen/index.md)
  set(DOXYGEN_FILE_PATTERNS *.cpp *.cpp.inc *.h.inc *.hpp *.h *.td *.md *.py)
  set(DOXYGEN_EXTENSION_MAPPING inc=C++)
  set(DOXYGEN_MACRO_EXPANSION YES)
  set(DOXYGEN_EXPAND_ONLY_PREDEF YES)
  set(DOXYGEN_PREDEFINED GET_OP_CLASSES GET_TYPEDEF_CLASSES GET_ATTR_CLASSES)
  set(DOXYGEN_SOURCE_BROWSER YES)
  set(DOXYGEN_JAVADOC_AUTOBRIEF YES)
  doxygen_add_docs(doxygen
    "${CMAKE_CURRENT_SOURCE_DIR}/lib/"
    "${CMAKE_CURRENT_SOURCE_DIR}/tools/"
    "${CMAKE_CURRENT_BINARY_DIR}/include/"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/"
    "${ZKIR_MLIR_DOC_OUTPUT_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/doc/doxygen/"
  )
  add_dependencies(doxygen ZKIRDialectHeaders mlir-doc)
  add_dependencies(doc doxygen)
endif()

### Tests

# lit tests
include(CTest)
enable_testing()
set(CMAKE_CTEST_ARGUMENTS "--output-on-failure" CACHE STRING "CTest arguments")
if(BUILD_TESTING)
  add_subdirectory(test)
endif()

# find_package(GTest)
# if(BUILD_TESTING AND GTest_FOUND)
#   message(STATUS "gtest found, enabling unit tests...")
#   include(GoogleTest)
#
#   if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.20")
#     set(GTEST_LIB_TARGETS "GTest::gtest;GTest::gmock")
#     set(GTEST_EXE_TARGET GTest::gtest_main)
#   else()
#     set(GTEST_LIB_TARGETS GTest::GTest)
#     set(GTEST_EXE_TARGET GTest::Main)
#   endif()
#
#   add_subdirectory(unittests)
# elseif(BUILD_TESTING)
#   message(STATUS "gtest not found, unit tests will not be run")
# else()
#   message(STATUS "Unit tests are disabled")
# endif()

# Catch-all target for running all tests
add_custom_target(check
  DEPENDS
  # test targets may not exist if BUILD_TESTING is off
  $<$<BOOL:BUILD_TESTING>:check-lit>
)

### Install

export(EXPORT ZKIRTargets
  FILE ${CMAKE_CURRENT_BINARY_DIR}/ZKIRTargets.cmake
  NAMESPACE ZKIR::
)
install(EXPORT ZKIRTargets
  NAMESPACE ZKIR::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ZKIR)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/ZKIRConfigVersion.cmake"
  VERSION "${PROJECT_VERSION}"
  COMPATIBILITY AnyNewerVersion
)
configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/ZKIRConfig.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/ZKIRConfig.cmake"
  INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/ZKIR"
)
install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/ZKIRConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/ZKIRConfigVersion.cmake"
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/ZKIR"
)



# TODO: I added this stuff below
# mlir_tablegen(Dialect.h.inc -gen-dialect-decls -dialect=zkir)
# mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs -dialect=zkir)
# mlir_tablegen(Ops.h.inc -gen-op-decls)
# mlir_tablegen(Ops.cpp.inc -gen-op-defs)
