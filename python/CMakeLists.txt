#
# Top level targets
#

declare_mlir_python_sources(LLZKPythonDialectSources)
declare_mlir_python_sources(LLZKPythonDialectSources.Dialects ADD_TO_PARENT
                            LLZKPythonDialectSources)

# Include the tablegen files
include_directories("${LLZK_INCLUDE_DIR}")

#
# Dialect bindings
#

# These must go under the mlir module/namespace, as the generated sources use
# relative imports from mlir.

# TODO: need to autogenerate & expose the C extensions or else attributes/types
# won't be supported.

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT
  LLZKPythonDialectSources.Dialects
  ROOT_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/mlir"
  TD_FILE
  dialects/LLZKOps.td
  SOURCES
  dialects/llzk_dialect.py
  DIALECT_NAME
  llzk_dialect)

#
# Python extensions
#

# All other LLZK Python bindings go under the llzk package.
declare_mlir_python_sources(LLZKPythonSources)

declare_mlir_python_sources(
  LLZKPythonSources.Core
  ADD_TO_PARENT
  LLZKPythonSources
  ROOT_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/llzk"
  SOURCES
  __init__.py
  registration.py)

declare_mlir_python_extension(
  LLZKPythonExtension
  ADD_TO_PARENT
  LLZKPythonSources
  MODULE_NAME
  _llzkRegistration
  ROOT_DIR
  "${CMAKE_SOURCE_DIR}/lib/LLZK/Bindings/Python"
  SOURCES
  LLZKRegistration.cpp
  PRIVATE_LINK_LIBS
  LLVMSupport
  LLZKPythonBindings)

#
# Final modules
#

add_mlir_python_modules(
  LLZKPythonDialectModules
  ROOT_PREFIX
  "${CMAKE_CURRENT_BINARY_DIR}/mlir"
  INSTALL_PREFIX
  python/mlir
  DECLARED_SOURCES
  LLZKPythonDialectSources)

# FIXME: MLIR bug?: the MLIRPythonCAPI library is not exported as a CMake
# target. Have to manually find it.
#
# TODO: this only works on macOS for now, need to fix it for Linux.
find_library(MLIRPythonCAPI_DYLIB NAMES MLIRPythonCAPI libMLIRPythonCAPI
                                        REQUIRED)
add_library(MLIRPythonCAPI SHARED IMPORTED)
set_target_properties(MLIRPythonCAPI PROPERTIES IMPORTED_LOCATION
                                                ${MLIRPythonCAPI_DYLIB})

add_mlir_python_modules(
  LLZKPythonExtensionModules
  ROOT_PREFIX
  "${CMAKE_CURRENT_BINARY_DIR}/llzk"
  INSTALL_PREFIX
  python/llzk
  DECLARED_SOURCES
  LLZKPythonSources
  COMMON_CAPI_LINK_LIBS
  MLIRPythonCAPI)
