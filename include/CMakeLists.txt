add_library(LLZKDialectHeaders INTERFACE)
target_include_directories(
  LLZKDialectHeaders SYSTEM
  INTERFACE # headers in this folder's subdirectories
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
            # TableGen'd files
            $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
            # We assume LLZK is installed as-is to include top level
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

install(TARGETS LLZKDialectHeaders EXPORT LLZKTargets)
install(
  DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/" "${CMAKE_CURRENT_BINARY_DIR}/"
  TYPE INCLUDE
  FILES_MATCHING
  PATTERN "*.h"
  PATTERN "*.h.inc"
  PATTERN "*.td"
  PATTERN CMakeFiles EXCLUDE)

# subdirectories will add their IncGen targets as deps to the header folder
add_subdirectory(llzk/Config)
add_subdirectory(llzk/Dialect/LLZK/Analysis)
add_subdirectory(llzk/Dialect/LLZK/IR)
add_subdirectory(llzk/Dialect/LLZK/Transforms)
add_subdirectory(llzk/Dialect/LLZK/Validators)
