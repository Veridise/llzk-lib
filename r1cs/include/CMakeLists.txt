add_library(R1CSDialectHeaders INTERFACE)
target_include_directories(
  R1CSDialectHeaders SYSTEM
  INTERFACE # headers in this folder's subdirectories
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
            # TableGen'd files
            $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
            # We assume LLZK is installed as-is to include top level
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

install(TARGETS R1CSDialectHeaders EXPORT LLZKTargets)

install(
  DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/" "${CMAKE_CURRENT_BINARY_DIR}/"
  TYPE INCLUDE
  FILES_MATCHING
  PATTERN "*.h"
  PATTERN "*.h.inc"
  PATTERN "*.td"
  PATTERN CMakeFiles EXCLUDE)

add_subdirectory(r1cs/Dialect/IR)
