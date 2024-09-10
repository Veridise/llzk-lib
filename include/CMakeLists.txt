add_library(ZKIRDialectHeaders INTERFACE)
target_include_directories(ZKIRDialectHeaders SYSTEM INTERFACE
  # headers in this folder's subdirectories
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  # TableGen'd files
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>

  # We assume ZKIR is installed as-is to include top level
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

install(TARGETS ZKIRDialectHeaders EXPORT ZKIRTargets)
install(DIRECTORY
  "${CMAKE_CURRENT_SOURCE_DIR}/"
  "${CMAKE_CURRENT_BINARY_DIR}/"
  TYPE INCLUDE
  FILES_MATCHING
  PATTERN "*.h"
  PATTERN "*.h.inc"
  PATTERN "*.td"
  PATTERN CMakeFiles EXCLUDE
)

# subdirectories will add their IncGen targets as deps to the header folder
add_subdirectory(Dialect/ZKIR/IR)
add_subdirectory(Dialect/ZKIR/Transforms)
