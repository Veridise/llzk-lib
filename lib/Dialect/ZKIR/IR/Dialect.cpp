#include <mlir/IR/DialectImplementation.h>

#include "Dialect/ZKIR/IR/Attrs.h"
#include "Dialect/ZKIR/IR/Dialect.h"
#include "Dialect/ZKIR/IR/Ops.h"
#include "Dialect/ZKIR/IR/Types.h"

// TableGen'd implementation files
#include "Dialect/ZKIR/IR/Dialect.cpp.inc"

// Need a complete declaration of storage classes
#define GET_TYPEDEF_CLASSES
#include "Dialect/ZKIR/IR/Types.cpp.inc"
#define GET_ATTRDEF_CLASSES
#include "Dialect/ZKIR/IR/Attrs.cpp.inc"

// -----
// ZKIRDialect
// -----

auto zkir::ZKIRDialect::initialize() -> void {
  // clang-format off
  addOperations<
    #define GET_OP_LIST
    #include "Dialect/ZKIR/IR/Ops.cpp.inc"
  >();

  addTypes<
    #define GET_TYPEDEF_LIST
    #include "Dialect/ZKIR/IR/Types.cpp.inc"
  >();

  addAttributes<
    #define GET_ATTRDEF_LIST
    #include "Dialect/ZKIR/IR/Attrs.cpp.inc"
  >();
  // clang-format on
}
