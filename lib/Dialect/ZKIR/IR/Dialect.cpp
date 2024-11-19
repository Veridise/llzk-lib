#include <llvm/ADT/Hashing.h>
#include <mlir/IR/DialectImplementation.h>

#include "zkir/Dialect/ZKIR/IR/Attrs.h"
#include "zkir/Dialect/ZKIR/IR/Dialect.h"
#include "zkir/Dialect/ZKIR/IR/Ops.h"
#include "zkir/Dialect/ZKIR/IR/Types.h"

// TableGen'd implementation files
#include "zkir/Dialect/ZKIR/IR/Dialect.cpp.inc"

// Need a complete declaration of storage classes
#define GET_TYPEDEF_CLASSES
#include "zkir/Dialect/ZKIR/IR/Types.cpp.inc"
#define GET_ATTRDEF_CLASSES
#include "zkir/Dialect/ZKIR/IR/Attrs.cpp.inc"

//===------------------------------------------------------------------===//
// ZKIRDialect
//===------------------------------------------------------------------===//

auto zkir::ZKIRDialect::initialize() -> void {
  // clang-format off
  addOperations<
    #define GET_OP_LIST
    #include "zkir/Dialect/ZKIR/IR/Ops.cpp.inc"
  >();

  addTypes<
    #define GET_TYPEDEF_LIST
    #include "zkir/Dialect/ZKIR/IR/Types.cpp.inc"
  >();

  addAttributes<
    #define GET_ATTRDEF_LIST
    #include "zkir/Dialect/ZKIR/IR/Attrs.cpp.inc"
  >();
  // clang-format on
}
