#include "llzk/Dialect/LLZK/IR/Attrs.h"
#include "llzk/Dialect/LLZK/IR/Dialect.h"
#include "llzk/Dialect/LLZK/IR/Ops.h"
#include "llzk/Dialect/LLZK/IR/Types.h"

#include <mlir/IR/DialectImplementation.h>

// TableGen'd implementation files
#include "llzk/Dialect/LLZK/IR/Dialect.cpp.inc"

template <> struct mlir::FieldParser<llvm::APInt> {
  static mlir::FailureOr<llvm::APInt> parse(mlir::AsmParser &parser) {
    auto loc = parser.getCurrentLocation();
    llvm::APInt val;
    auto result = parser.parseOptionalInteger(val);
    if (!result.has_value() || *result) {
      return parser.emitError(loc, "expected integer value");
    } else {
      return val;
    }
  }
};

// Need a complete declaration of storage classes for below
#define GET_TYPEDEF_CLASSES
#include "llzk/Dialect/LLZK/IR/Types.cpp.inc"
#define GET_ATTRDEF_CLASSES
#include "llzk/Dialect/LLZK/IR/Attrs.cpp.inc"

//===------------------------------------------------------------------===//
// LLZKDialect
//===------------------------------------------------------------------===//

auto llzk::LLZKDialect::initialize() -> void {
  // clang-format off
  addOperations<
    #define GET_OP_LIST
    #include "llzk/Dialect/LLZK/IR/Ops.cpp.inc"
  >();

  addTypes<
    #define GET_TYPEDEF_LIST
    #include "llzk/Dialect/LLZK/IR/Types.cpp.inc"
  >();

  addAttributes<
    #define GET_ATTRDEF_LIST
    #include "llzk/Dialect/LLZK/IR/Attrs.cpp.inc"
  >();
  // clang-format on
}
