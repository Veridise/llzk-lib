#pragma once

#include "zkir/Dialect/ZKIR/IR/Dialect.h"
#include "zkir/Dialect/ZKIR/IR/Enums.h"
#include "zkir/Dialect/ZKIR/IR/Types.h"

#include <mlir/IR/DialectImplementation.h>

// Include TableGen'd declarations
#define GET_ATTRDEF_CLASSES
#include "zkir/Dialect/ZKIR/IR/Attrs.h.inc"

namespace zkir {

/// Custom directive for reading a field element constant.
mlir::ParseResult parseAPInt(mlir::AsmParser &parser, llvm::APInt &value);

/// Custom directive for printing a field element constant.
void printAPInt(mlir::AsmPrinter &printer, const llvm::APInt &value);

} // namespace zkir
