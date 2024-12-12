#pragma once

#include "llzk/Dialect/LLZK/IR/Dialect.h"
#include "llzk/Dialect/LLZK/Util/SymbolLookupResult.h" // IWYU pragma: keep

#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/IR/Types.h>

#include <llvm/ADT/STLFunctionalExtras.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/TypeSwitch.h>

#include <vector>

// forward-declare ops
#define GET_OP_FWD_DEFINES
#include "llzk/Dialect/LLZK/IR/Ops.h.inc"

// Include TableGen'd declarations
#define GET_TYPEDEF_CLASSES
#include "llzk/Dialect/LLZK/IR/Types.h.inc"

namespace llzk {

// valid types: I1, Index, LLZK_FeltType, LLZK_ArrayType
bool isValidEmitEqType(mlir::Type type);

// valid types: I1, Index, LLZK_FeltType, LLZK_StructType, LLZK_ArrayType
bool isValidType(mlir::Type type);

inline mlir::LogicalResult
checkValidType(llvm::function_ref<mlir::InFlightDiagnostic()> emitError, mlir::Type type) {
  if (!isValidType(type)) {
    return emitError() << "expected a valid LLZK type but found " << type;
  } else {
    return mlir::success();
  }
}

/// Return `true` iff the two StructType instances are equivalent or could be equivalent after full
/// instantiation of struct parameters.
bool structTypesUnify(
    const StructType &lhs, const StructType &rhs, std::vector<llvm::StringRef> rhsRevPrefix = {}
);

/// Return `true` iff the two Type instances are equivalent or could be equivalent after full
/// instantiation of struct parameters (if applicable within the given types).
bool typesUnify(
    const mlir::Type &lhs, const mlir::Type &rhs, std::vector<llvm::StringRef> rhsRevPrefix = {}
);

} // namespace llzk
