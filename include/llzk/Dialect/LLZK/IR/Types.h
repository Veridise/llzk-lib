#pragma once

#include "llzk/Dialect/LLZK/IR/Attrs.h"
#include "llzk/Dialect/LLZK/IR/Dialect.h"
#include "llzk/Dialect/LLZK/IR/Enums.h"
#include "llzk/Dialect/LLZK/Util/SymbolLookupResult.h" // IWYU pragma: keep

#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/IR/Types.h>
#include <mlir/Interfaces/DataLayoutInterfaces.h>
#include <mlir/Interfaces/MemorySlotInterfaces.h>

#include <llvm/ADT/TypeSwitch.h>

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

} // namespace llzk
