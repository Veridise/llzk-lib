#pragma once

#include "zkir/Dialect/ZKIR/IR/Attrs.h"
#include "zkir/Dialect/ZKIR/IR/Dialect.h"
#include "zkir/Dialect/ZKIR/IR/Enums.h"

#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/IR/Types.h>
#include <mlir/Interfaces/DataLayoutInterfaces.h>
#include <mlir/Interfaces/MemorySlotInterfaces.h>

#include <llvm/ADT/TypeSwitch.h>

// forward-declare ops
#define GET_OP_FWD_DEFINES
#include "zkir/Dialect/ZKIR/IR/Ops.h.inc"

// Include TableGen'd declarations
#define GET_TYPEDEF_CLASSES
#include "zkir/Dialect/ZKIR/IR/Types.h.inc"

namespace zkir {

// valid types: I1, Index, ZKIR_FeltType, ZKIR_ArrayType
bool isValidEmitEqType(mlir::Type type);

// valid types: I1, Index, ZKIR_FeltType, ZKIR_StructType, ZKIR_ArrayType
bool isValidZkirType(mlir::Type type);

inline mlir::LogicalResult
checkValidZkirType(llvm::function_ref<mlir::InFlightDiagnostic()> emitError, mlir::Type type) {
  if (!isValidZkirType(type)) {
    return emitError() << "expected "
                       << "a valid ZKIR type"
                       << " but found " << type;
  } else {
    return mlir::success();
  }
}

} // namespace zkir
