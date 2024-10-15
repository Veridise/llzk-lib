#include "Dialect/ZKIR/IR/Types.h"
#include "Dialect/ZKIR/IR/Ops.h"
#include "Dialect/ZKIR/Util/SymbolHelper.h"

namespace zkir {

// valid types: I1, Index, ZKIR_FeltType, ZKIR_StructType, ZKIR_ArrayType
bool isValidZkirType(mlir::Type type) {
  return type.isSignlessInteger(1) || llvm::isa<::mlir::IndexType>(type) ||
         llvm::isa<zkir::FeltType>(type) || llvm::isa<zkir::StructType>(type) ||
         (llvm::isa<zkir::ArrayType>(type) &&
          isValidZkirType(llvm::cast<::zkir::ArrayType>(type).getElementType()));
}

// valid types: I1, Index, ZKIR_FeltType, ZKIR_ArrayType
bool isValidEmitEqType(mlir::Type type) {
  return type.isSignlessInteger(1) || llvm::isa<::mlir::IndexType>(type) ||
         llvm::isa<zkir::FeltType>(type) ||
         (llvm::isa<zkir::ArrayType>(type) &&
          isValidEmitEqType(llvm::cast<::zkir::ArrayType>(type).getElementType()));
}

StructDefOp
StructType::getDefinition(mlir::SymbolTableCollection &symbolTable, mlir::Operation *op) {
  return lookupTopLevelSymbol<StructDefOp>(symbolTable, op, getName());
}

mlir::LogicalResult
StructType::verifySymbol(mlir::SymbolTableCollection &symbolTable, mlir::Operation *op) {
  if (!getDefinition(symbolTable, op)) {
    return op->emitError() << "struct type references undefined component: " << *this;
  } else {
    return mlir::success();
  }
}

mlir::LogicalResult ArrayType::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError, mlir::Type elementType,
    uint64_t numElements
) {
  return checkValidZkirType(emitError, elementType);
}

} // namespace zkir
