#include "Dialect/ZKIR/IR/Types.h"
#include "Dialect/ZKIR/IR/Ops.h"
#include "Dialect/ZKIR/Util/SymbolHelper.h"

namespace zkir {

StructDefOp StructType::getDefinition(mlir::SymbolTableCollection &symbolTable,
                                      mlir::Operation *op) {
  return lookupTopLevelSymbol<StructDefOp>(symbolTable, op, getName());
}

mlir::LogicalResult
StructType::verifySymbol(mlir::SymbolTableCollection &symbolTable,
                         mlir::Operation *op) {
  if (!getDefinition(symbolTable, op)) {
    op->emitOpError() << "undefined component: " << *this;
    return mlir::failure();
  }
  return mlir::success();
}

} // namespace zkir
