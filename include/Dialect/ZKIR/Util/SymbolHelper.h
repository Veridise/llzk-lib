#pragma once

#include <mlir/IR/BuiltinOps.h>

namespace zkir {

inline mlir::ModuleOp getModule(mlir::Operation *op) {
  return op->getParentOfType<mlir::ModuleOp>();
}

template <typename T, typename NameT>
inline T lookupTopLevelSymbol(
    mlir::SymbolTableCollection &symbolTable, mlir::Operation *op, NameT &&symbol
) {
  return llvm::dyn_cast_or_null<T>(
      symbolTable.lookupSymbolIn(getModule(op), std::forward<NameT>(symbol))
  );
}

} // namespace zkir
