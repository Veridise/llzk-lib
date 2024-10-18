#pragma once

#include <mlir/IR/BuiltinOps.h>

namespace zkir {

extern const std::string LANG_ATTR_NAME;

mlir::FailureOr<mlir::ModuleOp> getRootModule(mlir::Operation *op);

template <typename T, typename NameT>
inline mlir::FailureOr<T> lookupSymbolIn(
    mlir::SymbolTableCollection &symbolTable, NameT &&symbol, mlir::Operation *at,
    mlir::Operation *origin
) {
  auto found = symbolTable.lookupSymbolIn(at, std::forward<NameT>(symbol));
  if (!found) {
    return origin->emitOpError() << "references unknown symbol \"" << symbol << "\"";
  }
  if (T ret = llvm::dyn_cast<T>(found)) {
    return ret;
  }
  return origin->emitError() << "symbol \"" << symbol << "\" references a '" << found->getName()
                             << "' but expected a '" << T::getOperationName() << "'";
}

template <typename T, typename NameT>
inline mlir::FailureOr<T> lookupTopLevelSymbol(
    mlir::SymbolTableCollection &symbolTable, mlir::Operation *op, NameT &&symbol
) {
  mlir::FailureOr<mlir::ModuleOp> root = getRootModule(op);
  if (mlir::failed(root)) {
    return root; // getRootModule() already emits a sufficient error message
  }
  return lookupSymbolIn<T, NameT>(symbolTable, std::forward<NameT>(symbol), root.value(), op);
}

} // namespace zkir
