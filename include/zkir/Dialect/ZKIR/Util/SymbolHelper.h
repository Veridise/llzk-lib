#pragma once

#include "zkir/Dialect/ZKIR/IR/Ops.h"

#include <mlir/IR/BuiltinOps.h>

namespace zkir {

constexpr char LANG_ATTR_NAME[] = "veridise.lang";

mlir::FailureOr<mlir::ModuleOp> getRootModule(mlir::Operation *from);
mlir::FailureOr<mlir::SymbolRefAttr> getPathFromRoot(StructDefOp &to);
mlir::FailureOr<mlir::SymbolRefAttr> getPathFromRoot(FuncOp &to);

ManagedOpPtr<mlir::Operation> lookupSymbolRec(
    mlir::SymbolTableCollection &tables, mlir::SymbolRefAttr sym, mlir::Operation *symTableOp
);

template <typename T>
inline mlir::FailureOr<ManagedOpPtr<T>> lookupSymbolIn(
    mlir::SymbolTableCollection &tables, mlir::SymbolRefAttr symbol, mlir::Operation *symTableOp,
    mlir::Operation *origin
) {
  // llvm::outs() << "[lookupSymbolIn] symbol = " << symbol << "\n";            // TODO: TEMP
  // if (!symTableOp) {                                                         // TODO: TEMP
  //   llvm::outs() << "[lookupSymbolIn] Found null sym table pointer" << "\n"; // TODO: TEMP
  //   // std::exit(1);                                                  // TODO: TEMP
  // } // TODO: TEMP

  llvm::outs() << "[lookupSymbolIn] BEFORE [lookupSymbolRec]" << "\n";
  ManagedOpPtr<mlir::Operation> found = lookupSymbolRec(tables, symbol, symTableOp);
  llvm::outs() << "[lookupSymbolIn] AFTER [lookupSymbolRec]" << "\n";
  if (!found) {
    return origin->emitOpError() << "references unknown symbol \"" << symbol << "\"";
  }
  if (T ret = llvm::dyn_cast<T>(found.get())) {
    return found.replacePtr(&ret);
  }
  return origin->emitError() << "symbol \"" << symbol << "\" references a '" << found->getName()
                             << "' but expected a '" << T::getOperationName() << "'";
}

template <typename T>
inline mlir::FailureOr<ManagedOpPtr<T>> lookupTopLevelSymbol(
    mlir::SymbolTableCollection &symbolTable, mlir::SymbolRefAttr symbol, mlir::Operation *origin
) {
  mlir::FailureOr<mlir::ModuleOp> root = getRootModule(origin);
  if (mlir::failed(root)) {
    return mlir::failure(); // getRootModule() already emits a sufficient error message
  }
  return lookupSymbolIn<T>(symbolTable, symbol, root.value(), origin);
}

mlir::LogicalResult verifyTypeResolution(
    mlir::SymbolTableCollection &symbolTable, mlir::Type ty, mlir::Operation *origin
);

mlir::LogicalResult verifyTypeResolution(
    mlir::SymbolTableCollection &symbolTable, llvm::ArrayRef<mlir::Type>::iterator start,
    llvm::ArrayRef<mlir::Type>::iterator end, mlir::Operation *origin
);

inline mlir::LogicalResult verifyTypeResolution(
    mlir::SymbolTableCollection &symbolTable, llvm::ArrayRef<mlir::Type> types,
    mlir::Operation *origin
) {
  return verifyTypeResolution(symbolTable, types.begin(), types.end(), origin);
}

} // namespace zkir
