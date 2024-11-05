#pragma once

#include "zkir/Dialect/ZKIR/IR/Ops.h"

#include <mlir/IR/BuiltinOps.h>

namespace zkir {
using namespace mlir;

constexpr char LANG_ATTR_NAME[] = "veridise.lang";

FailureOr<ModuleOp> getRootModule(Operation *from);
FailureOr<SymbolRefAttr> getPathFromRoot(StructDefOp &to);
FailureOr<SymbolRefAttr> getPathFromRoot(FuncOp &to);

Operation *lookupSymbolRec(SymbolTableCollection &tables, SymbolRefAttr sym, Operation *symTableOp);

template <typename T>
inline FailureOr<T> lookupSymbolIn(
    SymbolTableCollection &tables, SymbolRefAttr symbol, Operation *symTableOp, Operation *origin
) {
  Operation *found = lookupSymbolRec(tables, symbol, symTableOp);
  if (!found) {
    return origin->emitOpError() << "references unknown symbol \"" << symbol << "\"";
  }
  if (T ret = llvm::dyn_cast<T>(found)) {
    return ret;
  }
  return origin->emitError() << "symbol \"" << symbol << "\" references a '" << found->getName()
                             << "' but expected a '" << T::getOperationName() << "'";
}

template <typename T>
inline FailureOr<T>
lookupTopLevelSymbol(SymbolTableCollection &symbolTable, SymbolRefAttr symbol, Operation *origin) {
  FailureOr<ModuleOp> root = getRootModule(origin);
  if (failed(root)) {
    return root; // getRootModule() already emits a sufficient error message
  }
  return lookupSymbolIn<T>(symbolTable, symbol, root.value(), origin);
}

LogicalResult verifyTypeResolution(SymbolTableCollection &symbolTable, Type ty, Operation *origin);

LogicalResult verifyTypeResolution(
    SymbolTableCollection &symbolTable, llvm::ArrayRef<Type>::iterator start,
    llvm::ArrayRef<Type>::iterator end, Operation *origin
);

inline LogicalResult verifyTypeResolution(
    SymbolTableCollection &symbolTable, llvm::ArrayRef<Type> types, Operation *origin
) {
  return verifyTypeResolution(symbolTable, types.begin(), types.end(), origin);
}

} // namespace zkir
