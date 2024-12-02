#pragma once

#include "llzk/Dialect/LLZK/IR/Ops.h"
#include "llzk/Dialect/LLZK/Util/SymbolLookupResult.h"

#include <llvm/Support/Casting.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/OwningOpRef.h>

namespace llzk {

constexpr char LANG_ATTR_NAME[] = "veridise.lang";

mlir::FailureOr<mlir::ModuleOp> getRootModule(mlir::Operation *from);
mlir::FailureOr<mlir::SymbolRefAttr> getPathFromRoot(StructDefOp &to);
mlir::FailureOr<mlir::SymbolRefAttr> getPathFromRoot(FuncOp &to);

SymbolLookupResultUntyped lookupSymbolRec(
    mlir::SymbolTableCollection &tables, mlir::SymbolRefAttr sym, mlir::Operation *symTableOp
);

template <typename T>
inline mlir::FailureOr<SymbolLookupResult<T>> lookupSymbolIn(
    mlir::SymbolTableCollection &tables, mlir::SymbolRefAttr symbol, mlir::Operation *symTableOp,
    mlir::Operation *origin
) {
  auto found = lookupSymbolRec(tables, symbol, symTableOp);
  if (!found) {
    return origin->emitOpError() << "references unknown symbol \"" << symbol << "\"";
  }
  // Keep a copy of the op ptr in case we need it for displaying diagnostics
  auto *op = found.get();
  // Since the untyped result gets moved here into a typed result.
  SymbolLookupResult<T> ret(std::move(found));
  if (!ret) {
    return origin->emitError() << "symbol \"" << symbol << "\" references a '" << op->getName()
                               << "' but expected a '" << T::getOperationName() << "'";
  }
  return std::move(ret);
}

template <typename T>
inline mlir::FailureOr<SymbolLookupResult<T>> lookupTopLevelSymbol(
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

} // namespace llzk
