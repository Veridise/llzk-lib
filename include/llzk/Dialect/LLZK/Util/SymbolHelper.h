#pragma once

#include "llzk/Dialect/LLZK/IR/Ops.h"
#include "llzk/Dialect/LLZK/Util/SymbolLookupResult.h"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/OwningOpRef.h>

#include <llvm/Support/Casting.h>

namespace llzk {

constexpr char LANG_ATTR_NAME[] = "veridise.lang";

llvm::SmallVector<mlir::StringRef> getNames(const mlir::SymbolRefAttr &ref);
llvm::SmallVector<mlir::FlatSymbolRefAttr> getPieces(const mlir::SymbolRefAttr &ref);

inline mlir::SymbolRefAttr asSymbolRefAttr(mlir::StringAttr root, mlir::SymbolRefAttr tail) {
  return mlir::SymbolRefAttr::get(root, getPieces(tail));
}

inline mlir::SymbolRefAttr asSymbolRefAttr(llvm::ArrayRef<mlir::FlatSymbolRefAttr> path) {
  return mlir::SymbolRefAttr::get(path.front().getAttr(), path.drop_front());
}

inline mlir::SymbolRefAttr asSymbolRefAttr(std::vector<mlir::FlatSymbolRefAttr> path) {
  return asSymbolRefAttr(llvm::ArrayRef<mlir::FlatSymbolRefAttr>(path));
}

inline mlir::SymbolRefAttr getTailAsSymbolRefAttr(mlir::SymbolRefAttr &symbol) {
  return asSymbolRefAttr(symbol.getNestedReferences());
}

mlir::FailureOr<mlir::ModuleOp> getRootModule(mlir::Operation *from);
mlir::FailureOr<mlir::SymbolRefAttr> getPathFromRoot(StructDefOp &to);
mlir::FailureOr<mlir::SymbolRefAttr> getPathFromRoot(FuncOp &to);

SymbolLookupResultUntyped lookupSymbolRec(
    mlir::SymbolTableCollection &tables, mlir::SymbolRefAttr sym, mlir::Operation *symTableOp
);

inline mlir::FailureOr<SymbolLookupResultUntyped> lookupSymbolIn(
    mlir::SymbolTableCollection &tables, mlir::SymbolRefAttr symbol, mlir::Operation *symTableOp,
    mlir::Operation *origin
) {
  auto found = lookupSymbolRec(tables, symbol, symTableOp);
  if (!found) {
    return origin->emitOpError() << "references unknown symbol \"" << symbol << "\"";
  }
  return found;
}

inline mlir::FailureOr<SymbolLookupResultUntyped> lookupTopLevelSymbol(
    mlir::SymbolTableCollection &tables, mlir::SymbolRefAttr symbol, mlir::Operation *origin
) {
  mlir::FailureOr<mlir::ModuleOp> root = getRootModule(origin);
  if (mlir::failed(root)) {
    return mlir::failure(); // getRootModule() already emits a sufficient error message
  }
  return lookupSymbolIn(tables, symbol, root.value(), origin);
}

template <typename T>
inline mlir::FailureOr<SymbolLookupResult<T>> lookupSymbolIn(
    mlir::SymbolTableCollection &tables, mlir::SymbolRefAttr symbol, mlir::Operation *symTableOp,
    mlir::Operation *origin
) {
  auto found = lookupSymbolIn(tables, symbol, symTableOp, origin);
  if (mlir::failed(found)) {
    return mlir::failure(); // lookupSymbolIn() already emits a sufficient error message
  }
  // Keep a copy of the op ptr in case we need it for displaying diagnostics
  auto *op = found->get();
  // Since the untyped result gets moved here into a typed result.
  SymbolLookupResult<T> ret(std::move(*found));
  if (!ret) {
    return origin->emitError() << "symbol \"" << symbol << "\" references a '" << op->getName()
                               << "' but expected a '" << T::getOperationName() << "'";
  }
  return std::move(ret);
}

template <typename T>
inline mlir::FailureOr<SymbolLookupResult<T>> lookupTopLevelSymbol(
    mlir::SymbolTableCollection &tables, mlir::SymbolRefAttr symbol, mlir::Operation *origin
) {
  mlir::FailureOr<mlir::ModuleOp> root = getRootModule(origin);
  if (mlir::failed(root)) {
    return mlir::failure(); // getRootModule() already emits a sufficient error message
  }
  return lookupSymbolIn<T>(tables, symbol, root.value(), origin);
}

mlir::LogicalResult verifyStructOrArrayParam(
    mlir::SymbolTableCollection &tables, mlir::SymbolRefAttr param, mlir::Type structOrArrayType,
    mlir::Operation *origin
);

mlir::LogicalResult verifyStructOrArrayParams(
    mlir::SymbolTableCollection &tables, mlir::ArrayRef<mlir::Attribute> tyParams,
    mlir::Type structOrArrayType, mlir::Operation *origin
);

/// @brief Based on mlir::CallOpInterface::resolveCallable, but using LLZK lookup helpers
/// @tparam T the type of symbol being resolved (e.g., llzk::FuncOp)
/// @param symbolTable
/// @param call
/// @return the symbol or failure
template <typename T>
inline mlir::FailureOr<SymbolLookupResult<T>>
resolveCallable(mlir::SymbolTableCollection &symbolTable, mlir::CallOpInterface call) {
  mlir::CallInterfaceCallable callable = call.getCallableForCallee();
  if (auto symbolVal = dyn_cast<mlir::Value>(callable)) {
    return SymbolLookupResult<T>(symbolVal.getDefiningOp());
  }

  // If the callable isn't a value, lookup the symbol reference.
  // We first try to resolve in the nearest symbol table, as per the default
  // MLIR behavior. If the resulting operation is not found, we will then
  // use the LLZK lookup helpers.
  auto symbolRef = callable.get<mlir::SymbolRefAttr>();
  mlir::Operation *op = symbolTable.lookupNearestSymbolFrom(call.getOperation(), symbolRef);

  if (op) {
    return SymbolLookupResult<T>(std::move(op));
  }
  // Otherwise, use the top-level lookup.
  return lookupTopLevelSymbol<T>(symbolTable, symbolRef, call.getOperation());
}

template <typename T>
inline mlir::FailureOr<SymbolLookupResult<T>> resolveCallable(mlir::CallOpInterface call) {
  mlir::SymbolTableCollection symbolTable;
  return resolveCallable<T>(symbolTable, call);
}

mlir::FailureOr<StructDefOp> verifyStructTypeResolution(
    mlir::SymbolTableCollection &tables, StructType ty, mlir::Operation *origin
);

mlir::LogicalResult
verifyTypeResolution(mlir::SymbolTableCollection &tables, mlir::Type ty, mlir::Operation *origin);

mlir::LogicalResult verifyTypeResolution(
    mlir::SymbolTableCollection &tables, llvm::ArrayRef<mlir::Type>::iterator start,
    llvm::ArrayRef<mlir::Type>::iterator end, mlir::Operation *origin
);

inline mlir::LogicalResult verifyTypeResolution(
    mlir::SymbolTableCollection &tables, llvm::ArrayRef<mlir::Type> types, mlir::Operation *origin
) {
  return verifyTypeResolution(tables, types.begin(), types.end(), origin);
}

} // namespace llzk
