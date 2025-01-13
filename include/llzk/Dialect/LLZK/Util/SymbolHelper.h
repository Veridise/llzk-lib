#pragma once

#include "llzk/Dialect/LLZK/IR/Ops.h"
#include "llzk/Dialect/LLZK/Util/SymbolLookup.h"

#include <mlir/IR/BuiltinOps.h>

#include <llvm/Support/Casting.h>

namespace llzk {

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

/// @brief With include statements, there may be root modules nested within
/// other root modules. This function resolves the topmost root module.
mlir::FailureOr<mlir::ModuleOp> getTopRootModule(mlir::Operation *from);
mlir::FailureOr<mlir::SymbolRefAttr> getPathFromTopRoot(StructDefOp &to);
mlir::FailureOr<mlir::SymbolRefAttr> getPathFromTopRoot(FuncOp &to);

/// @brief Based on mlir::CallOpInterface::resolveCallable, but using LLZK lookup helpers
/// @tparam T the type of symbol being resolved (e.g., llzk::FuncOp)
/// @param symbolTable
/// @param call
/// @return the symbol or failure
template <typename T>
inline mlir::FailureOr<SymbolLookupResult<T>>
resolveCallable(mlir::SymbolTableCollection &symbolTable, mlir::CallOpInterface call) {
  mlir::CallInterfaceCallable callable = call.getCallableForCallee();
  if (auto symbolVal = llvm::dyn_cast<mlir::Value>(callable)) {
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

mlir::LogicalResult verifyParamOfType(
    mlir::SymbolTableCollection &tables, mlir::SymbolRefAttr param, mlir::Type structOrArrayType,
    mlir::Operation *origin
);

mlir::LogicalResult verifyParamsOfType(
    mlir::SymbolTableCollection &tables, mlir::ArrayRef<mlir::Attribute> tyParams,
    mlir::Type structOrArrayType, mlir::Operation *origin
);

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
