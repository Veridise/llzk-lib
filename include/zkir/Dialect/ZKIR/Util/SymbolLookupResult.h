#pragma once

#include <llvm/Support/Casting.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/OwningOpRef.h>

namespace zkir {
class SymbolLookupResultUntyped {
public:
  SymbolLookupResultUntyped();
  SymbolLookupResultUntyped(mlir::Operation *op);

  // Since we don't want to copy around this class
  // the move operations are manually implemented to
  // respect the rule of 5.
  SymbolLookupResultUntyped(const SymbolLookupResultUntyped &) = delete;
  SymbolLookupResultUntyped(SymbolLookupResultUntyped &&);
  SymbolLookupResultUntyped &operator=(const SymbolLookupResultUntyped &) = delete;
  SymbolLookupResultUntyped &operator=(SymbolLookupResultUntyped &&);

  /// Access the internal operation.
  mlir::Operation *operator->();
  mlir::Operation &operator*();
  mlir::Operation &operator*() const;
  mlir::Operation *get();
  mlir::Operation *get() const;

  /// True iff the symbol was found.
  operator bool() const;

  /// Adds a pointer to the set of resources the result has to manage the lifetime of.
  void manage(mlir::OwningOpRef<mlir::ModuleOp> &&);

private:
  mlir::Operation *op;
  // It HAS to be a std::vector because llvm::SmallVector doesn't
  // play nice with deleted copy constructor and copy assignment.
  std::vector<mlir::OwningOpRef<mlir::ModuleOp>> managedResources;
};

template <typename T> class SymbolLookupResult {
public:
  SymbolLookupResult(SymbolLookupResultUntyped inner) : inner(std::move(inner)) {}

  /// Access the internal operation as type T.
  /// Follows the behaviors of llvm::dyn_cast if the internal operation cannot be casted to that
  /// type.
  T operator->() { return llvm::dyn_cast<T>(*inner); }
  T operator*() { return llvm::dyn_cast<T>(*inner); }
  const T operator*() const { return llvm::dyn_cast<T>(*inner); }
  T get() { return llvm::dyn_cast<T>(inner.get()); }
  T get() const { return llvm::dyn_cast<T>(inner.get()); }

  operator bool() const { return inner && llvm::isa<T>(*inner); }

private:
  SymbolLookupResultUntyped inner;
};

} // namespace zkir
