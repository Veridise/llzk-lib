#pragma once

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/OwningOpRef.h>

#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Casting.h>

#include <vector>

namespace llzk {
class SymbolLookupResultUntyped {
public:
  SymbolLookupResultUntyped();
  SymbolLookupResultUntyped(mlir::Operation *op);

  /// Access the internal operation.
  mlir::Operation *operator->();
  mlir::Operation &operator*();
  mlir::Operation &operator*() const;
  mlir::Operation *get();
  mlir::Operation *get() const;

  /// True iff the symbol was found.
  operator bool() const;

  mlir::SmallVector<llvm::StringRef> getIncludeSymNames() { return includeSymNameStack; }

  /// Adds a pointer to the set of resources the result has to manage the lifetime of.
  void manage(mlir::OwningOpRef<mlir::ModuleOp> &&ptr);

  /// Adds the symbol name from the IncludeOp that caused the module to be loaded.
  void trackIncludeAsName(llvm::StringRef includeOpSymName);

private:
  mlir::Operation *op;
  // Wrap owning refs in shared pointers to enable copying.
  mlir::SmallVector<std::shared_ptr<mlir::OwningOpRef<mlir::ModuleOp>>> managedResources;
  /// Stack of symbol names from the IncludeOp that were traversed in order to load the Operation.
  mlir::SmallVector<llvm::StringRef> includeSymNameStack;
};

template <typename T> class SymbolLookupResult {
public:
  SymbolLookupResult(SymbolLookupResultUntyped &&inner) : inner(std::move(inner)) {}

  /// Access the internal operation as type T.
  /// Follows the behaviors of llvm::dyn_cast if the internal operation cannot cast to that type.
  T operator->() { return llvm::dyn_cast<T>(*inner); }
  T operator*() { return llvm::dyn_cast<T>(*inner); }
  const T operator*() const { return llvm::dyn_cast<T>(*inner); }
  T get() { return llvm::dyn_cast<T>(inner.get()); }
  T get() const { return llvm::dyn_cast<T>(inner.get()); }

  operator bool() const { return inner && llvm::isa<T>(*inner); }

  mlir::SmallVector<llvm::StringRef> getIncludeSymNames() { return inner.getIncludeSymNames(); }

private:
  SymbolLookupResultUntyped inner;
};

} // namespace llzk
