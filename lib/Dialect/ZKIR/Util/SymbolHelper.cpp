#include "zkir/Dialect/ZKIR/Util/SymbolHelper.h"
#include "zkir/Dialect/ZKIR/IR/Ops.h"
#include "zkir/Dialect/ZKIR/Util/IncludeHelper.h"

#include <llvm/Support/Debug.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/OwningOpRef.h>

#define DEBUG_TYPE "zkir-symbol-helpers"

namespace zkir {
using namespace mlir;

//===------------------------------------------------------------------===//
// SymbolLookupResultUntyped
//===------------------------------------------------------------------===//

SymbolLookupResultUntyped::SymbolLookupResultUntyped(mlir::Operation *t_op) : op(t_op) {}
SymbolLookupResultUntyped::SymbolLookupResultUntyped() : op(nullptr) {}

// Move constructor
SymbolLookupResultUntyped::SymbolLookupResultUntyped(SymbolLookupResultUntyped &&other)
    : op(other.op), managedResources(std::move(other.managedResources)) {
  other.op = nullptr;
}

// Move assigment
SymbolLookupResultUntyped &SymbolLookupResultUntyped::operator=(SymbolLookupResultUntyped &&other) {
  if (this != &other) {
    managedResources.clear();
    managedResources = std::move(other.managedResources);
    op = other.op;
    other.op = nullptr;
  }
  return *this;
}

/// Access the internal operation.
mlir::Operation *SymbolLookupResultUntyped::operator->() { return op; }
mlir::Operation &SymbolLookupResultUntyped::operator*() { return *op; }
mlir::Operation &SymbolLookupResultUntyped::operator*() const { return *op; }
mlir::Operation *SymbolLookupResultUntyped::get() { return op; }
mlir::Operation *SymbolLookupResultUntyped::get() const { return op; }

/// True iff the symbol was found.
SymbolLookupResultUntyped::operator bool() const { return op != nullptr; }

/// Adds a pointer to the set of resources the result has to manage the lifetime of.
void SymbolLookupResultUntyped::manage(mlir::OwningOpRef<mlir::ModuleOp> &&ptr) {
  // Hand over the pointer
  managedResources.push_back(std::move(ptr));
}

namespace {

/// Traverse ModuleOp ancestors of `from` and add their names to `path` until the ModuleOp with the
/// LANG_ATTR_NAME attribute is reached. If a ModuleOp without a name is reached or a ModuleOp with
/// the LANG_ATTR_NAME attribute is never found, produce an error (referencing the `origin`
/// Operation). Returns the module containing the LANG_ATTR_NAME attribute.
FailureOr<ModuleOp>
collectPathToRoot(Operation *from, Operation *origin, std::vector<FlatSymbolRefAttr> &path) {
  Operation *check = from;
  do {
    if (ModuleOp m = llvm::dyn_cast_if_present<ModuleOp>(check)) {
      // We need this attribute restriction because some stages of parsing have
      //  an extra module wrapping the top-level module from the input file.
      // This module, even if it has a name, does not contribute to path names.
      if (m->hasAttr(LANG_ATTR_NAME)) {
        return m;
      }
      if (StringAttr modName = m.getSymNameAttr()) {
        path.push_back(FlatSymbolRefAttr::get(modName));
      } else {
        return origin->emitOpError()
            .append(
                "has ancestor '", ModuleOp::getOperationName(), "' without \"", LANG_ATTR_NAME,
                "\" attribute or a name"
            )
            .attachNote(m.getLoc())
            .append("unnamed '", ModuleOp::getOperationName(), "' here");
      }
    }
  } while ((check = check->getParentOp()));
  //
  return origin->emitOpError().append(
      "has no ancestor '", ModuleOp::getOperationName(), "' with \"", LANG_ATTR_NAME, "\" attribute"
  );
}

/// Appends the `path` via `collectPathToRoot()` starting from `position` and then convert that path
/// into a SymbolRefAttr.
FailureOr<SymbolRefAttr>
buildPathFromRoot(Operation *position, Operation *origin, std::vector<FlatSymbolRefAttr> &&path) {
  // Collect the rest of the path to the root module
  if (failed(collectPathToRoot(position, origin, path))) {
    return failure();
  }
  // Get the root module off the back of the vector
  FlatSymbolRefAttr root = path.back();
  path.pop_back();
  // Reverse the vector and convert it to a SymbolRefAttr
  std::vector<FlatSymbolRefAttr> reversedVec(path.rbegin(), path.rend());
  llvm::ArrayRef<FlatSymbolRefAttr> nestedReferences(reversedVec);
  return SymbolRefAttr::get(root.getAttr(), nestedReferences);
}

/// Appends the `path` via `collectPathToRoot()` starting from the given `StructDefOp` and then
/// convert that path into a SymbolRefAttr.
FailureOr<SymbolRefAttr>
buildPathFromRoot(StructDefOp &to, Operation *origin, std::vector<FlatSymbolRefAttr> &&path) {
  // Add the name of the struct (its name is not optional) and then delegate to helper
  path.push_back(FlatSymbolRefAttr::get(to.getSymNameAttr()));
  return buildPathFromRoot(to.getOperation(), origin, std::move(path));
}
} // namespace

FailureOr<ModuleOp> getRootModule(Operation *from) {
  std::vector<FlatSymbolRefAttr> path;
  return collectPathToRoot(from, from, path);
}

FailureOr<SymbolRefAttr> getPathFromRoot(StructDefOp &to) {
  std::vector<FlatSymbolRefAttr> path;
  return buildPathFromRoot(to, to.getOperation(), std::move(path));
}

FailureOr<SymbolRefAttr> getPathFromRoot(FuncOp &to) {
  std::vector<FlatSymbolRefAttr> path;
  // Add the name of the function (its name is not optional)
  path.push_back(FlatSymbolRefAttr::get(to.getSymNameAttr()));

  // Delegate based on the type of the parent op
  Operation *current = to.getOperation();
  Operation *parent = current->getParentOp();
  if (StructDefOp parentStruct = llvm::dyn_cast_if_present<StructDefOp>(parent)) {
    return buildPathFromRoot(parentStruct, current, std::move(path));
  } else if (ModuleOp parentMod = llvm::dyn_cast_if_present<ModuleOp>(parent)) {
    return buildPathFromRoot(parentMod.getOperation(), current, std::move(path));
  } else {
    // This is an error in the compiler itself. In current implementation,
    //  FuncOp must have either StructDefOp or ModuleOp as its parent.
    return current->emitError().append("orphaned '", FuncOp::getOperationName(), "'");
  }
}

namespace {
inline SymbolRefAttr getTailAsSymbolRefAttr(SymbolRefAttr &symbol) {
  llvm::ArrayRef<FlatSymbolRefAttr> nest = symbol.getNestedReferences();
  return SymbolRefAttr::get(nest.front().getAttr(), nest.drop_front());
}
} // namespace

SymbolLookupResultUntyped
lookupSymbolRec(SymbolTableCollection &tables, SymbolRefAttr symbol, Operation *symTableOp) {
  Operation *found = tables.lookupSymbolIn(symTableOp, symbol);
  if (!found) {
    // If not found, check if the reference can be found by manually doing a lookup for each part of
    // the reference in turn, traversing through IncludeOp symbols by parsing the included file.
    if (Operation *rootOp = tables.lookupSymbolIn(symTableOp, symbol.getRootReference())) {
      if (IncludeOp rootOpInc = llvm::dyn_cast<IncludeOp>(rootOp)) {
        FailureOr<OwningOpRef<ModuleOp>> otherMod = rootOpInc.openModule();
        if (succeeded(otherMod)) {
          auto result = lookupSymbolRec(tables, getTailAsSymbolRefAttr(symbol), otherMod->get());
          result.manage(std::move(*otherMod));
          return result;
        }
      } else if (ModuleOp rootOpMod = llvm::dyn_cast<ModuleOp>(rootOp)) {
        return lookupSymbolRec(tables, getTailAsSymbolRefAttr(symbol), rootOpMod);
      }
    }
  }
  return found;
}

LogicalResult verifyTypeResolution(SymbolTableCollection &symbolTable, Type ty, Operation *origin) {
  if (StructType sTy = llvm::dyn_cast<StructType>(ty)) {
    return sTy.getDefinition(symbolTable, origin);
  } else if (ArrayType aTy = llvm::dyn_cast<ArrayType>(ty)) {
    return verifyTypeResolution(symbolTable, aTy.getElementType(), origin);
  } else {
    return success();
  }
}

LogicalResult verifyTypeResolution(
    SymbolTableCollection &symbolTable, llvm::ArrayRef<Type>::iterator start,
    llvm::ArrayRef<Type>::iterator end, Operation *origin
) {
  LogicalResult res = success();
  for (; start != end; ++start) {
    if (failed(verifyTypeResolution(symbolTable, *start, origin))) {
      res = failure();
    }
  }
  return res;
}

} // namespace zkir
