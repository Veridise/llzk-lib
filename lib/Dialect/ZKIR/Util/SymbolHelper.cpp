#include "zkir/Dialect/ZKIR/Util/SymbolHelper.h"
#include "zkir/Dialect/ZKIR/IR/Ops.h"
#include "zkir/Dialect/ZKIR/Util/IncludeHelper.h"

#include <mlir/IR/BuiltinOps.h>

namespace zkir {
using namespace mlir;

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

ManagedOpPtr<Operation>
lookupSymbolRec(SymbolTableCollection &tables, SymbolRefAttr symbol, Operation *symTableOp) {
  llvm::outs() << "[lookupSymbolRec] symbol = " << symbol << "\n";            // TODO: TEMP
  if (!symTableOp) {                                                          // TODO: TEMP
    llvm::outs() << "[lookupSymbolRec] Found null sym table pointer" << "\n"; // TODO: TEMP
    // std::exit(1);                                                  // TODO: TEMP
  } // TODO: TEMP
  Operation *found = tables.lookupSymbolIn(symTableOp, symbol);
  if (!found) {
    llvm::outs() << "[lookupSymbolRec] lookup via root" << "\n";
    // If not found, check if the reference can be found by manually doing a lookup for each part of
    // the reference in turn, traversing through IncludeOp symbols by parsing the included file.
    if (Operation *rootOp = tables.lookupSymbolIn(symTableOp, symbol.getRootReference())) {
      if (IncludeOp rootOpInc = llvm::dyn_cast<IncludeOp>(rootOp)) {
        llvm::outs() << "[lookupSymbolRec] loading module from \"" << rootOpInc.getPath() << "\"\n";
        FailureOr<OwningOpRef<ModuleOp>> otherMod = rootOpInc.loadModule();
        if (succeeded(otherMod)) {
          llvm::outs() << "[lookupSymbolRec] successfully loaded module "
                       << otherMod->get().getSymName() << "\n";
          ManagedOpPtr<Operation> res =
              lookupSymbolRec(tables, getTailAsSymbolRefAttr(symbol), otherMod->get());
          if (!res) {
            llvm::outs() << "[lookupSymbolRec] RETURN 1" << "\n";
            return res;
          }
          // If recursive lookup returned an Operation*, wrap it in a new ManagedOpPtr that includes
          // the ModuleOp so they have the same lifetime, controlled by the ManagedOpPtr.

          llvm::outs() << "[lookupSymbolRec] ownsTheModule = "
                       << (otherMod.value().get() == nullptr) << "\n"; // TODO: TEMP
          ManagedOpPtr<Operation> x(res.get(), std::move(otherMod.value()));
          llvm::outs() << "[lookupSymbolRec] '" << res.get()->getName() << "' named '" << symbol
                       << "' ownsTheModule = " << x.ownsTheModule() << "\n"; // TODO: TEMP
          llvm::outs() << "[lookupSymbolRec] RETURN 2" << "\n";
          return x;
        }
      } else if (ModuleOp rootOpMod = llvm::dyn_cast<ModuleOp>(rootOp)) {
        auto x = lookupSymbolRec(tables, getTailAsSymbolRefAttr(symbol), rootOpMod);
        llvm::outs() << "[lookupSymbolRec] RETURN 3" << "\n";
        return x;
      }
    }
  }
  llvm::outs() << "[lookupSymbolRec] RETURN 4" << "\n";
  return ManagedOpPtr<Operation>(found);
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
