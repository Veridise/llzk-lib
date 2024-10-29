#include "Dialect/ZKIR/Util/SymbolHelper.h"
#include "Dialect/ZKIR/IR/Ops.h"

#include <mlir/IR/BuiltinOps.h>

namespace zkir {

namespace {
using namespace mlir;

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

mlir::FailureOr<mlir::ModuleOp> getRootModule(mlir::Operation *from) {
  std::vector<FlatSymbolRefAttr> path;
  return collectPathToRoot(from, from, path);
}

mlir::FailureOr<mlir::SymbolRefAttr> getPathFromRoot(StructDefOp &to) {
  std::vector<FlatSymbolRefAttr> path;
  return buildPathFromRoot(to, to.getOperation(), std::move(path));
}

mlir::FailureOr<mlir::SymbolRefAttr> getPathFromRoot(FuncOp &to) {
  using namespace mlir;

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

mlir::LogicalResult verifyTypeResolution(
    mlir::SymbolTableCollection &symbolTable, mlir::Type ty, mlir::Operation *origin
) {
  if (StructType sTy = llvm::dyn_cast<StructType>(ty)) {
    return sTy.getDefinition(symbolTable, origin);
  } else if (ArrayType aTy = llvm::dyn_cast<ArrayType>(ty)) {
    return verifyTypeResolution(symbolTable, aTy.getElementType(), origin);
  } else {
    return mlir::success();
  }
}

mlir::LogicalResult verifyTypeResolution(
    mlir::SymbolTableCollection &symbolTable, llvm::ArrayRef<mlir::Type>::iterator start,
    llvm::ArrayRef<mlir::Type>::iterator end, mlir::Operation *origin
) {
  mlir::LogicalResult res = mlir::success();
  for (; start != end; ++start) {
    if (mlir::failed(verifyTypeResolution(symbolTable, *start, origin))) {
      res = mlir::failure();
    }
  }
  return res;
}

} // namespace zkir
