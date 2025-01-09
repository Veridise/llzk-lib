#include "llzk/Dialect/LLZK/IR/Ops.h"
#include "llzk/Dialect/LLZK/Util/IncludeHelper.h"
#include "llzk/Dialect/LLZK/Util/SymbolHelper.h"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/OwningOpRef.h>

#include <llvm/Support/Debug.h>

#define DEBUG_TYPE "llzk-symbol-helpers"

namespace llzk {
using namespace mlir;

//===------------------------------------------------------------------===//
// SymbolLookupResultUntyped
//===------------------------------------------------------------------===//

SymbolLookupResultUntyped::SymbolLookupResultUntyped(mlir::Operation *t_op) : op(t_op) {}
SymbolLookupResultUntyped::SymbolLookupResultUntyped() : op(nullptr) {}

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
  managedResources.push_back(std::make_shared<mlir::OwningOpRef<mlir::ModuleOp>>(std::move(ptr)));
}

/// Adds a pointer to the set of resources the result has to manage the lifetime of.
void SymbolLookupResultUntyped::trackIncludeAsName(llvm::StringRef includeOpSymName) {
  includeSymNameStack.push_back(includeOpSymName);
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
  // Reverse the vector and convert it to a SymbolRefAttr
  std::vector<FlatSymbolRefAttr> reversedVec(path.rbegin(), path.rend());
  return asSymbolRefAttr(reversedVec);
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

llvm::SmallVector<StringRef> getNames(const SymbolRefAttr &ref) {
  llvm::SmallVector<StringRef> names;
  names.push_back(ref.getRootReference().getValue());
  for (const FlatSymbolRefAttr &r : ref.getNestedReferences()) {
    names.push_back(r.getValue());
  }
  return names;
}

llvm::SmallVector<FlatSymbolRefAttr> getPieces(const SymbolRefAttr &ref) {
  llvm::SmallVector<FlatSymbolRefAttr> pieces;
  pieces.push_back(FlatSymbolRefAttr::get(ref.getRootReference()));
  for (const FlatSymbolRefAttr &r : ref.getNestedReferences()) {
    pieces.push_back(r);
  }
  return pieces;
}

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

SymbolLookupResultUntyped
lookupSymbolRec(SymbolTableCollection &tables, SymbolRefAttr symbol, Operation *symTableOp) {
  // First try a direct lookup via the SymbolTableCollection.  Must use a low-level lookup function
  // in order to properly account for modules that were added due to inlining IncludeOp.
  {
    SmallVector<Operation *, 4> symbolsFound;
    if (succeeded(tables.lookupSymbolIn(symTableOp, symbol, symbolsFound))) {
      SymbolLookupResultUntyped ret(symbolsFound.back());
      for (auto it = symbolsFound.rbegin(); it != symbolsFound.rend(); ++it) {
        Operation *op = *it;
        if (op->hasAttr(LANG_ATTR_NAME)) {
          ret.trackIncludeAsName(op->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName()));
        }
      }
      return ret;
    }
  }
  // Otherwise, check if the reference can be found by manually doing a lookup for each part of
  // the reference in turn, traversing through IncludeOp symbols by parsing the included file.
  if (Operation *rootOp = tables.lookupSymbolIn(symTableOp, symbol.getRootReference())) {
    if (IncludeOp rootOpInc = llvm::dyn_cast<IncludeOp>(rootOp)) {
      FailureOr<OwningOpRef<ModuleOp>> otherMod = rootOpInc.openModule();
      if (succeeded(otherMod)) {
        SymbolTableCollection external;
        auto result = lookupSymbolRec(external, getTailAsSymbolRefAttr(symbol), otherMod->get());
        if (result) {
          result.manage(std::move(*otherMod));
          result.trackIncludeAsName(rootOpInc.getSymName());
        }
        return result;
      }
    } else if (ModuleOp rootOpMod = llvm::dyn_cast<ModuleOp>(rootOp)) {
      return lookupSymbolRec(tables, getTailAsSymbolRefAttr(symbol), rootOpMod);
    }
  }
  // Otherwise, return empty result
  return SymbolLookupResultUntyped();
}

LogicalResult verifyParamOfType(
    SymbolTableCollection &tables, SymbolRefAttr param, Type parameterizedType, Operation *origin
) {
  // Most often, StructType and ArrayType SymbolRefAttr parameters will be defined as parameters of
  // the StructDefOp that the current Operation is nested within. These are always flat references
  // (i.e. contain no nested references).
  if (param.getNestedReferences().empty()) {
    FailureOr<StructDefOp> getParentRes = getParentOfType<StructDefOp>(origin);
    if (succeeded(getParentRes)) {
      if (getParentRes->hasParamNamed(param.getRootReference())) {
        return success();
      }
    }
  }
  // Otherwise, see if the symbol can be found via lookup from the `origin` Operation.
  auto lookupRes = lookupTopLevelSymbol(tables, param, origin);
  if (failed(lookupRes)) {
    return failure(); // lookupTopLevelSymbol() already emits a sufficient error message
  }
  Operation *foundOp = lookupRes->get();
  // TODO: Currently there is no type of Symbol Operation that is valid here. However, when
  //  the GlobalConstDef Operation is added, it will be valid to use in this context.
  return origin->emitError() << "ref \"" << param << "\" in type " << parameterizedType
                             << " refers to a '" << foundOp->getName() << "' which is not allowed";
}

LogicalResult verifyParamsOfType(
    SymbolTableCollection &tables, ArrayRef<Attribute> tyParams, Type parameterizedType,
    Operation *origin
) {
  // Rather than immediately returning on failure, we check all params and aggregate to provide as
  // many errors are possible in a single verifier run.
  LogicalResult paramCheckResult = success();
  for (Attribute attr : tyParams) {
    assertValidAttrForParamOfType(attr);
    if (SymbolRefAttr symRefParam = llvm::dyn_cast<SymbolRefAttr>(attr)) {
      if (failed(verifyParamOfType(tables, symRefParam, parameterizedType, origin))) {
        paramCheckResult = failure();
      }
    } else if (TypeAttr typeParam = llvm::dyn_cast<TypeAttr>(attr)) {
      if (failed(verifyTypeResolution(tables, typeParam.getValue(), origin))) {
        paramCheckResult = failure();
      }
    }
  }
  return paramCheckResult;
}

FailureOr<StructDefOp>
verifyStructTypeResolution(SymbolTableCollection &tables, StructType ty, Operation *origin) {
  auto res = ty.getDefinition(tables, origin);
  if (failed(res)) {
    return failure();
  }
  StructDefOp defForType = res.value().get();
  if (!structTypesUnify(ty, defForType.getType({}), res->getIncludeSymNames())) {
    return origin->emitError()
        .append(
            "Cannot unify parameters of type ", ty, " with parameters of '",
            StructDefOp::getOperationName(), "' \"", defForType.getHeaderString(), "\""
        )
        .attachNote(defForType.getLoc())
        .append("type parameters must unify with parameters defined here");
  }
  // If there are any SymbolRefAttr parameters on the StructType, ensure those refs are valid.
  if (ArrayAttr tyParams = ty.getParams()) {
    if (failed(verifyParamsOfType(tables, tyParams.getValue(), ty, origin))) {
      return failure();
    }
  }
  return defForType;
}

LogicalResult verifyTypeResolution(SymbolTableCollection &tables, Type ty, Operation *origin) {
  if (StructType sTy = llvm::dyn_cast<StructType>(ty)) {
    return verifyStructTypeResolution(tables, sTy, origin);
  } else if (ArrayType aTy = llvm::dyn_cast<ArrayType>(ty)) {
    if (failed(verifyParamsOfType(tables, aTy.getDimensionSizes(), aTy, origin))) {
      return failure();
    }
    return verifyTypeResolution(tables, aTy.getElementType(), origin);
  } else if (TypeVarType vTy = llvm::dyn_cast<TypeVarType>(ty)) {
    return verifyParamOfType(tables, vTy.getNameRef(), vTy, origin);
  } else {
    return success();
  }
}

LogicalResult verifyTypeResolution(
    SymbolTableCollection &tables, llvm::ArrayRef<Type>::iterator start,
    llvm::ArrayRef<Type>::iterator end, Operation *origin
) {
  LogicalResult res = success();
  for (; start != end; ++start) {
    if (failed(verifyTypeResolution(tables, *start, origin))) {
      res = failure();
    }
  }
  return res;
}

} // namespace llzk
