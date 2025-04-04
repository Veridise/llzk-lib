//===-- SymbolHelper.cpp - LLZK Symbol Helpers ------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the implementations for symbol helper functions.
///
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/LLZK/IR/Ops.h"
#include "llzk/Dialect/LLZK/Util/SymbolHelper.h"
#include "llzk/Dialect/LLZK/Util/SymbolLookup.h"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Operation.h>

#include <llvm/Support/Debug.h>

#define DEBUG_TYPE "llzk-symbol-helpers"

namespace llzk {
using namespace mlir;

namespace {

enum RootSelector { CLOSEST, FURTHEST };

/// Traverse ModuleOp ancestors of `from` and add their names to `path` until
/// the desired ModuleOp with the LANG_ATTR_NAME attribute is reached (either the closest or
/// furthest, based on the arguments). If a ModuleOp without a name is reached or a ModuleOp with
/// the LANG_ATTR_NAME attribute is never found, produce an error (referencing the `origin`
/// Operation). Returns the module containing the LANG_ATTR_NAME attribute.
FailureOr<ModuleOp> collectPathToRoot(
    Operation *from, Operation *origin, std::vector<FlatSymbolRefAttr> &path, RootSelector whichRoot
) {
  Operation *check = from;
  ModuleOp currRoot = nullptr;
  do {
    if (ModuleOp m = llvm::dyn_cast_if_present<ModuleOp>(check)) {
      // We need this attribute restriction because some stages of parsing have
      //  an extra module wrapping the top-level module from the input file.
      // This module, even if it has a name, does not contribute to path names.
      if (m->hasAttr(LANG_ATTR_NAME)) {
        if (whichRoot == RootSelector::CLOSEST) {
          return m;
        }
        currRoot = m;
      }
      if (StringAttr modName = m.getSymNameAttr()) {
        path.push_back(FlatSymbolRefAttr::get(modName));
      } else if (!currRoot) {
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

  if (whichRoot == RootSelector::FURTHEST && currRoot) {
    return currRoot;
  }

  return origin->emitOpError().append(
      "has no ancestor '", ModuleOp::getOperationName(), "' with \"", LANG_ATTR_NAME, "\" attribute"
  );
}

/// Appends the `path` via `collectPathToRoot()` starting from `position` and then convert that path
/// into a SymbolRefAttr.
FailureOr<SymbolRefAttr> buildPathFromRoot(
    Operation *position, Operation *origin, std::vector<FlatSymbolRefAttr> &&path,
    RootSelector whichRoot
) {
  // Collect the rest of the path to the root module
  if (failed(collectPathToRoot(position, origin, path, whichRoot))) {
    return failure();
  }
  // Reverse the vector and convert it to a SymbolRefAttr
  std::vector<FlatSymbolRefAttr> reversedVec(path.rbegin(), path.rend());
  return asSymbolRefAttr(reversedVec);
}

/// Appends the `path` via `collectPathToRoot()` starting from the given `StructDefOp` and then
/// convert that path into a SymbolRefAttr.
FailureOr<SymbolRefAttr> buildPathFromRoot(
    StructDefOp &to, Operation *origin, std::vector<FlatSymbolRefAttr> &&path,
    RootSelector whichRoot
) {
  // Add the name of the struct (its name is not optional) and then delegate to helper
  path.push_back(FlatSymbolRefAttr::get(to.getSymNameAttr()));
  return buildPathFromRoot(to.getOperation(), origin, std::move(path), whichRoot);
}

FailureOr<SymbolRefAttr> getPathFromRoot(StructDefOp &to, RootSelector whichRoot) {
  std::vector<FlatSymbolRefAttr> path;
  return buildPathFromRoot(to, to.getOperation(), std::move(path), whichRoot);
}

FailureOr<SymbolRefAttr> getPathFromRoot(FuncOp &to, RootSelector whichRoot) {
  std::vector<FlatSymbolRefAttr> path;
  // Add the name of the function (its name is not optional)
  path.push_back(FlatSymbolRefAttr::get(to.getSymNameAttr()));

  // Delegate based on the type of the parent op
  Operation *current = to.getOperation();
  Operation *parent = current->getParentOp();
  if (StructDefOp parentStruct = llvm::dyn_cast_if_present<StructDefOp>(parent)) {
    return buildPathFromRoot(parentStruct, current, std::move(path), whichRoot);
  } else if (ModuleOp parentMod = llvm::dyn_cast_if_present<ModuleOp>(parent)) {
    return buildPathFromRoot(parentMod.getOperation(), current, std::move(path), whichRoot);
  } else {
    // This is an error in the compiler itself. In current implementation,
    //  FuncOp must have either StructDefOp or ModuleOp as its parent.
    return current->emitError().append("orphaned '", FuncOp::getOperationName(), "'");
  }
}
} // namespace

llvm::SmallVector<StringRef> getNames(SymbolRefAttr ref) {
  llvm::SmallVector<StringRef> names;
  names.push_back(ref.getRootReference().getValue());
  for (const FlatSymbolRefAttr &r : ref.getNestedReferences()) {
    names.push_back(r.getValue());
  }
  return names;
}

llvm::SmallVector<FlatSymbolRefAttr> getPieces(SymbolRefAttr ref) {
  llvm::SmallVector<FlatSymbolRefAttr> pieces;
  pieces.push_back(FlatSymbolRefAttr::get(ref.getRootReference()));
  for (const FlatSymbolRefAttr &r : ref.getNestedReferences()) {
    pieces.push_back(r);
  }
  return pieces;
}

namespace {

SymbolRefAttr changeLeafImpl(
    StringAttr origRoot, ArrayRef<FlatSymbolRefAttr> origTail, FlatSymbolRefAttr newLeaf,
    size_t drop = 1
) {
  llvm::SmallVector<FlatSymbolRefAttr> newTail;
  newTail.append(origTail.begin(), origTail.drop_back(drop).end());
  newTail.push_back(newLeaf);
  return SymbolRefAttr::get(origRoot, newTail);
}

} // namespace

SymbolRefAttr replaceLeaf(SymbolRefAttr orig, FlatSymbolRefAttr newLeaf) {
  ArrayRef<FlatSymbolRefAttr> origTail = orig.getNestedReferences();
  if (origTail.empty()) {
    // If there is no tail, the root is the leaf so replace the whole thing
    return newLeaf;
  } else {
    return changeLeafImpl(orig.getRootReference(), origTail, newLeaf);
  }
}

SymbolRefAttr appendLeaf(SymbolRefAttr orig, FlatSymbolRefAttr newLeaf) {
  return changeLeafImpl(orig.getRootReference(), orig.getNestedReferences(), newLeaf, 0);
}

SymbolRefAttr appendLeafName(SymbolRefAttr orig, const Twine &newLeafSuffix) {
  ArrayRef<FlatSymbolRefAttr> origTail = orig.getNestedReferences();
  if (origTail.empty()) {
    // If there is no tail, the root is the leaf so append on the root instead
    return getFlatSymbolRefAttr(
        orig.getContext(), orig.getRootReference().getValue() + newLeafSuffix
    );
  } else {
    return changeLeafImpl(
        orig.getRootReference(), origTail,
        getFlatSymbolRefAttr(orig.getContext(), origTail.back().getValue() + newLeafSuffix)
    );
  }
}

FailureOr<ModuleOp> getRootModule(Operation *from) {
  std::vector<FlatSymbolRefAttr> path;
  return collectPathToRoot(from, from, path, RootSelector::CLOSEST);
}

FailureOr<SymbolRefAttr> getPathFromRoot(StructDefOp &to) {
  return getPathFromRoot(to, RootSelector::CLOSEST);
}

FailureOr<SymbolRefAttr> getPathFromRoot(FuncOp &to) {
  return getPathFromRoot(to, RootSelector::CLOSEST);
}

FailureOr<ModuleOp> getTopRootModule(Operation *from) {
  std::vector<FlatSymbolRefAttr> path;
  return collectPathToRoot(from, from, path, RootSelector::FURTHEST);
}

FailureOr<SymbolRefAttr> getPathFromTopRoot(StructDefOp &to) {
  return getPathFromRoot(to, RootSelector::FURTHEST);
}

FailureOr<SymbolRefAttr> getPathFromTopRoot(FuncOp &to) {
  return getPathFromRoot(to, RootSelector::FURTHEST);
}

bool hasUsesWithin(Operation *symbol, Operation *from) {
  assert(symbol && "pre-condition");
  assert(from && "pre-condition");
  bool result = false;
  SymbolTable::walkSymbolTables(from, false, [symbol, &result](Operation *symbolTableOp, bool) {
    assert(symbolTableOp->hasTrait<OpTrait::SymbolTable>());
    bool hasUse = (symbol != symbolTableOp) &&
                  !SymbolTable::symbolKnownUseEmpty(symbol, &symbolTableOp->getRegion(0));
    result |= hasUse;
    LLVM_DEBUG({
      if (hasUse) {
        auto uses = SymbolTable::getSymbolUses(symbol, &symbolTableOp->getRegion(0));
        assert(uses.has_value()); // must be consistent with symbolKnownUseEmpty()
        llvm::dbgs() << "Found users of " << *symbol << "\n";
        for (SymbolTable::SymbolUse user : uses.value()) {
          llvm::dbgs() << " * " << *user.getUser() << "\n";
        }
      }
    });
  });
  return result;
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
  if (!llvm::isa<GlobalDefOp>(foundOp)) {
    return origin->emitError() << "ref \"" << param << "\" in type " << parameterizedType
                               << " refers to a '" << foundOp->getName()
                               << "' which is not allowed";
  }
  return success();
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
      if (failed(verifyTypeResolution(tables, origin, typeParam.getValue()))) {
        paramCheckResult = failure();
      }
    }
    // IntegerAttr and AffineMapAttr cannot contain symbol references
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
      return failure(); // verifyParamsOfType() already emits a sufficient error message
    }
  }
  return defForType;
}

LogicalResult verifyTypeResolution(SymbolTableCollection &tables, Operation *origin, Type ty) {
  if (StructType sTy = llvm::dyn_cast<StructType>(ty)) {
    return verifyStructTypeResolution(tables, sTy, origin);
  } else if (ArrayType aTy = llvm::dyn_cast<ArrayType>(ty)) {
    if (failed(verifyParamsOfType(tables, aTy.getDimensionSizes(), aTy, origin))) {
      return failure();
    }
    return verifyTypeResolution(tables, origin, aTy.getElementType());
  } else if (TypeVarType vTy = llvm::dyn_cast<TypeVarType>(ty)) {
    return verifyParamOfType(tables, vTy.getNameRef(), vTy, origin);
  } else {
    return success();
  }
}

} // namespace llzk
