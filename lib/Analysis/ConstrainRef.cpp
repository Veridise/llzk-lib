//===-- ConstraintRef.cpp - ConstrainRef implementation ---------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Analysis/ConstrainRef.h"
#include "llzk/Dialect/Array/IR/Ops.h"
#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Dialect/String/IR/Types.h"
#include "llzk/Transforms/LLZKLoweringUtils.h"
#include "llzk/Util/APIntHelper.h"
#include "llzk/Util/Compare.h"
#include "llzk/Util/Debug.h"
#include "llzk/Util/SymbolHelper.h"
#include "llzk/Util/SymbolLookup.h"

using namespace mlir;

namespace llzk {

using namespace array;
using namespace component;
using namespace felt;
using namespace function;
using namespace polymorphic;
using namespace string;

/* ConstrainRefIndex */

void ConstrainRefIndex::print(mlir::raw_ostream &os) const {
  if (isField()) {
    os << '@' << getField().getName();
  } else if (isIndex()) {
    os << getIndex();
  } else {
    auto r = getIndexRange();
    os << std::get<0>(r) << ':' << std::get<1>(r);
  }
}

bool ConstrainRefIndex::operator<(const ConstrainRefIndex &rhs) const {
  if (isField() && rhs.isField()) {
    return NamedOpLocationLess<FieldDefOp> {}(getField(), rhs.getField());
  }
  if (isIndex() && rhs.isIndex()) {
    return safeLt(APSInt(getIndex()), APSInt(rhs.getIndex()));
  }
  if (isIndexRange() && rhs.isIndexRange()) {
    auto l = getIndexRange(), r = rhs.getIndexRange();
    auto ll = APSInt(std::get<0>(l)), lu = APSInt(std::get<1>(l));
    auto rl = APSInt(std::get<0>(r)), ru = APSInt(std::get<1>(r));
    return safeLt(ll, rl) || (safeEq(ll, rl) && safeLt(lu, ru));
  }

  if (isField()) {
    return true;
  }
  if (isIndex() && !rhs.isField()) {
    return true;
  }

  return false;
}

size_t ConstrainRefIndex::Hash::operator()(const ConstrainRefIndex &c) const {
  if (c.isIndex()) {
    return llvm::hash_value(c.getIndex());
  } else if (c.isIndexRange()) {
    auto r = c.getIndexRange();
    return llvm::hash_value(std::get<0>(r)) ^ llvm::hash_value(std::get<1>(r));
  } else {
    return OpHash<component::FieldDefOp> {}(c.getField());
  }
}

/* ConstrainRef */

/// @brief Lookup a `StructDefOp` from a given `StructType`.
/// @param tables
/// @param mod
/// @param ty
/// @return A `SymbolLookupResult` for the `StructDefOp` found. Note that returning the
/// lookup result is important, as it may manage a ModuleOp if the struct is found
/// via an include.
SymbolLookupResult<StructDefOp>
getStructDef(mlir::SymbolTableCollection &tables, mlir::ModuleOp mod, StructType ty) {
  auto sDef = ty.getDefinition(tables, mod);
  ensure(
      mlir::succeeded(sDef),
      "could not find '" + StructDefOp::getOperationName() + "' op from struct type"
  );

  return std::move(sDef.value());
}

std::vector<ConstrainRef> ConstrainRef::getAllConstrainRefs(
    mlir::SymbolTableCollection &tables, mlir::ModuleOp mod, ArrayType arrayTy, ConstrainRef root
) {
  std::vector<ConstrainRef> res;
  // Add root item
  res.push_back(root);

  // Recurse into arrays by iterating over their elements
  int64_t maxSz = arrayTy.getDimSize(0);
  for (int64_t i = 0; i < maxSz; i++) {
    auto elemTy = arrayTy.getElementType();

    ConstrainRef childRef = root.createChild(ConstrainRefIndex(i));

    if (auto arrayElemTy = mlir::dyn_cast<ArrayType>(elemTy)) {
      // recurse
      auto subRes = getAllConstrainRefs(tables, mod, arrayElemTy, childRef);
      res.insert(res.end(), subRes.begin(), subRes.end());
    } else if (auto structTy = mlir::dyn_cast<StructType>(elemTy)) {
      // recurse into struct def
      auto subRes = getAllConstrainRefs(tables, mod, getStructDef(tables, mod, structTy), childRef);
      res.insert(res.end(), subRes.begin(), subRes.end());
    } else {
      // scalar type
      res.push_back(childRef);
    }
  }

  return res;
}

std::vector<ConstrainRef> ConstrainRef::getAllConstrainRefs(
    mlir::SymbolTableCollection &tables, mlir::ModuleOp mod,
    SymbolLookupResult<StructDefOp> structDefRes, ConstrainRef root
) {
  std::vector<ConstrainRef> res;
  // Add root item
  res.emplace_back(root);
  // Recurse into struct types by iterating over all their field definitions
  for (auto f : structDefRes.get().getOps<FieldDefOp>()) {
    // We want to store the FieldDefOp, but without the possibility of accidentally dropping the
    // reference, so we need to re-lookup the symbol to create a SymbolLookupResult, which will
    // manage the external module containing the field defs, if needed.
    // TODO: It would be nice if we could manage module op references differently
    // so we don't have to do this.
    auto structDefCopy = structDefRes;
    auto fieldLookup = lookupSymbolIn<FieldDefOp>(
        tables, mlir::SymbolRefAttr::get(f.getContext(), f.getSymNameAttr()),
        std::move(structDefCopy), mod.getOperation()
    );
    ensure(mlir::succeeded(fieldLookup), "could not get SymbolLookupResult of existing FieldDefOp");
    ConstrainRef childRef = root.createChild(ConstrainRefIndex(fieldLookup.value()));
    // Make a reference to the current field, regardless of if it is a composite
    // type or not.
    res.push_back(childRef);
    if (auto structTy = mlir::dyn_cast<StructType>(f.getType())) {
      // Create refs for each field
      auto subRes = getAllConstrainRefs(tables, mod, getStructDef(tables, mod, structTy), childRef);
      res.insert(res.end(), subRes.begin(), subRes.end());
    } else if (auto arrayTy = mlir::dyn_cast<ArrayType>(f.getType())) {
      // Create refs for each array element
      auto subRes = getAllConstrainRefs(tables, mod, arrayTy, childRef);
      res.insert(res.end(), subRes.begin(), subRes.end());
    }
  }
  return res;
}

std::vector<ConstrainRef> ConstrainRef::getAllConstrainRefs(
    mlir::SymbolTableCollection &tables, mlir::ModuleOp mod, ConstrainRef root
) {
  auto ty = root.getType();
  std::vector<ConstrainRef> res;
  if (auto structTy = mlir::dyn_cast<StructType>(ty)) {
    // recurse over fields
    res = getAllConstrainRefs(tables, mod, getStructDef(tables, mod, structTy), root);
  } else if (auto arrayType = mlir::dyn_cast<ArrayType>(ty)) {
    res = getAllConstrainRefs(tables, mod, arrayType, root);
  } else if (mlir::isa<FeltType, IndexType, StringType>(ty)) {
    // Scalar type
    res.emplace_back(root);
  } else {
    std::string err;
    debug::Appender(err) << "unsupported type: " << ty;
    llvm::report_fatal_error(mlir::Twine(err));
  }
  return res;
}

std::vector<ConstrainRef> ConstrainRef::getAllConstrainRefs(StructDefOp structDef, FuncDefOp fnOp) {
  std::vector<ConstrainRef> res;

  ensure(
      structDef == fnOp->getParentOfType<StructDefOp>(), "function must be within the given struct"
  );

  FailureOr<ModuleOp> modOp = getRootModule(structDef);
  ensure(
      mlir::succeeded(modOp),
      "could not lookup module from struct " + mlir::Twine(structDef.getName())
  );

  mlir::SymbolTableCollection tables;
  for (auto a : fnOp.getArguments()) {
    auto argRes = getAllConstrainRefs(tables, modOp.value(), ConstrainRef(a));
    res.insert(res.end(), argRes.begin(), argRes.end());
  }

  // For compute functions, the "self" field is not arg0 like for constrain, but
  // rather the struct value returned from the function.
  if (fnOp.isStructCompute()) {
    Value selfVal = getSelfValueFromCompute(fnOp);
    auto createOp = dyn_cast_if_present<CreateStructOp>(selfVal.getDefiningOp());
    ensure(createOp, "self value should originate from struct.new operation");
    auto selfRes = getAllConstrainRefs(tables, modOp.value(), ConstrainRef(createOp));
    res.insert(res.end(), selfRes.begin(), selfRes.end());
  }

  return res;
}

std::vector<ConstrainRef>
ConstrainRef::getAllConstrainRefs(StructDefOp structDef, FieldDefOp fieldDef) {
  std::vector<ConstrainRef> res;
  FuncDefOp constrainFnOp = structDef.getConstrainFuncOp();
  ensure(
      fieldDef->getParentOfType<StructDefOp>() == structDef,
      "Field " + mlir::Twine(fieldDef.getName()) + " is not a field of struct " +
          mlir::Twine(structDef.getName())
  );
  FailureOr<ModuleOp> modOp = getRootModule(structDef);
  ensure(
      mlir::succeeded(modOp),
      "could not lookup module from struct " + mlir::Twine(structDef.getName())
  );

  // Get the self argument
  BlockArgument self = constrainFnOp.getBody().getArgument(0);
  ConstrainRef fieldRef = ConstrainRef(self, {ConstrainRefIndex(fieldDef)});

  mlir::SymbolTableCollection tables;
  return getAllConstrainRefs(tables, modOp.value(), fieldRef);
}

mlir::Type ConstrainRef::getType() const {
  if (isConstantFelt()) {
    return std::get<FeltConstantOp>(*constantVal).getType();
  } else if (isConstantIndex()) {
    return std::get<mlir::arith::ConstantIndexOp>(*constantVal).getType();
  } else if (isTemplateConstant()) {
    return std::get<ConstReadOp>(*constantVal).getType();
  } else {
    int array_derefs = 0;
    int idx = fieldRefs.size() - 1;
    while (idx >= 0 && fieldRefs[idx].isIndex()) {
      array_derefs++;
      idx--;
    }

    if (idx >= 0) {
      mlir::Type currTy = fieldRefs[idx].getField().getType();
      while (array_derefs > 0) {
        currTy = mlir::dyn_cast<ArrayType>(currTy).getElementType();
        array_derefs--;
      }
      return currTy;
    }

    return isBlockArgument() ? getBlockArgument().getType() : getCreateStructOp().getType();
  }
}

bool ConstrainRef::isValidPrefix(const ConstrainRef &prefix) const {
  if (isConstant()) {
    return false;
  }

  if (root != prefix.root || fieldRefs.size() < prefix.fieldRefs.size()) {
    return false;
  }
  for (size_t i = 0; i < prefix.fieldRefs.size(); i++) {
    if (fieldRefs[i] != prefix.fieldRefs[i]) {
      return false;
    }
  }
  return true;
}

mlir::FailureOr<std::vector<ConstrainRefIndex>> ConstrainRef::getSuffix(const ConstrainRef &prefix
) const {
  if (!isValidPrefix(prefix)) {
    return mlir::failure();
  }
  std::vector<ConstrainRefIndex> suffix;
  for (size_t i = prefix.fieldRefs.size(); i < fieldRefs.size(); i++) {
    suffix.push_back(fieldRefs[i]);
  }
  return suffix;
}

mlir::FailureOr<ConstrainRef>
ConstrainRef::translate(const ConstrainRef &prefix, const ConstrainRef &other) const {
  if (isConstant()) {
    return *this;
  }
  auto suffix = getSuffix(prefix);
  if (mlir::failed(suffix)) {
    return mlir::failure();
  }

  auto newSignalUsage = other;
  newSignalUsage.fieldRefs.insert(newSignalUsage.fieldRefs.end(), suffix->begin(), suffix->end());
  return newSignalUsage;
}

void ConstrainRef::print(mlir::raw_ostream &os) const {
  if (isConstantFelt()) {
    os << "<felt.const: " << getConstantFeltValue() << '>';
  } else if (isConstantIndex()) {
    os << "<index: " << getConstantIndexValue() << '>';
  } else if (isTemplateConstant()) {
    auto constRead = std::get<ConstReadOp>(*constantVal);
    auto structDefOp = constRead->getParentOfType<StructDefOp>();
    ensure(structDefOp, "struct template should have a struct parent");
    os << '@' << structDefOp.getName() << "<[@" << constRead.getConstName() << "]>";
  } else {
    if (isCreateStructOp()) {
      os << "%self";
    } else {
      ensure(isBlockArgument(), "unhandled print case");
      os << "%arg" << getInputNum();
    }

    for (auto f : fieldRefs) {
      os << "[" << f << "]";
    }
  }
}

bool ConstrainRef::operator==(const ConstrainRef &rhs) const {
  // This way two felt constants can be equal even if the declared in separate ops.
  if (isConstantInt() && rhs.isConstantInt()) {
    return getConstantValue() == rhs.getConstantValue();
  }
  return (root == rhs.root) && (fieldRefs == rhs.fieldRefs) && (constantVal == rhs.constantVal);
}

// required for EquivalenceClasses usage
bool ConstrainRef::operator<(const ConstrainRef &rhs) const {
  if (isConstantFelt() && !rhs.isConstantFelt()) {
    // Put all constants at the end
    return false;
  } else if (!isConstantFelt() && rhs.isConstantFelt()) {
    return true;
  } else if (isConstantFelt() && rhs.isConstantFelt()) {
    auto lhsInt = getConstantFeltValue();
    auto rhsInt = rhs.getConstantFeltValue();
    auto bitWidthMax = std::max(lhsInt.getBitWidth(), rhsInt.getBitWidth());
    return lhsInt.zext(bitWidthMax).ult(rhsInt.zext(bitWidthMax));
  }

  if (isConstantIndex() && !rhs.isConstantIndex()) {
    // Put all constant indices next at the end
    return false;
  } else if (!isConstantIndex() && rhs.isConstantIndex()) {
    return true;
  } else if (isConstantIndex() && rhs.isConstantIndex()) {
    return getConstantIndexValue().ult(rhs.getConstantIndexValue());
  }

  if (isTemplateConstant() && !rhs.isTemplateConstant()) {
    // Put all template constants next at the end
    return false;
  } else if (!isTemplateConstant() && rhs.isTemplateConstant()) {
    return true;
  } else if (isTemplateConstant() && rhs.isTemplateConstant()) {
    auto lhsName = std::get<ConstReadOp>(*constantVal).getConstName();
    auto rhsName = std::get<ConstReadOp>(*rhs.constantVal).getConstName();
    return lhsName.compare(rhsName) < 0;
  }

  // Sort out the block argument vs struct.new cases
  if (isBlockArgument() && rhs.isCreateStructOp()) {
    return true;
  } else if (isCreateStructOp() && rhs.isBlockArgument()) {
    return false;
  } else if (isBlockArgument() && rhs.isBlockArgument()) {
    if (getInputNum() < rhs.getInputNum()) {
      return true;
    } else if (getInputNum() > rhs.getInputNum()) {
      return false;
    }
  } else if (isCreateStructOp() && rhs.isCreateStructOp()) {
    CreateStructOp lhsOp = getCreateStructOp(), rhsOp = rhs.getCreateStructOp();
    if (lhsOp < rhsOp) {
      return true;
    } else if (lhsOp > rhsOp) {
      return false;
    }
  } else {
    llvm_unreachable("unhandled operator< case");
  }

  for (size_t i = 0; i < fieldRefs.size() && i < rhs.fieldRefs.size(); i++) {
    if (fieldRefs[i] < rhs.fieldRefs[i]) {
      return true;
    } else if (fieldRefs[i] > rhs.fieldRefs[i]) {
      return false;
    }
  }
  return fieldRefs.size() < rhs.fieldRefs.size();
}

size_t ConstrainRef::Hash::operator()(const ConstrainRef &val) const {
  if (val.isConstantFelt()) {
    return OpHash<FeltConstantOp> {}(std::get<FeltConstantOp>(*val.constantVal));
  } else if (val.isConstantIndex()) {
    return OpHash<mlir::arith::ConstantIndexOp> {
    }(std::get<mlir::arith::ConstantIndexOp>(*val.constantVal));
  } else if (val.isTemplateConstant()) {
    return OpHash<ConstReadOp> {}(std::get<ConstReadOp>(*val.constantVal));
  } else {
    ensure(val.isBlockArgument() || val.isCreateStructOp(), "unhandled ConstrainRef hash case");

    size_t hash = val.isBlockArgument() ? std::hash<unsigned> {}(val.getInputNum())
                                        : OpHash<CreateStructOp> {}(val.getCreateStructOp());
    for (auto f : val.fieldRefs) {
      hash ^= f.getHash();
    }
    return hash;
  }
}

mlir::raw_ostream &operator<<(mlir::raw_ostream &os, const ConstrainRef &rhs) {
  rhs.print(os);
  return os;
}

/* ConstrainRefSet */

ConstrainRefSet &ConstrainRefSet::join(const ConstrainRefSet &rhs) {
  insert(rhs.begin(), rhs.end());
  return *this;
}

mlir::raw_ostream &operator<<(mlir::raw_ostream &os, const ConstrainRefSet &rhs) {
  os << "{ ";
  std::vector<ConstrainRef> sortedRefs(rhs.begin(), rhs.end());
  std::sort(sortedRefs.begin(), sortedRefs.end());
  for (auto it = sortedRefs.begin(); it != sortedRefs.end();) {
    os << *it;
    it++;
    if (it != sortedRefs.end()) {
      os << ", ";
    } else {
      os << ' ';
    }
  }
  os << '}';
  return os;
}

} // namespace llzk
