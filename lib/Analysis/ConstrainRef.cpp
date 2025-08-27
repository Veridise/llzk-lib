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

void ConstrainRefIndex::print(raw_ostream &os) const {
  if (isField()) {
    os << '@' << getField().getName();
  } else if (isIndex()) {
    os << getIndex();
  } else {
    auto [low, high] = getIndexRange();
    if (ShapedType::isDynamic(high.getSExtValue())) {
      os << "<dynamic>";
    } else {
      os << low << ':' << high;
    }
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
    // We don't hash the index directly, because the built-in LLVM hash includes
    // the bitwidth of the APInt in the hash, which is undesirable for this application.
    // i.e., We want a N-bit version of x to hash to the same value as an M-bit version of X,
    // because our equality checks would consider them equal regardless of bitwidth.
    APInt idx = c.getIndex();
    unsigned requiredBits = idx.getSignificantBits();
    auto hash = llvm::hash_value(idx.trunc(requiredBits));
    return hash;
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
getStructDef(SymbolTableCollection &tables, ModuleOp mod, StructType ty) {
  auto sDef = ty.getDefinition(tables, mod);
  ensure(
      succeeded(sDef),
      "could not find '" + StructDefOp::getOperationName() + "' op from struct type"
  );

  return std::move(sDef.value());
}

std::vector<ConstrainRef>
ConstrainRef::getAllConstrainRefs(SymbolTableCollection &tables, ModuleOp mod, ConstrainRef root) {
  std::vector<ConstrainRef> res = {root};
  for (const ConstrainRef &child : root.getAllChildren(tables, mod)) {
    auto recursiveChildren = getAllConstrainRefs(tables, mod, child);
    res.insert(res.end(), recursiveChildren.begin(), recursiveChildren.end());
  }
  return res;
}

std::vector<ConstrainRef> ConstrainRef::getAllConstrainRefs(StructDefOp structDef, FuncDefOp fnOp) {
  std::vector<ConstrainRef> res;

  ensure(
      structDef == fnOp->getParentOfType<StructDefOp>(), "function must be within the given struct"
  );

  FailureOr<ModuleOp> modOp = getRootModule(structDef);
  ensure(succeeded(modOp), "could not lookup module from struct " + Twine(structDef.getName()));

  SymbolTableCollection tables;
  for (auto a : fnOp.getArguments()) {
    auto argRes = getAllConstrainRefs(tables, modOp.value(), ConstrainRef(a));
    res.insert(res.end(), argRes.begin(), argRes.end());
  }

  // For compute functions, the "self" field is not arg0 like for constrain, but
  // rather the struct value returned from the function.
  if (fnOp.isStructCompute()) {
    Value selfVal = fnOp.getSelfValueFromCompute();
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
      fieldDef->getParentOfType<StructDefOp>() == structDef, "Field " + Twine(fieldDef.getName()) +
                                                                 " is not a field of struct " +
                                                                 Twine(structDef.getName())
  );
  FailureOr<ModuleOp> modOp = getRootModule(structDef);
  ensure(succeeded(modOp), "could not lookup module from struct " + Twine(structDef.getName()));

  // Get the self argument (like `FuncDefOp::getSelfValueFromConstrain()`)
  BlockArgument self = constrainFnOp.getArguments().front();
  ConstrainRef fieldRef = ConstrainRef(self, {ConstrainRefIndex(fieldDef)});

  SymbolTableCollection tables;
  return getAllConstrainRefs(tables, modOp.value(), fieldRef);
}

Type ConstrainRef::getType() const {
  if (isConstantFelt()) {
    return std::get<FeltConstantOp>(*constantVal).getType();
  } else if (isConstantIndex()) {
    return std::get<arith::ConstantIndexOp>(*constantVal).getType();
  } else if (isTemplateConstant()) {
    return std::get<ConstReadOp>(*constantVal).getType();
  } else {
    int array_derefs = 0;
    int idx = fieldRefs.size() - 1;
    while (idx >= 0 && fieldRefs[idx].isIndex()) {
      array_derefs++;
      idx--;
    }

    Type currTy = nullptr;
    if (idx >= 0) {
      currTy = fieldRefs[idx].getField().getType();
    } else {
      currTy = isBlockArgument() ? getBlockArgument().getType() : getCreateStructOp().getType();
    }

    while (array_derefs > 0) {
      currTy = dyn_cast<ArrayType>(currTy).getElementType();
      array_derefs--;
    }
    return currTy;
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

FailureOr<std::vector<ConstrainRefIndex>> ConstrainRef::getSuffix(const ConstrainRef &prefix
) const {
  if (!isValidPrefix(prefix)) {
    return failure();
  }
  std::vector<ConstrainRefIndex> suffix;
  for (size_t i = prefix.fieldRefs.size(); i < fieldRefs.size(); i++) {
    suffix.push_back(fieldRefs[i]);
  }
  return suffix;
}

FailureOr<ConstrainRef>
ConstrainRef::translate(const ConstrainRef &prefix, const ConstrainRef &other) const {
  if (isConstant()) {
    return *this;
  }
  auto suffix = getSuffix(prefix);
  if (failed(suffix)) {
    return failure();
  }

  auto newSignalUsage = other;
  newSignalUsage.fieldRefs.insert(newSignalUsage.fieldRefs.end(), suffix->begin(), suffix->end());
  return newSignalUsage;
}

std::vector<ConstrainRef>
getAllChildren(SymbolTableCollection &tables, ModuleOp mod, ArrayType arrayTy, ConstrainRef root) {
  std::vector<ConstrainRef> res;
  // Recurse into arrays by iterating over their elements
  for (int64_t i = 0; i < arrayTy.getDimSize(0); i++) {
    ConstrainRef childRef = root.createChild(ConstrainRefIndex(i));
    res.push_back(childRef);
  }

  return res;
}

std::vector<ConstrainRef> getAllChildren(
    SymbolTableCollection &tables, ModuleOp mod, SymbolLookupResult<StructDefOp> structDefRes,
    ConstrainRef root
) {
  std::vector<ConstrainRef> res;
  // Recurse into struct types by iterating over all their field definitions
  for (auto f : structDefRes.get().getOps<FieldDefOp>()) {
    // We want to store the FieldDefOp, but without the possibility of accidentally dropping the
    // reference, so we need to re-lookup the symbol to create a SymbolLookupResult, which will
    // manage the external module containing the field defs, if needed.
    // TODO: It would be nice if we could manage module op references differently
    // so we don't have to do this.
    auto structDefCopy = structDefRes;
    auto fieldLookup = lookupSymbolIn<FieldDefOp>(
        tables, SymbolRefAttr::get(f.getContext(), f.getSymNameAttr()), std::move(structDefCopy),
        mod.getOperation()
    );
    ensure(succeeded(fieldLookup), "could not get SymbolLookupResult of existing FieldDefOp");
    ConstrainRef childRef = root.createChild(ConstrainRefIndex(fieldLookup.value()));
    // Make a reference to the current field, regardless of if it is a composite
    // type or not.
    res.push_back(childRef);
  }
  return res;
}

std::vector<ConstrainRef>
ConstrainRef::getAllChildren(SymbolTableCollection &tables, ModuleOp mod) const {
  auto ty = getType();
  if (auto structTy = dyn_cast<StructType>(ty)) {
    return llzk::getAllChildren(tables, mod, getStructDef(tables, mod, structTy), *this);
  } else if (auto arrayType = dyn_cast<ArrayType>(ty)) {
    return llzk::getAllChildren(tables, mod, arrayType, *this);
  }
  // Scalar type, no children
  return {};
}

void ConstrainRef::print(raw_ostream &os) const {
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
    APSInt lhsVal(getConstantValue()), rhsVal(rhs.getConstantValue());
    return getType() == rhs.getType() && safeEq(lhsVal, rhsVal);
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
    APSInt lhsInt(getConstantFeltValue()), rhsInt(rhs.getConstantFeltValue());
    return safeLt(lhsInt, rhsInt);
  }

  if (isConstantIndex() && !rhs.isConstantIndex()) {
    // Put all constant indices next at the end
    return false;
  } else if (!isConstantIndex() && rhs.isConstantIndex()) {
    return true;
  } else if (isConstantIndex() && rhs.isConstantIndex()) {
    APSInt lhsVal(getConstantIndexValue()), rhsVal(rhs.getConstantIndexValue());
    return safeLt(lhsVal, rhsVal);
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
  if (val.isConstantInt()) {
    return llvm::hash_value(val.getConstantValue());
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

raw_ostream &operator<<(raw_ostream &os, const ConstrainRef &rhs) {
  rhs.print(os);
  return os;
}

/* ConstrainRefSet */

ConstrainRefSet &ConstrainRefSet::join(const ConstrainRefSet &rhs) {
  insert(rhs.begin(), rhs.end());
  return *this;
}

raw_ostream &operator<<(raw_ostream &os, const ConstrainRefSet &rhs) {
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
