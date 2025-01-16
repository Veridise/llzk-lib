#include "llzk/Dialect/LLZK/Analysis/ConstrainRef.h"
#include "llzk/Dialect/LLZK/Util/Compare.h"
#include "llzk/Dialect/LLZK/Util/Debug.h"
#include "llzk/Dialect/LLZK/Util/SymbolHelper.h"
#include "llzk/Dialect/LLZK/Util/SymbolLookup.h"

namespace llzk {

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
    return getIndex().ult(rhs.getIndex());
  }
  if (isIndexRange() && rhs.isIndexRange()) {
    auto l = getIndexRange(), r = rhs.getIndexRange();
    auto ll = std::get<0>(l), lu = std::get<1>(l);
    auto rl = std::get<0>(r), ru = std::get<1>(r);
    return ll.ult(rl) || (ll == rl && lu.ult(ru));
  }

  if (isField()) {
    return true;
  }
  if (isIndex() && !rhs.isField()) {
    return true;
  }

  return false;
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
  debug::ensure(mlir::succeeded(sDef), "could not find struct definition from struct type");

  return std::move(sDef.value());
}

std::vector<ConstrainRef> ConstrainRef::getAllConstrainRefs(
    mlir::SymbolTableCollection &tables, mlir::ModuleOp mod, ArrayType arrayTy,
    mlir::BlockArgument blockArg, std::vector<ConstrainRefIndex> fields = {}
) {
  std::vector<ConstrainRef> res;
  // Add root item
  res.emplace_back(blockArg, fields);

  // Recurse into arrays by iterating over their elements
  int64_t maxSz = arrayTy.getDimSize(0);
  for (int64_t i = 0; i < maxSz; i++) {
    auto elemTy = arrayTy.getElementType();

    std::vector<ConstrainRefIndex> subFields = fields;
    subFields.emplace_back(i);

    if (auto arrayElemTy = mlir::dyn_cast<ArrayType>(elemTy)) {
      // recurse
      auto subRes = getAllConstrainRefs(tables, mod, arrayElemTy, blockArg, subFields);
      res.insert(res.end(), subRes.begin(), subRes.end());
    } else if (auto structTy = mlir::dyn_cast<StructType>(elemTy)) {
      // recurse into struct def
      auto subRes = getAllConstrainRefs(
          tables, mod, getStructDef(tables, mod, structTy), blockArg, subFields
      );
      res.insert(res.end(), subRes.begin(), subRes.end());
    } else {
      // scalar type
      res.emplace_back(blockArg, subFields);
    }
  }

  return res;
}

std::vector<ConstrainRef> ConstrainRef::getAllConstrainRefs(
    mlir::SymbolTableCollection &tables, mlir::ModuleOp mod,
    SymbolLookupResult<StructDefOp> structDefRes, mlir::BlockArgument blockArg,
    std::vector<ConstrainRefIndex> fields = {}
) {
  std::vector<ConstrainRef> res;
  // Add root item
  res.emplace_back(blockArg, fields);
  // Recurse into struct types by iterating over all their field definitions
  for (auto f : structDefRes.get().getOps<FieldDefOp>()) {
    std::vector<ConstrainRefIndex> subFields = fields;
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
    debug::ensure(
        mlir::succeeded(fieldLookup), "could not get SymbolLookupResult of existing FieldDefOp"
    );
    subFields.emplace_back(fieldLookup.value());
    // Make a reference to the current field, regardless of if it is a composite
    // type or not.
    res.emplace_back(blockArg, subFields);
    if (auto structTy = mlir::dyn_cast<llzk::StructType>(f.getType())) {
      // Create refs for each field
      auto subRes = getAllConstrainRefs(
          tables, mod, getStructDef(tables, mod, structTy), blockArg, subFields
      );
      res.insert(res.end(), subRes.begin(), subRes.end());
    } else if (auto arrayTy = mlir::dyn_cast<llzk::ArrayType>(f.getType())) {
      // Create refs for each array element
      auto subRes = getAllConstrainRefs(tables, mod, arrayTy, blockArg, subFields);
      res.insert(res.end(), subRes.begin(), subRes.end());
    }
  }
  return res;
}

std::vector<ConstrainRef> ConstrainRef::getAllConstrainRefs(
    mlir::SymbolTableCollection &tables, mlir::ModuleOp mod, mlir::BlockArgument arg
) {
  auto ty = arg.getType();
  std::vector<ConstrainRef> res;
  if (auto structTy = mlir::dyn_cast<llzk::StructType>(ty)) {
    // recurse over fields
    res = getAllConstrainRefs(tables, mod, getStructDef(tables, mod, structTy), arg);
  } else if (auto arrayType = mlir::dyn_cast<llzk::ArrayType>(ty)) {
    res = getAllConstrainRefs(tables, mod, arrayType, arg);
  } else if (mlir::isa<llzk::FeltType>(ty) || mlir::isa<mlir::IndexType>(ty)) {
    // Scalar type
    res.emplace_back(arg);
  } else {
    std::string msg = "unsupported type: ";
    llvm::raw_string_ostream ss(msg);
    ss << ty;
    llvm::report_fatal_error(ss.str().c_str());
  }
  return res;
}

std::vector<ConstrainRef> ConstrainRef::getAllConstrainRefs(StructDefOp structDef) {
  std::vector<ConstrainRef> res;
  auto constrainFnOp = structDef.getConstrainFuncOp();
  debug::ensure(
      constrainFnOp,
      "malformed struct " + mlir::Twine(structDef.getName()) + " must define a constrain function"
  );

  auto modOp = getRootModule(structDef);
  debug::ensure(
      mlir::succeeded(modOp),
      "could not lookup module from struct " + mlir::Twine(structDef.getName())
  );

  mlir::SymbolTableCollection tables;
  for (auto a : constrainFnOp.getArguments()) {
    auto argRes = getAllConstrainRefs(tables, modOp.value(), a);
    res.insert(res.end(), argRes.begin(), argRes.end());
  }
  return res;
}

mlir::Type ConstrainRef::getType() const {
  if (isConstantFelt()) {
    return std::get<FeltConstantOp>(*constantVal).getType();
  } else if (isConstantIndex()) {
    return std::get<mlir::index::ConstantOp>(*constantVal).getType();
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
    } else {
      return blockArg.getType();
    }
  }
}

bool ConstrainRef::isValidPrefix(const ConstrainRef &prefix) const {
  if (isConstant()) {
    return false;
  }

  if (blockArg != prefix.blockArg || fieldRefs.size() < prefix.fieldRefs.size()) {
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
    os << "<constfelt: " << getConstantFeltValue() << '>';
  } else if (isConstantIndex()) {
    os << "<index: " << getConstantIndexValue() << '>';
  } else if (isTemplateConstant()) {
    os << "<template: " << std::get<ConstReadOp>(*constantVal).getConstName() << '>';
  } else {
    debug::ensure(isBlockArgument(), "unhandled print case");
    os << "%arg" << getInputNum();
    for (auto f : fieldRefs) {
      os << "[" << f << "]";
    }
  }
}

bool ConstrainRef::operator==(const ConstrainRef &rhs) const {
  return blockArg == rhs.blockArg && fieldRefs == rhs.fieldRefs && constantVal == rhs.constantVal;
}

// required for EquivalenceClasses usage
bool ConstrainRef::operator<(const ConstrainRef &rhs) const {
  if (isConstantFelt() && !rhs.isConstantFelt()) {
    // Put all constants at the end
    return false;
  } else if (!isConstantFelt() && rhs.isConstantFelt()) {
    return true;
  } else if (isConstantFelt() && rhs.isConstantFelt()) {
    return getConstantFeltValue().ult(rhs.getConstantFeltValue());
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

  // both are not constants
  debug::ensure(isBlockArgument() && rhs.isBlockArgument(), "unhandled operator< case");
  if (getInputNum() < rhs.getInputNum()) {
    return true;
  } else if (getInputNum() > rhs.getInputNum()) {
    return false;
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
    return OpHash<mlir::index::ConstantOp> {}(std::get<mlir::index::ConstantOp>(*val.constantVal));
  } else if (val.isTemplateConstant()) {
    return OpHash<ConstReadOp> {}(std::get<ConstReadOp>(*val.constantVal));
  } else {
    debug::ensure(val.isBlockArgument(), "unhandled operator() case");

    size_t hash = std::hash<unsigned> {}(val.getInputNum());
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
