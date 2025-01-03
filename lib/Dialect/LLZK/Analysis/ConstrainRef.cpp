#include "llzk/Dialect/LLZK/Analysis/ConstrainRef.h"

namespace llzk {

mlir::FailureOr<ConstrainRef>
ConstrainRef::translate(const ConstrainRef &prefix, const ConstrainRef &other) const {
  if (isConstant()) {
    return *this;
  }

  if (blockArg != prefix.blockArg || fieldRefs.size() < prefix.fieldRefs.size()) {
    return mlir::failure();
  }
  for (size_t i = 0; i < prefix.fieldRefs.size(); i++) {
    if (fieldRefs[i] != prefix.fieldRefs[i]) {
      return mlir::failure();
    }
  }
  auto newSignalUsage = other;
  for (size_t i = prefix.fieldRefs.size(); i < fieldRefs.size(); i++) {
    newSignalUsage.fieldRefs.push_back(fieldRefs[i]);
  }
  return newSignalUsage;
}

void ConstrainRef::print(mlir::raw_ostream &os) const {
  if (isConstantFelt()) {
    os << "<constfelt: " << getConstantFeltValue() << '>';
  } else if (isConstantIndex()) {
    os << "<index: " << getConstantIndexValue() << '>';
  } else {
    os << "%arg" << blockArg.getArgNumber();
    for (auto f : fieldRefs) {
      os << "[" << f << "]";
    }
  }
}

bool ConstrainRef::operator==(const ConstrainRef &rhs) const {
  return blockArg == rhs.blockArg && fieldRefs == rhs.fieldRefs && constFelt == rhs.constFelt;
}

// required for EquivalenceClasses usage
bool ConstrainRef::operator<(const ConstrainRef &rhs) const {
  if (isConstantFelt() && !rhs.isConstantFelt()) {
    // Put all constants at the end
    return false;
  } else if (!isConstantFelt() && rhs.isConstantFelt()) {
    return true;
  } else if (isConstantFelt() && rhs.isConstantFelt()) {
    return constFelt < rhs.constFelt;
  }

  // both are not constants
  if (blockArg.getArgNumber() < rhs.blockArg.getArgNumber()) {
    return true;
  } else if (blockArg.getArgNumber() > rhs.blockArg.getArgNumber()) {
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
    return OpHash<FeltConstantOp>{}(val.constFelt);
  } else if (val.isConstantIndex()) {
    return OpHash<mlir::index::ConstantOp>{}(val.constIdx);
  } else {
    size_t hash = std::hash<unsigned>{}(val.blockArg.getArgNumber());
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

} // namespace llzk