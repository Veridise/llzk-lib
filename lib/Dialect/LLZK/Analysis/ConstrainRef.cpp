#include "llzk/Dialect/LLZK/Analysis/ConstrainRef.h"

namespace llzk {

mlir::FailureOr<std::vector<ConstrainRef>> ConstrainRef::get(mlir::Value val) {
  std::vector<ConstrainRef> res;

  // If it's a field read, it reads a field def from a component.
  // If it's a felt, it doesn't need a field read

  // Due to the way constrain is defined, all signals are read from inputs.
  if (auto blockArg = mlir::dyn_cast_or_null<mlir::BlockArgument>(val)) {
    // to use this constructor, the block arg must be a felt
    res.push_back(ConstrainRef(blockArg));
  } else if (auto fieldRead = mlir::dyn_cast_or_null<FieldReadOp>(val.getDefiningOp())) {
    std::deque<FieldDefOp> fields;
    mlir::SymbolTableCollection tables;
    mlir::BlockArgument arg;
    FieldReadOp currRead = fieldRead;
    while (currRead != nullptr) {
      auto component = currRead.getComponent();
      auto fieldOpRes = currRead.getFieldDefOp(tables);
      if (mlir::failed(fieldOpRes)) {
        fieldRead.emitError() << "could not find field read\n";
        return mlir::failure();
      }
      fields.push_front(fieldOpRes->get());
      arg = mlir::dyn_cast_or_null<mlir::BlockArgument>(component);
      currRead = mlir::dyn_cast_or_null<FieldReadOp>(component.getDefiningOp());
    }
    if (arg == nullptr) {
      fieldRead.emitError() << "could not follow a read chain!\n";
      return mlir::failure();
    }
    // We only want to generate this if the end value is a felt
    res.emplace_back(arg, std::vector<ConstrainRefIndex>(fields.begin(), fields.end()));
  } else if (val.getDefiningOp() != nullptr && mlir::isa<FeltConstantOp>(val.getDefiningOp())) {
    auto constFelt = mlir::dyn_cast<FeltConstantOp>(val.getDefiningOp());
    res.emplace_back(constFelt);
  } else if (val.getDefiningOp() != nullptr) {
    // Fallback for every other type of operation
    // This also works for global func call ops, since we use an interprocedural dataflow solver
    for (auto operand : val.getDefiningOp()->getOperands()) {
      auto uses = ConstrainRef::get(operand);
      if (mlir::succeeded(uses)) {
        res.insert(res.end(), uses->begin(), uses->end());
      }
    }
  } else {
    std::string str;
    llvm::raw_string_ostream ss(str);
    ss << val;
    llvm::report_fatal_error("unsupported value in SignalUsage::get: " + mlir::Twine(ss.str()));
  }

  if (res.empty()) {
    return mlir::failure();
  }
  return res;
}

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
  if (isConstant()) {
    os << "<constfelt: " << const_cast<FeltConstantOp &>(constFelt).getValueAttr().getValue()
       << ">";
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
  if (isConstant() && !rhs.isConstant()) {
    // Put all constants at the end
    return false;
  } else if (!isConstant() && rhs.isConstant()) {
    return true;
  } else if (isConstant() && rhs.isConstant()) {
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
  if (val.isConstant()) {
    return OpHash<FeltConstantOp>{}(val.constFelt);
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