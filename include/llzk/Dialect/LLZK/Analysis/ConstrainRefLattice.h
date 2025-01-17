#pragma once

#include "llzk/Dialect/LLZK/Analysis/ConstrainRef.h"

namespace llzk {

using TranslationMap =
    std::unordered_map<ConstrainRef, ConstrainRefLatticeValue, ConstrainRef::Hash>;

/// @brief A value at a given point of the ConstrainRefLattice.
class ConstrainRefLatticeValue {
  /// For scalar values.
  using ScalarTy = ConstrainRefSet;
  /// For arrays of values created by, e.g., the LLZK new_array op. A recursive
  /// definition to support arrays of arbitrary dimensions.
  /// Unique pointers are used as each value must be self contained for the
  /// sake of consistent translations.
  /// This array is flattened.
  using ArrayTy = std::vector<std::unique_ptr<ConstrainRefLatticeValue>>;

  static ArrayTy constructArrayTy(const mlir::ArrayRef<int64_t> &shape);

public:
  explicit ConstrainRefLatticeValue(ScalarTy s) : value(s), arrayShape(std::nullopt) {}
  explicit ConstrainRefLatticeValue(ConstrainRef r) : ConstrainRefLatticeValue(ScalarTy {r}) {}
  ConstrainRefLatticeValue() : ConstrainRefLatticeValue(ScalarTy {}) {}

  // Create an empty array of the given shape.
  explicit ConstrainRefLatticeValue(mlir::ArrayRef<int64_t> shape)
      : value(constructArrayTy(shape)), arrayShape(shape) {}

  // Enable copying by duplicating unique_ptrs
  ConstrainRefLatticeValue(const ConstrainRefLatticeValue &rhs) { *this = rhs; }

  ConstrainRefLatticeValue &operator=(const ConstrainRefLatticeValue &rhs);

  bool isScalar() const { return std::holds_alternative<ScalarTy>(value); }
  bool isSingleValue() const { return isScalar() && getScalarValue().size() == 1; }
  bool isArray() const { return std::holds_alternative<ArrayTy>(value); }

  const ScalarTy &getScalarValue() const {
    debug::ensure(isScalar(), "not a scalar value");
    return std::get<ScalarTy>(value);
  }

  ScalarTy &getScalarValue() {
    debug::ensure(isScalar(), "not a scalar value");
    return std::get<ScalarTy>(value);
  }

  const ConstrainRef &getSingleValue() const {
    debug::ensure(isSingleValue(), "not a single value");
    return *getScalarValue().begin();
  }

  const ArrayTy &getArrayValue() const {
    debug::ensure(isArray(), "not an array value");
    return std::get<ArrayTy>(value);
  }

  size_t getArraySize() const { return getArrayValue().size(); }

  ArrayTy &getArrayValue() {
    debug::ensure(isArray(), "not an array value");
    return std::get<ArrayTy>(value);
  }

  const ConstrainRefLatticeValue &getElemFlatIdx(unsigned i) const {
    debug::ensure(isArray(), "not an array value");
    auto &arr = getArrayValue();
    debug::ensure(i < arr.size(), "index out of range");
    return *arr.at(i);
  }

  ConstrainRefLatticeValue &getElemFlatIdx(unsigned i) {
    debug::ensure(isArray(), "not an array value");
    auto &arr = getArrayValue();
    debug::ensure(i < arr.size(), "index out of range");
    return *arr.at(i);
  }

  /// @brief Sets this value to be equal to `rhs`.
  /// Like the assignment operator, but returns a mlir::ChangeResult if an update
  /// is created,
  mlir::ChangeResult setValue(const ConstrainRefLatticeValue &rhs);

  /// @brief Union this value with that of rhs.
  mlir::ChangeResult update(const ConstrainRefLatticeValue &rhs);

  mlir::ChangeResult insert(const ConstrainRef &rhs);

  /// Translate
  std::pair<ConstrainRefLatticeValue, mlir::ChangeResult>
  translate(const TranslationMap &translation) const;

  std::pair<ConstrainRefLatticeValue, mlir::ChangeResult> index(const ConstrainRefIndex &idx) const;

  std::pair<ConstrainRefLatticeValue, mlir::ChangeResult> index(const ConstrainRef &fieldRef) const;

  /// Perform an extractarr or readarr operation, depending on how many indices
  /// are provided.
  std::pair<ConstrainRefLatticeValue, mlir::ChangeResult>
  extract(const std::vector<ConstrainRefIndex> &indices) const;

  ScalarTy foldToScalar() const;

  void print(mlir::raw_ostream &os) const;

  bool operator==(const ConstrainRefLatticeValue &rhs) const;

private:
  std::variant<ScalarTy, ArrayTy> value;
  std::optional<std::vector<int64_t>> arrayShape;

  mlir::ChangeResult updateScalar(const ScalarTy &rhs);

  mlir::ChangeResult updateArray(const ArrayTy &rhs);

  mlir::ChangeResult foldAndUpdate(const ConstrainRefLatticeValue &rhs);

  mlir::ChangeResult translateScalar(const TranslationMap &translation);

  /// @brief Perform a recursive transformation over all elements of this value and
  /// return a new value with the modifications.
  std::pair<ConstrainRefLatticeValue, mlir::ChangeResult>
  elementwiseTransform(std::function<ConstrainRef(const ConstrainRef &)> transform) const;
};

mlir::raw_ostream &operator<<(mlir::raw_ostream &os, const ConstrainRefLatticeValue &v);

/// A lattice for use in dense analysis.
class ConstrainRefLattice : public dataflow::AbstractDenseLattice {
public:
  using ValueMap = mlir::DenseMap<mlir::Value, ConstrainRefLatticeValue>;
  using AbstractDenseLattice::AbstractDenseLattice;

  /* Static utilities */

  /// If val is the source of other values (i.e., a block argument from the function
  /// args or a constant), create the base reference to the val. Otherwise,
  /// return failure.
  /// Our lattice values must originate from somewhere.
  static mlir::FailureOr<ConstrainRef> getSourceRef(mlir::Value val);

  /* Required methods */

  /// Maximum upper bound
  mlir::ChangeResult join(const AbstractDenseLattice &rhs) override {
    if (auto *r = dynamic_cast<const ConstrainRefLattice *>(&rhs)) {
      return setValues(r->valMap);
    }
    llvm::report_fatal_error("invalid join lattice type");
    return mlir::ChangeResult::NoChange;
  }

  /// Minimum lower bound
  virtual mlir::ChangeResult meet(const AbstractDenseLattice &rhs) override {
    llvm::report_fatal_error("meet operation is not supported for ConstrainRefLattice");
    return mlir::ChangeResult::NoChange;
  }

  void print(mlir::raw_ostream &os) const override;

  /* Update utility methods */

  mlir::ChangeResult setValues(const ValueMap &rhs);

  mlir::ChangeResult setValue(mlir::Value v, const ConstrainRefLatticeValue &rhs) {
    return valMap[v].setValue(rhs);
  }

  mlir::ChangeResult setValue(mlir::Value v, const ConstrainRef &ref) {
    return valMap[v].setValue(ConstrainRefLatticeValue(ref));
  }

  ConstrainRefLatticeValue getOrDefault(mlir::Value v) const;

  ConstrainRefLatticeValue getReturnValue(unsigned i) const;

private:
  ValueMap valMap;
};

mlir::raw_ostream &operator<<(mlir::raw_ostream &os, const ConstrainRefLattice &v);

} // namespace llzk
