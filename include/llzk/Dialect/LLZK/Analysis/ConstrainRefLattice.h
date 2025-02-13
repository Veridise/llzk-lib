#pragma once

#include "llzk/Dialect/LLZK/Analysis/ConstrainRef.h"
#include "llzk/Dialect/LLZK/Analysis/DenseAnalysis.h"
#include "llzk/Dialect/LLZK/Util/Helpers.h"

namespace llzk {

class ConstrainRefLatticeValue;
using TranslationMap =
    std::unordered_map<ConstrainRef, ConstrainRefLatticeValue, ConstrainRef::Hash>;

/// @brief A value at a given point of the ConstrainRefLattice.
class ConstrainRefLatticeValue {
  /// For scalar values.
  using ScalarTy = ConstrainRefSet;
  /// For arrays of values created by, e.g., the LLZK new_array op. A recursive
  /// definition allows arrays to be constructed of other existing values, which is
  /// how the `new_array` operator works.
  /// - Unique pointers are used as each value must be self contained for the
  /// sake of consistent translations. Copies are explicit.
  /// - This array is flattened, with the dimensions stored in another structure.
  /// This simplifies the construction of multidimensional arrays.
  using ArrayTy = std::vector<std::unique_ptr<ConstrainRefLatticeValue>>;

  /// @brief Create a new array with the given `shape`. The values are pre-allocated
  /// to empty scalar values.
  static ArrayTy constructArrayTy(const mlir::ArrayRef<int64_t> &shape);

public:
  explicit ConstrainRefLatticeValue(ScalarTy s) : value(s), arrayShape(std::nullopt) {}
  explicit ConstrainRefLatticeValue(ConstrainRef r) : ConstrainRefLatticeValue(ScalarTy {r}) {}
  ConstrainRefLatticeValue() : ConstrainRefLatticeValue(ScalarTy {}) {}

  // Create an empty array of the given shape.
  explicit ConstrainRefLatticeValue(mlir::ArrayRef<int64_t> shape)
      : value(constructArrayTy(shape)), arrayShape(shape) {}

  ConstrainRefLatticeValue(const ConstrainRefLatticeValue &rhs) { *this = rhs; }

  // Enable copying by duplicating unique_ptrs and copying the contained values.
  ConstrainRefLatticeValue &operator=(const ConstrainRefLatticeValue &rhs);

  bool isScalar() const { return std::holds_alternative<ScalarTy>(value); }
  bool isSingleValue() const { return isScalar() && getScalarValue().size() == 1; }
  bool isArray() const { return std::holds_alternative<ArrayTy>(value); }

  const ScalarTy &getScalarValue() const {
    ensure(isScalar(), "not a scalar value");
    return std::get<ScalarTy>(value);
  }

  ScalarTy &getScalarValue() {
    ensure(isScalar(), "not a scalar value");
    return std::get<ScalarTy>(value);
  }

  const ConstrainRef &getSingleValue() const {
    ensure(isSingleValue(), "not a single value");
    return *getScalarValue().begin();
  }

  const ArrayTy &getArrayValue() const {
    ensure(isArray(), "not an array value");
    return std::get<ArrayTy>(value);
  }

  size_t getArraySize() const { return getArrayValue().size(); }

  ArrayTy &getArrayValue() {
    ensure(isArray(), "not an array value");
    return std::get<ArrayTy>(value);
  }

  /// @brief Directly index into the flattened array using a single index.
  const ConstrainRefLatticeValue &getElemFlatIdx(unsigned i) const {
    ensure(isArray(), "not an array value");
    auto &arr = getArrayValue();
    ensure(i < arr.size(), "index out of range");
    return *arr.at(i);
  }

  ConstrainRefLatticeValue &getElemFlatIdx(unsigned i) {
    ensure(isArray(), "not an array value");
    auto &arr = getArrayValue();
    ensure(i < arr.size(), "index out of range");
    return *arr.at(i);
  }

  /// @brief Sets this value to be equal to `rhs`.
  /// @return A `mlir::ChangeResult` indicating if an update was performed or not.
  mlir::ChangeResult setValue(const ConstrainRefLatticeValue &rhs);

  /// @brief Union this value with that of rhs.
  mlir::ChangeResult update(const ConstrainRefLatticeValue &rhs);

  /// @brief Directly insert the ref into this value. If this is a scalar value,
  /// insert the ref into the value's set. If this is an array value, the array
  /// is folded into a single scalar, then the ref is inserted.
  mlir::ChangeResult insert(const ConstrainRef &rhs);

  /// @brief For the refs contained in this value, translate them given the `translation`
  /// map and return the transformed value.
  std::pair<ConstrainRefLatticeValue, mlir::ChangeResult>
  translate(const TranslationMap &translation) const;

  /// @brief Add the given `fieldRef` to the constrain refs contained within this value.
  /// For example, if `fieldRef` is a field reference `@foo` and this value represents `%self`,
  /// the new value will represent `%self[@foo]`.
  /// @param fieldRef The field reference into the current value.
  /// @return The new value and a change result indicating if the value is different than the
  /// original value.
  std::pair<ConstrainRefLatticeValue, mlir::ChangeResult>
  referenceField(SymbolLookupResult<FieldDefOp> fieldRef) const;

  /// @brief Perform an extractarr or readarr operation, depending on how many indices
  /// are provided.
  std::pair<ConstrainRefLatticeValue, mlir::ChangeResult>
  extract(const std::vector<ConstrainRefIndex> &indices) const;

  /// @brief If this is an array value, combine all elements into a single scalar
  /// value and return it. If this is already a scalar value, return the scalar value.
  ScalarTy foldToScalar() const;

  void print(mlir::raw_ostream &os) const;

  bool operator==(const ConstrainRefLatticeValue &rhs) const;

private:
  std::variant<ScalarTy, ArrayTy> value;
  std::optional<std::vector<int64_t>> arrayShape;

  /// @brief Union this value with the given scalar.
  mlir::ChangeResult updateScalar(const ScalarTy &rhs);

  /// @brief Union this value with the given array.
  mlir::ChangeResult updateArray(const ArrayTy &rhs);

  /// @brief Folds the current value into a scalar and folds `rhs` to a scalar and updates
  /// the current value to the union of the two scalars.
  mlir::ChangeResult foldAndUpdate(const ConstrainRefLatticeValue &rhs);

  /// @brief Translate this value using the translation map, assuming this value
  /// is a scalar.
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
