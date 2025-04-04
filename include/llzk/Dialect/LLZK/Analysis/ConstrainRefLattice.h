//===-- ConstrainRefLattice.h -----------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llzk/Dialect/LLZK/Analysis/AbstractLatticeValue.h"
#include "llzk/Dialect/LLZK/Analysis/ConstrainRef.h"
#include "llzk/Dialect/LLZK/Analysis/DenseAnalysis.h"
#include "llzk/Dialect/LLZK/Util/ErrorHelper.h"

namespace llzk {

class ConstrainRefLatticeValue;
using TranslationMap =
    std::unordered_map<ConstrainRef, ConstrainRefLatticeValue, ConstrainRef::Hash>;

/// @brief A value at a given point of the ConstrainRefLattice.
class ConstrainRefLatticeValue
    : public dataflow::AbstractLatticeValue<ConstrainRefLatticeValue, ConstrainRefSet> {
  using Base = dataflow::AbstractLatticeValue<ConstrainRefLatticeValue, ConstrainRefSet>;
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

public:
  explicit ConstrainRefLatticeValue(ScalarTy s) : Base(s) {}
  explicit ConstrainRefLatticeValue(ConstrainRef r) : Base(ScalarTy {r}) {}
  ConstrainRefLatticeValue() : Base(ScalarTy {}) {}

  // Create an empty array of the given shape.
  explicit ConstrainRefLatticeValue(mlir::ArrayRef<int64_t> shape) : Base(shape) {}

  const ConstrainRef &getSingleValue() const {
    ensure(isSingleValue(), "not a single value");
    return *getScalarValue().begin();
  }

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

protected:
  /// @brief Translate this value using the translation map, assuming this value
  /// is a scalar.
  mlir::ChangeResult translateScalar(const TranslationMap &translation);

  /// @brief Perform a recursive transformation over all elements of this value and
  /// return a new value with the modifications.
  virtual std::pair<ConstrainRefLatticeValue, mlir::ChangeResult>
  elementwiseTransform(llvm::function_ref<ConstrainRef(const ConstrainRef &)> transform) const;
};

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

  size_t size() const { return valMap.size(); }

  friend mlir::raw_ostream &operator<<(mlir::raw_ostream &os, const ConstrainRefLattice &v);

private:
  ValueMap valMap;
};

} // namespace llzk
