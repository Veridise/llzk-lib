//===-- ConstrainRefLattice.h -----------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llzk/Analysis/AbstractLatticeValue.h"
#include "llzk/Analysis/ConstrainRef.h"
#include "llzk/Analysis/DenseAnalysis.h"
#include "llzk/Util/ErrorHelper.h"

#include <llvm/ADT/PointerUnion.h>

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
  /// For arrays of values created by, e.g., the LLZK array.new op. A recursive
  /// definition allows arrays to be constructed of other existing values, which is
  /// how the `array.new` operator works.
  /// - Unique pointers are used as each value must be self contained for the
  /// sake of consistent translations. Copies are explicit.
  /// - This array is flattened, with the dimensions stored in another structure.
  /// This simplifies the construction of multidimensional arrays.
  using ArrayTy = std::vector<std::unique_ptr<ConstrainRefLatticeValue>>;

public:
  explicit ConstrainRefLatticeValue(ScalarTy s) : Base(s) {}
  explicit ConstrainRefLatticeValue(ConstrainRef r) : Base(ScalarTy {r}) {}
  ConstrainRefLatticeValue() : Base(ScalarTy {}) {}
  virtual ~ConstrainRefLatticeValue() = default;

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
  referenceField(SymbolLookupResult<component::FieldDefOp> fieldRef) const;

  /// @brief Perform an array.extract or array.read operation, depending on how many indices
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
  // mlir::Value is used for read-like operations that create references in their results,
  // mlir::Operation* is used for write-like operations that reference values as their destinations
  using ValueTy = llvm::PointerUnion<mlir::Value, mlir::Operation *>;
  using ValueMap = mlir::DenseMap<ValueTy, ConstrainRefLatticeValue>;
  // Used to lookup MLIR values/operations from a given ConstrainRef (all values that a ref is referenced by)
  using ValueSet = mlir::DenseSet<ValueTy>;
  using Ref2Val = mlir::DenseMap<ConstrainRef, mlir::DenseSet<ValueTy>>;
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

  mlir::ChangeResult setValue(ValueTy v, const ConstrainRefLatticeValue &rhs);

  mlir::ChangeResult setValue(ValueTy v, const ConstrainRef &ref);

  ConstrainRefLatticeValue getOrDefault(ValueTy v) const;

  ConstrainRefLatticeValue getReturnValue(unsigned i) const;

  ValueSet lookupValues(const ConstrainRef &r) const;

  size_t size() const { return valMap.size(); }

  const ValueMap &getMap() const { return valMap; }

  const Ref2Val &getRef2Val() const { return refMap; }

  ValueMap::iterator begin() { return valMap.begin(); }
  ValueMap::iterator end() { return valMap.end(); }
  ValueMap::const_iterator begin() const { return valMap.begin(); }
  ValueMap::const_iterator end() const { return valMap.end(); }

  friend mlir::raw_ostream &operator<<(mlir::raw_ostream &os, const ConstrainRefLattice &v);

private:
  ValueMap valMap;
  Ref2Val refMap;
};

} // namespace llzk

namespace llvm {
class raw_ostream;

raw_ostream &operator<<(raw_ostream &os, llvm::PointerUnion<mlir::Value, mlir::Operation *> ptr);
}
