//===-- ConstrainRef.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llzk/Dialect/LLZK/Analysis/AbstractLatticeValue.h"
#include "llzk/Dialect/LLZK/IR/Ops.h"
#include "llzk/Dialect/LLZK/Util/AttributeHelper.h"
#include "llzk/Dialect/LLZK/Util/ErrorHelper.h"
#include "llzk/Dialect/LLZK/Util/Hash.h"

#include <mlir/Analysis/DataFlowFramework.h>
#include <mlir/Pass/AnalysisManager.h>

#include <llvm/ADT/EquivalenceClasses.h>

#include <unordered_set>
#include <vector>

namespace llzk {

/// @brief Defines an index into an LLZK object. For structs, this is a field
/// definition, and for arrays, this is an element index.
/// Effectively a wrapper around a std::variant with extra utility methods.
class ConstrainRefIndex {
  using IndexRange = std::pair<mlir::APInt, mlir::APInt>;

public:
  explicit ConstrainRefIndex(SymbolLookupResult<FieldDefOp> f) : index(f) {}
  explicit ConstrainRefIndex(mlir::APInt i) : index(i) {}
  explicit ConstrainRefIndex(int64_t i) : index(toAPInt(i)) {}
  ConstrainRefIndex(mlir::APInt low, mlir::APInt high) : index(IndexRange {low, high}) {}
  explicit ConstrainRefIndex(IndexRange r) : index(r) {}

  bool isField() const { return std::holds_alternative<SymbolLookupResult<FieldDefOp>>(index); }
  FieldDefOp getField() const {
    ensure(isField(), "ConstrainRefIndex: field requested but not contained");
    return std::get<SymbolLookupResult<FieldDefOp>>(index).get();
  }

  bool isIndex() const { return std::holds_alternative<mlir::APInt>(index); }
  mlir::APInt getIndex() const {
    ensure(isIndex(), "ConstrainRefIndex: index requested but not contained");
    return std::get<mlir::APInt>(index);
  }

  bool isIndexRange() const { return std::holds_alternative<IndexRange>(index); }
  IndexRange getIndexRange() const {
    ensure(isIndexRange(), "ConstrainRefIndex: index range requested but not contained");
    return std::get<IndexRange>(index);
  }

  inline void dump() const { print(llvm::errs()); }
  void print(mlir::raw_ostream &os) const;

  inline bool operator==(const ConstrainRefIndex &rhs) const { return index == rhs.index; }

  bool operator<(const ConstrainRefIndex &rhs) const;

  bool operator>(const ConstrainRefIndex &rhs) const { return rhs < *this; }

  struct Hash {
    size_t operator()(const ConstrainRefIndex &c) const {
      if (c.isIndex()) {
        return llvm::hash_value(c.getIndex());
      } else if (c.isIndexRange()) {
        auto r = c.getIndexRange();
        return llvm::hash_value(std::get<0>(r)) ^ llvm::hash_value(std::get<1>(r));
      } else {
        return OpHash<FieldDefOp> {}(c.getField());
      }
    }
  };

  size_t getHash() const { return Hash {}(*this); }

private:
  /// Either:
  /// 1. A field within a struct (as a SymbolLookupResult to be cautious of external module scopes)
  /// 2. An index into an array
  /// 3. A half-open range of indices into an array, for when we're unsure about a specifc index
  /// Likely, this will be from [0, size) at this point.
  std::variant<SymbolLookupResult<FieldDefOp>, mlir::APInt, IndexRange> index;
};

static inline mlir::raw_ostream &operator<<(mlir::raw_ostream &os, const ConstrainRefIndex &rhs) {
  rhs.print(os);
  return os;
}

/// @brief Defines a reference to a llzk object within a constrain function call.
/// The object may be a reference to an individual felt, constfelt, or a composite type,
/// like an array or an entire struct.
/// - ConstrainRefs are allowed to reference composite types so that references can be generated
/// for intermediate operations (e.g., readf to read a nested struct).
/// These references are relative to a particular constrain call, so they are either (1) constants,
/// or (2) rooted at a block argument (which is either "self" or another input) and optionally
/// contain indices into that block argument (e.g., a field reference in a struct or a index into an
/// array).
class ConstrainRef {

  /// Produce all possible ConstraintRefs that are present starting from the given
  /// arrayField, originating from a given blockArg,
  /// and partially-specified indices into that object (fields).
  /// This produces refs for composite types (e.g., full structs and full arrays)
  /// as well as individual fields and constants.
  static std::vector<ConstrainRef> getAllConstrainRefs(
      mlir::SymbolTableCollection &tables, mlir::ModuleOp mod, ArrayType arrayTy,
      mlir::BlockArgument blockArg, std::vector<ConstrainRefIndex> fields
  );

  /// Produce all possible ConstraintRefs that are present starting from the given
  /// BlockArgument and partially-specified indices into that object (fields).
  /// This produces refs for composite types (e.g., full structs and full arrays)
  /// as well as individual fields and constants.
  static std::vector<ConstrainRef> getAllConstrainRefs(
      mlir::SymbolTableCollection &tables, mlir::ModuleOp mod, SymbolLookupResult<StructDefOp> s,
      mlir::BlockArgument blockArg, std::vector<ConstrainRefIndex> fields
  );

  /// Produce all possible ConstraintRefs that are present starting from the given BlockArgument.
  static std::vector<ConstrainRef> getAllConstrainRefs(
      mlir::SymbolTableCollection &tables, mlir::ModuleOp mod, mlir::BlockArgument arg
  );

public:
  /// Produce all possible ConstraintRefs that are present from the struct's constrain function.
  static std::vector<ConstrainRef> getAllConstrainRefs(StructDefOp structDef);

  explicit ConstrainRef(mlir::BlockArgument b)
      : blockArg(b), fieldRefs(), constantVal(std::nullopt) {}
  ConstrainRef(mlir::BlockArgument b, std::vector<ConstrainRefIndex> f)
      : blockArg(b), fieldRefs(std::move(f)), constantVal(std::nullopt) {}
  explicit ConstrainRef(FeltConstantOp c) : blockArg(nullptr), fieldRefs(), constantVal(c) {}
  explicit ConstrainRef(mlir::arith::ConstantIndexOp c)
      : blockArg(nullptr), fieldRefs(), constantVal(c) {}
  explicit ConstrainRef(ConstReadOp c) : blockArg(nullptr), fieldRefs(), constantVal(c) {}

  mlir::Type getType() const;

  bool isConstantFelt() const {
    return constantVal.has_value() && std::holds_alternative<FeltConstantOp>(*constantVal);
  }
  bool isConstantIndex() const {
    return constantVal.has_value() &&
           std::holds_alternative<mlir::arith::ConstantIndexOp>(*constantVal);
  }
  bool isTemplateConstant() const {
    return constantVal.has_value() && std::holds_alternative<ConstReadOp>(*constantVal);
  }
  bool isConstant() const { return constantVal.has_value(); }

  bool isFeltVal() const { return mlir::isa<FeltType>(getType()); }
  bool isIndexVal() const { return mlir::isa<mlir::IndexType>(getType()); }
  bool isIntegerVal() const { return mlir::isa<mlir::IntegerType>(getType()); }
  bool isTypeVarVal() const { return mlir::isa<TypeVarType>(getType()); }
  bool isScalar() const {
    return isConstant() || isFeltVal() || isIndexVal() || isIntegerVal() || isTypeVarVal();
  }
  bool isSignal() const { return isSignalType(getType()); }

  bool isBlockArgument() const { return blockArg != nullptr; }
  mlir::BlockArgument getBlockArgument() const {
    ensure(isBlockArgument(), "is not a block argument");
    return blockArg;
  }
  unsigned getInputNum() const { return blockArg.getArgNumber(); }

  mlir::APInt getConstantFeltValue() const {
    ensure(isConstantFelt(), __FUNCTION__ + mlir::Twine(" requires a constant felt!"));
    return std::get<FeltConstantOp>(*constantVal).getValueAttr().getValue();
  }
  mlir::APInt getConstantIndexValue() const {
    ensure(isConstantIndex(), __FUNCTION__ + mlir::Twine(" requires a constant index!"));
    return toAPInt(std::get<mlir::arith::ConstantIndexOp>(*constantVal).value());
  }
  mlir::APInt getConstantInt() const {
    ensure(
        isConstantFelt() || isConstantIndex(),
        __FUNCTION__ + mlir::Twine(" requires a constant int type!")
    );
    return isConstantFelt() ? getConstantFeltValue() : getConstantIndexValue();
  }

  /// @brief Returns true iff `prefix` is a valid prefix of this reference.
  bool isValidPrefix(const ConstrainRef &prefix) const;

  /// @brief If `prefix` is a valid prefix of this reference, return the suffix that
  /// remains after removing the prefix. I.e., `this` = `prefix` + `suffix`
  /// @param prefix
  /// @return the suffix
  mlir::FailureOr<std::vector<ConstrainRefIndex>> getSuffix(const ConstrainRef &prefix) const;

  /// @brief Create a new reference with prefix replaced with other iff prefix is a valid prefix for
  /// this reference. If this reference is a constfelt, the translation will always succeed and
  /// return the constfelt unchanged.
  /// @param prefix
  /// @param other
  /// @return
  mlir::FailureOr<ConstrainRef>
  translate(const ConstrainRef &prefix, const ConstrainRef &other) const;

  /// @brief Create a new reference that is the immediate prefix of this reference if possible.
  /// @return
  mlir::FailureOr<ConstrainRef> getParentPrefix() const {
    if (isConstantFelt() || fieldRefs.empty()) {
      return mlir::failure();
    }
    auto copy = *this;
    copy.fieldRefs.pop_back();
    return copy;
  }

  ConstrainRef createChild(ConstrainRefIndex r) const {
    auto copy = *this;
    copy.fieldRefs.push_back(r);
    return copy;
  }

  ConstrainRef createChild(ConstrainRef other) const {
    auto copy = *this;
    assert(other.isConstantIndex());
    copy.fieldRefs.push_back(ConstrainRefIndex(other.getConstantIndexValue()));
    return copy;
  }

  const std::vector<ConstrainRefIndex> &getPieces() const { return fieldRefs; }

  void print(mlir::raw_ostream &os) const;
  void dump() const { print(llvm::errs()); }

  bool operator==(const ConstrainRef &rhs) const;

  bool operator!=(const ConstrainRef &rhs) const { return !(*this == rhs); }

  // required for EquivalenceClasses usage
  bool operator<(const ConstrainRef &rhs) const;

  bool operator>(const ConstrainRef &rhs) const { return rhs < *this; }

  struct Hash {
    size_t operator()(const ConstrainRef &val) const;
  };

private:
  /**
   * If the block arg is 0, then it refers to "self", meaning the signal is internal or an output
   * (public means an output). Otherwise, it is an input, either public or private.
   */
  mlir::BlockArgument blockArg;

  std::vector<ConstrainRefIndex> fieldRefs;
  // using mutable to reduce constant casts for certain get* functions.
  mutable std::optional<std::variant<FeltConstantOp, mlir::arith::ConstantIndexOp, ConstReadOp>>
      constantVal;
};

mlir::raw_ostream &operator<<(mlir::raw_ostream &os, const ConstrainRef &rhs);

/* ConstrainRefSet */

class ConstrainRefSet : public std::unordered_set<ConstrainRef, ConstrainRef::Hash> {
  using Base = std::unordered_set<ConstrainRef, ConstrainRef::Hash>;

public:
  using Base::Base;

  ConstrainRefSet &join(const ConstrainRefSet &rhs);

  friend mlir::raw_ostream &operator<<(mlir::raw_ostream &os, const ConstrainRefSet &rhs);
};

static_assert(
    dataflow::ScalarLatticeValue<ConstrainRefSet>,
    "ConstrainRefSet must satisfy the ScalarLatticeValue requirements"
);

} // namespace llzk

namespace llvm {

template <> struct DenseMapInfo<llzk::ConstrainRef> {
  static llzk::ConstrainRef getEmptyKey() {
    return llzk::ConstrainRef(mlir::BlockArgument(reinterpret_cast<mlir::detail::ValueImpl *>(1)));
  }
  static inline llzk::ConstrainRef getTombstoneKey() {
    return llzk::ConstrainRef(mlir::BlockArgument(reinterpret_cast<mlir::detail::ValueImpl *>(2)));
  }
  static unsigned getHashValue(const llzk::ConstrainRef &ref) {
    if (ref == getEmptyKey() || ref == getTombstoneKey()) {
      return llvm::hash_value(ref.getBlockArgument().getAsOpaquePointer());
    }
    return llzk::ConstrainRef::Hash {}(ref);
  }
  static bool isEqual(const llzk::ConstrainRef &lhs, const llzk::ConstrainRef &rhs) {
    return lhs == rhs;
  }
};

} // namespace llvm
