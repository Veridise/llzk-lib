//===-- ConstrainRef.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llzk/Analysis/AbstractLatticeValue.h"
#include "llzk/Dialect/Felt/IR/Ops.h"
#include "llzk/Dialect/LLZK/IR/AttributeHelper.h"
#include "llzk/Dialect/Polymorphic/IR/Ops.h"
#include "llzk/Dialect/Struct/IR/Ops.h"
#include "llzk/Util/ErrorHelper.h"
#include "llzk/Util/Hash.h"

#include <mlir/Analysis/DataFlowFramework.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
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
  explicit ConstrainRefIndex(component::FieldDefOp f) : index(f) {}
  explicit ConstrainRefIndex(SymbolLookupResult<component::FieldDefOp> f) : index(f) {}
  explicit ConstrainRefIndex(mlir::APInt i) : index(i) {}
  explicit ConstrainRefIndex(int64_t i) : index(toAPInt(i)) {}
  ConstrainRefIndex(mlir::APInt low, mlir::APInt high) : index(IndexRange {low, high}) {}
  explicit ConstrainRefIndex(IndexRange r) : index(r) {}

  bool isField() const {
    return std::holds_alternative<SymbolLookupResult<component::FieldDefOp>>(index) ||
           std::holds_alternative<component::FieldDefOp>(index);
  }
  component::FieldDefOp getField() const {
    ensure(isField(), "ConstrainRefIndex: field requested but not contained");
    if (std::holds_alternative<component::FieldDefOp>(index)) {
      return std::get<component::FieldDefOp>(index);
    }
    return std::get<SymbolLookupResult<component::FieldDefOp>>(index).get();
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

  inline bool operator==(const ConstrainRefIndex &rhs) const {
    if (isField() && rhs.isField()) {
      // We compare the underlying fields, since the field could be in a symbol
      // lookup or not.
      return getField() == rhs.getField();
    }
    return index == rhs.index;
  }

  bool operator<(const ConstrainRefIndex &rhs) const;

  bool operator>(const ConstrainRefIndex &rhs) const { return rhs < *this; }

  struct Hash {
    size_t operator()(const ConstrainRefIndex &c) const;
  };

  inline size_t getHash() const { return Hash {}(*this); }

private:
  /// Either:
  /// 1. A field within a struct (possibly as a SymbolLookupResult to be cautious of external module
  /// scopes)
  /// 2. An index into an array
  /// 3. A half-open range of indices into an array, for when we're unsure about a specific index
  /// Likely, this will be from [0, size) at this point.
  std::variant<
      component::FieldDefOp, SymbolLookupResult<component::FieldDefOp>, mlir::APInt, IndexRange>
      index;
};

static inline mlir::raw_ostream &operator<<(mlir::raw_ostream &os, const ConstrainRefIndex &rhs) {
  rhs.print(os);
  return os;
}

/// @brief Defines a reference to a llzk object within a constrain function call.
/// The object may be a reference to an individual felt, felt.const, or a composite type,
/// like an array or an entire struct.
/// - ConstrainRefs are allowed to reference composite types so that references can be generated
/// for intermediate operations (e.g., readf to read a nested struct).
/// These references are relative to a particular constrain call, so they are either (1) constants,
/// or (2) rooted at a block argument (which is either "self" or another input) and optionally
/// contain indices into that block argument (e.g., a field reference in a struct or a index into an
/// array).
class ConstrainRef {

public:
  /// Produce all possible ConstraintRefs that are present starting from the given root.
  static std::vector<ConstrainRef>
  getAllConstrainRefs(mlir::SymbolTableCollection &tables, mlir::ModuleOp mod, ConstrainRef root);

  /// Produce all possible ConstrainRefs that are present from given struct function.
  static std::vector<ConstrainRef>
  getAllConstrainRefs(component::StructDefOp structDef, function::FuncDefOp fnOp);

  /// Produce all possible ConstrainRefs from a specific field in a struct.
  /// May produce multiple if the given field is of an aggregate type.
  static std::vector<ConstrainRef>
  getAllConstrainRefs(component::StructDefOp structDef, component::FieldDefOp fieldDef);

  explicit ConstrainRef(mlir::BlockArgument b) : root(b), fieldRefs(), constantVal(std::nullopt) {}
  ConstrainRef(mlir::BlockArgument b, std::vector<ConstrainRefIndex> f)
      : root(b), fieldRefs(std::move(f)), constantVal(std::nullopt) {}

  explicit ConstrainRef(component::CreateStructOp createOp)
      : root(createOp), fieldRefs(), constantVal(std::nullopt) {}
  ConstrainRef(component::CreateStructOp createOp, std::vector<ConstrainRefIndex> f)
      : root(createOp), fieldRefs(std::move(f)), constantVal(std::nullopt) {}

  explicit ConstrainRef(felt::FeltConstantOp c) : root(std::nullopt), fieldRefs(), constantVal(c) {}
  explicit ConstrainRef(mlir::arith::ConstantIndexOp c)
      : root(std::nullopt), fieldRefs(), constantVal(c) {}
  explicit ConstrainRef(polymorphic::ConstReadOp c)
      : root(std::nullopt), fieldRefs(), constantVal(c) {}

  mlir::Type getType() const;

  bool isConstantFelt() const {
    return constantVal.has_value() && std::holds_alternative<felt::FeltConstantOp>(*constantVal);
  }
  bool isConstantIndex() const {
    return constantVal.has_value() &&
           std::holds_alternative<mlir::arith::ConstantIndexOp>(*constantVal);
  }
  bool isTemplateConstant() const {
    return constantVal.has_value() &&
           std::holds_alternative<polymorphic::ConstReadOp>(*constantVal);
  }
  bool isConstant() const { return constantVal.has_value(); }
  bool isConstantInt() const { return isConstantFelt() || isConstantIndex(); }

  bool isFeltVal() const { return mlir::isa<felt::FeltType>(getType()); }
  bool isIndexVal() const { return mlir::isa<mlir::IndexType>(getType()); }
  bool isIntegerVal() const { return mlir::isa<mlir::IntegerType>(getType()); }
  bool isTypeVarVal() const { return mlir::isa<polymorphic::TypeVarType>(getType()); }
  bool isScalar() const {
    return isConstant() || isFeltVal() || isIndexVal() || isIntegerVal() || isTypeVarVal();
  }
  bool isSignal() const { return isSignalType(getType()); }

  bool isBlockArgument() const {
    return root.has_value() && std::holds_alternative<mlir::BlockArgument>(*root);
  }
  mlir::BlockArgument getBlockArgument() const {
    ensure(isBlockArgument(), "is not a block argument");
    return std::get<mlir::BlockArgument>(*root);
  }
  unsigned getInputNum() const { return getBlockArgument().getArgNumber(); }

  bool isCreateStructOp() const {
    return root.has_value() && std::holds_alternative<component::CreateStructOp>(*root);
  }
  component::CreateStructOp getCreateStructOp() const {
    ensure(isCreateStructOp(), "is not a create struct op");
    return std::get<component::CreateStructOp>(*root);
  }

  mlir::APInt getConstantFeltValue() const {
    ensure(isConstantFelt(), __FUNCTION__ + mlir::Twine(" requires a constant felt!"));
    return std::get<felt::FeltConstantOp>(*constantVal).getValueAttr().getValue();
  }
  mlir::APInt getConstantIndexValue() const {
    ensure(isConstantIndex(), __FUNCTION__ + mlir::Twine(" requires a constant index!"));
    return toAPInt(std::get<mlir::arith::ConstantIndexOp>(*constantVal).value());
  }
  mlir::APInt getConstantValue() const {
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
  /// this reference. If this reference is a felt.const, the translation will always succeed and
  /// return the felt.const unchanged.
  /// @param prefix
  /// @param other
  /// @return
  mlir::FailureOr<ConstrainRef>
  translate(const ConstrainRef &prefix, const ConstrainRef &other) const;

  /// @brief Create a new reference that is the immediate prefix of this reference if possible.
  mlir::FailureOr<ConstrainRef> getParentPrefix() const {
    if (isConstantFelt() || fieldRefs.empty()) {
      return mlir::failure();
    }
    auto copy = *this;
    copy.fieldRefs.pop_back();
    return copy;
  }

  /// @brief Get all direct children of this ConstrainRef, assuming this ref is not a scalar.
  std::vector<ConstrainRef>
  getAllChildren(mlir::SymbolTableCollection &tables, mlir::ModuleOp mod) const;

  ConstrainRef createChild(ConstrainRefIndex r) const {
    auto copy = *this;
    copy.fieldRefs.push_back(r);
    return copy;
  }

  ConstrainRef createChild(ConstrainRef other) const {
    assert(other.isConstantIndex());
    return createChild(ConstrainRefIndex(other.getConstantIndexValue()));
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
   * BlockArgument:
   * - If the block arg is 0, then it refers to "self", meaning the signal is internal or an output
   * (public means an output).
   * - Otherwise, it is an input, either public or private.
   *
   * CreateStructOp
   * - For compute functions, the "self" argument is an allocation site.
   */
  std::optional<std::variant<mlir::BlockArgument, component::CreateStructOp>> root;

  std::vector<ConstrainRefIndex> fieldRefs;
  // using mutable to reduce constant casts for certain get* functions.
  mutable std::optional<
      std::variant<felt::FeltConstantOp, mlir::arith::ConstantIndexOp, polymorphic::ConstReadOp>>
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
