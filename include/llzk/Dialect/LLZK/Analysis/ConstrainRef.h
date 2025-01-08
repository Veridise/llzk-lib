#pragma once

#include "llzk/Dialect/LLZK/IR/Ops.h"
#include "llzk/Dialect/LLZK/Util/Hash.h"

#include <mlir/Analysis/DataFlowFramework.h>
#include <mlir/Dialect/Index/IR/IndexOps.h>
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
  explicit ConstrainRefIndex(FieldDefOp f) : index(f) {}
  explicit ConstrainRefIndex(mlir::APInt i) : index(i) {}
  explicit ConstrainRefIndex(int64_t i) : index(mlir::APInt(64, i)) {}
  ConstrainRefIndex(mlir::APInt low, mlir::APInt high) : index(IndexRange{low, high}) {}
  explicit ConstrainRefIndex(IndexRange r) : index(r) {}

  bool isField() const { return std::holds_alternative<FieldDefOp>(index); }
  FieldDefOp getField() const {
    ensureIsField();
    return std::get<FieldDefOp>(index);
  }

  bool isIndex() const { return std::holds_alternative<mlir::APInt>(index); }
  mlir::APInt getIndex() const {
    ensureIsIndex();
    return std::get<mlir::APInt>(index);
  }

  bool isIndexRange() const { return std::holds_alternative<IndexRange>(index); }
  IndexRange getIndexRange() const {
    ensureIsIndexRange();
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
        return OpHash<FieldDefOp>{}(c.getField());
      }
    }
  };

  size_t getHash() const { return Hash{}(*this); }

private:
  /// Either:
  /// 1. A field within a struct
  /// 2. An index into an array
  /// 3. A half-open range of indices into an array, for when we're unsure about a specifc index.
  /// Likely, this will be from [0, size) at this point.
  std::variant<FieldDefOp, mlir::APInt, IndexRange> index;

  void ensureIsField() const {
    if (!isField()) {
      llvm::report_fatal_error("ConstrainRefIndex: field requested but not contained");
    }
  }

  void ensureIsIndex() const {
    if (!isIndex()) {
      llvm::report_fatal_error("ConstrainRefIndex: index requested but not contained");
    }
  }

  void ensureIsIndexRange() const {
    if (!isIndexRange()) {
      llvm::report_fatal_error("ConstrainRefIndex: index range requested but not contained");
    }
  }
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
      mlir::SymbolTableCollection &tables, mlir::ModuleOp mod, StructDefOp s,
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
      : blockArg(b), fieldRefs({}), constFelt(nullptr), constIdx(nullptr) {}
  ConstrainRef(mlir::BlockArgument b, std::vector<ConstrainRefIndex> f)
      : blockArg(b), fieldRefs(f), constFelt(nullptr), constIdx(nullptr) {}
  explicit ConstrainRef(FeltConstantOp c)
      : blockArg(nullptr), fieldRefs({}), constFelt(c), constIdx(nullptr) {}
  explicit ConstrainRef(mlir::index::ConstantOp c)
      : blockArg(nullptr), fieldRefs({}), constFelt(nullptr), constIdx(c) {}

  mlir::Type getType() const;

  bool isConstantFelt() const { return constFelt != nullptr; }
  bool isConstantIndex() const { return constIdx != nullptr; }
  bool isConstant() const { return isConstantFelt() || isConstantIndex(); }

  bool isFeltVal() const { return mlir::isa<FeltType>(getType()); }
  bool isIndexVal() const { return mlir::isa<mlir::IndexType>(getType()); }
  bool isIntegerVal() const { return mlir::isa<mlir::IntegerType>(getType()); }
  bool isScalar() const { return isConstant() || isFeltVal() || isIndexVal() || isIntegerVal(); }

  unsigned getInputNum() const { return blockArg.getArgNumber(); }
  mlir::APInt getConstantFeltValue() const {
    return const_cast<FeltConstantOp &>(constFelt).getValueAttr().getValue();
  }
  mlir::APInt getConstantIndexValue() const {
    return const_cast<mlir::index::ConstantOp &>(constIdx).getValue();
  }

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

  void print(mlir::raw_ostream &os) const;
  void dump() const { print(llvm::errs()); }

  bool operator==(const ConstrainRef &rhs) const;

  // required for EquivalenceClasses usage
  bool operator<(const ConstrainRef &rhs) const;

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
  FeltConstantOp constFelt;
  mlir::index::ConstantOp constIdx;
};

mlir::raw_ostream &operator<<(mlir::raw_ostream &os, const ConstrainRef &rhs);

using ConstrainRefSet = std::unordered_set<ConstrainRef, ConstrainRef::Hash>;

} // namespace llzk