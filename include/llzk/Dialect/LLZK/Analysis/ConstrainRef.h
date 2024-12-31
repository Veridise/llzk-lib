#pragma once

#include "llzk/Dialect/LLZK/IR/Ops.h"
#include "llzk/Dialect/LLZK/Util/Hash.h"

#include <mlir/Analysis/DataFlowFramework.h>
#include <mlir/Pass/AnalysisManager.h>

#include <llvm/ADT/EquivalenceClasses.h>

#include <vector>

namespace llzk {

/// @brief Defines an index into an LLZK object. For structs, this is a field
/// definition, and for arrays, this is an element index.
/// Effectively a wrapper around a std::variant with extra utility methods.
class ConstrainRefIndex {
public:
  explicit ConstrainRefIndex(FieldDefOp f) : index(f) {}
  explicit ConstrainRefIndex(int64_t i) : index(i) {}

  bool isField() const { return std::holds_alternative<FieldDefOp>(index); }
  FieldDefOp getField() const {
    ensureIsField();
    return std::get<FieldDefOp>(index);
  }

  bool isIndex() const { return std::holds_alternative<int64_t>(index); }
  int64_t getIndex() const {
    ensureIsIndex();
    return std::get<int64_t>(index);
  }

  void dump() const { print(llvm::errs()); }
  void print(mlir::raw_ostream &os) const {
    if (isField()) {
      os << '@' << getField().getName();
    } else {
      os << getIndex();
    }
  }

  bool operator==(const ConstrainRefIndex &rhs) const { return index == rhs.index; }

  bool operator<(const ConstrainRefIndex &rhs) const { return index < rhs.index; }

  bool operator>(const ConstrainRefIndex &rhs) const { return rhs < *this; }

  struct Hash {
    size_t operator()(const ConstrainRefIndex &c) const {
      if (c.isIndex()) {
        return std::hash<int64_t>{}(c.getIndex());
      } else {
        return OpHash<FieldDefOp>{}(c.getField());
      }
    }
  };

  size_t getHash() const { return Hash{}(*this); }

private:
  std::variant<FieldDefOp, int64_t> index;

  void ensureIsField() const {
    if (!isField()) {
      llvm::report_fatal_error("ConstrainRefIndex: field requested, but holds index type");
    }
  }

  void ensureIsIndex() const {
    if (!isIndex()) {
      llvm::report_fatal_error("ConstrainRefIndex: index requested, but holds field type");
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
public:
  /// Try to create references out of a given operation.
  /// A single operation may contain multiple usages, e.g. addition of signals.
  static mlir::FailureOr<std::vector<ConstrainRef>> get(mlir::Value val);

  explicit ConstrainRef(mlir::BlockArgument b) : blockArg(b), fieldRefs({}), constFelt(nullptr) {}
  ConstrainRef(mlir::BlockArgument b, std::vector<ConstrainRefIndex> f)
      : blockArg(b), fieldRefs(f), constFelt(nullptr) {}
  explicit ConstrainRef(FeltConstantOp c) : blockArg(nullptr), fieldRefs({}), constFelt(c) {}

  mlir::Type getType() const {
    if (isConstant()) {
      return const_cast<FeltConstantOp &>(constFelt).getType();
    } else {
      unsigned array_derefs = 0;
      int idx = fieldRefs.size() - 1;
      while (idx >= 0 && fieldRefs[idx].isIndex()) {
        array_derefs++;
        idx--;
      }
      mlir::Type currTy;
      if (idx >= 0) {
        currTy = fieldRefs[idx].getField().getType();
      } else {
        currTy = blockArg.getType();
      }
      while (array_derefs > 0) {
        currTy = mlir::dyn_cast<ArrayType>(currTy).getElementType();
      }
      return currTy;
    }
  }

  bool isConstant() const { return constFelt != nullptr; }
  bool isFelt() const { mlir::isa<FeltType>(getType()); }
  bool isIndex() const { mlir::isa<mlir::IndexType>(getType()); }
  bool isInteger() const { mlir::isa<mlir::IntegerType>(getType()); }
  bool isScalar() const { return isConstant() || isFelt() || isIndex() || isInteger(); }
  unsigned getInputNum() const { return blockArg.getArgNumber(); }

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
    if (isConstant() || fieldRefs.empty()) {
      return mlir::failure();
    }
    auto copy = *this;
    copy.fieldRefs.pop_back();
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
   * (public means an output) Otherwise, it is an input, either public or private.
   */
  mlir::BlockArgument blockArg;
  std::vector<ConstrainRefIndex> fieldRefs;
  FeltConstantOp constFelt;
};

mlir::raw_ostream &operator<<(mlir::raw_ostream &os, const ConstrainRef &rhs);

} // namespace llzk