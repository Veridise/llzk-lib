#pragma once

#include "llzk/Dialect/LLZK/IR/Ops.h"
#include "llzk/Dialect/LLZK/Util/Hash.h"

#include <mlir/Analysis/DataFlowFramework.h>
#include <mlir/Pass/AnalysisManager.h>

#include <llvm/ADT/EquivalenceClasses.h>

#include <map>
#include <memory>

namespace mlir {

class DataFlowSolver;

} // namespace mlir

namespace llzk {

/// @brief Defines a signal usage.
/// A signal usage is:
/// - The block argument index (all signals, even internal, are referenced via inputs arguments)
///   - "self", or internal signals, will always be block argument 0.
/// - The field definitions within the argument, if present. This would be if an input argument is
/// another struct.
///   - Like array references, this may be nested (e.g., signal foo of struct X within struct Y
///   would be Y[X[foo]]).
class SignalUsage {
public:
  /// Try to create SignalUsages out of a given operation.
  /// A single operation may contain multiple usages, e.g. addition of signals.
  static mlir::FailureOr<std::vector<SignalUsage>> get(mlir::Value val);

  explicit SignalUsage(mlir::BlockArgument b) : blockArg(b), fieldRefs({}), constFelt(nullptr) {}
  SignalUsage(mlir::BlockArgument b, std::vector<FieldDefOp> f)
      : blockArg(b), fieldRefs(f), constFelt(nullptr) {}
  explicit SignalUsage(FeltConstantOp c) : blockArg(nullptr), fieldRefs({}), constFelt(c) {}

  bool isConstant() const { return constFelt != nullptr; }
  unsigned getInputNum() const { return blockArg.getArgNumber(); }

  /// @brief Resolve
  /// @param other
  /// @return
  mlir::FailureOr<SignalUsage> translate(const SignalUsage &prefix, const SignalUsage &other) const;

  void print(mlir::raw_ostream &os) const {
    if (isConstant()) {
      os << "<constfelt: " << const_cast<FeltConstantOp &>(constFelt).getValueAttr().getValue()
         << ">";
    } else {
      os << "\%arg" << blockArg.getArgNumber();
      for (auto f : fieldRefs) {
        os << "[@" << f.getName() << "]";
      }
    }
  }

  bool operator==(const SignalUsage &rhs) const {
    return blockArg == rhs.blockArg && fieldRefs == rhs.fieldRefs;
  }

  // required for EquivalenceClasses usage
  bool operator<(const SignalUsage &rhs) const {
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

  struct Hash {
    size_t operator()(const SignalUsage &val) const {
      if (val.isConstant()) {
        return OpHash<FeltConstantOp>{}(val.constFelt);
      } else {
        size_t hash = std::hash<unsigned>{}(val.blockArg.getArgNumber());
        for (auto f : val.fieldRefs) {
          hash ^= OpHash<FieldDefOp>{}(f);
        }
        return hash;
      }
    }
  };

private:
  /**
   * If the block arg is 0, then it refers to "self", meaning the signal is internal or an output
   * (public means an output) Otherwise, it is an input, either public or private.
   */
  mlir::BlockArgument blockArg;
  std::vector<FieldDefOp> fieldRefs;
  FeltConstantOp constFelt;
};

using SignalUsageRemappings = std::vector<std::pair<SignalUsage, SignalUsage>>;

static inline mlir::raw_ostream &operator<<(mlir::raw_ostream &os, const SignalUsage &s) {
  s.print(os);
  return os;
}

/// @brief A summary of constraints enforced by an LLZK struct.
class ConstraintSummary {
public:
  static mlir::FailureOr<ConstraintSummary> compute(
      mlir::ModuleOp mod, StructDefOp s, mlir::DataFlowSolver &solver, mlir::AnalysisManager &am
  );

  void dump() const;
  void print(llvm::raw_ostream &os) const;

  /// Omit untranslated signals, those are internal.
  ConstraintSummary translate(SignalUsageRemappings translation);

private:
  mlir::ModuleOp mod;
  StructDefOp structDef;
  llvm::EquivalenceClasses<SignalUsage> constraintSets;

  ConstraintSummary(mlir::ModuleOp m, StructDefOp s) : mod(m), structDef(s), constraintSets() {}

  mlir::LogicalResult computeConstraints(mlir::DataFlowSolver &solver, mlir::AnalysisManager &am);

  void walkConstrainOp(mlir::DataFlowSolver &solver, mlir::Operation *emitOp);
};

class ConstraintSummaryModuleAnalysis;

/// @brief An analysis wrapper around the ConstraintSummary that performs additional checks.
class ConstraintSummaryAnalysis {
public:
  ConstraintSummaryAnalysis(mlir::Operation *op);

  mlir::LogicalResult
  constructSummary(mlir::DataFlowSolver &solver, mlir::AnalysisManager &moduleAnalysisManager);

  ConstraintSummary &getSummary() {
    ensureSummaryCreated();
    return *summary;
  }
  const ConstraintSummary &getSummary() const {
    ensureSummaryCreated();
    return *summary;
  }

private:
  mlir::ModuleOp modOp;
  StructDefOp structDefOp;
  std::shared_ptr<ConstraintSummary> summary;

  void ensureSummaryCreated() const {
    if (!summary) {
      llvm::report_fatal_error("constraint summary does not exist; must invoke constructSummary");
    }
  }

  friend class ConstraintSummaryModuleAnalysis;
};

///
class ConstraintSummaryModuleAnalysis {
  // Using a map to keep insertion order for iteration.
  using SummaryMap = std::map<StructDefOp, std::shared_ptr<ConstraintSummary>>;

public:
  ConstraintSummaryModuleAnalysis(mlir::Operation *op, mlir::AnalysisManager &am);

  bool hasSummary(StructDefOp op) const { return summaries.find(op) != summaries.end(); }
  ConstraintSummary &getSummary(StructDefOp op) { return *summaries.at(op); }
  const ConstraintSummary &getSummary(StructDefOp op) const { return *summaries.at(op); }

  SummaryMap::iterator begin() { return summaries.begin(); }
  SummaryMap::iterator end() { return summaries.end(); }
  SummaryMap::const_iterator cbegin() const { return summaries.cbegin(); }
  SummaryMap::const_iterator cend() const { return summaries.cend(); }

private:
  SummaryMap summaries;

  /// @brief Mark all operations from the top and included in the top operation
  /// as live so the solver will perform dataflow analyses.
  /// @param solver
  /// @param top
  void makeLive(mlir::DataFlowSolver &solver, mlir::Operation *top);
};

} // namespace llzk