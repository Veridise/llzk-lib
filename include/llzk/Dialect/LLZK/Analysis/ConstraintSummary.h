#pragma once

#include "llzk/Dialect/LLZK/IR/Ops.h"
#include "llzk/Dialect/LLZK/Util/Hash.h"

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
  static mlir::FailureOr<std::vector<SignalUsage>> get(mlir::Value val) {
    std::vector<SignalUsage> res;

    // If it's a field read, it reads a field def from a component.
    // If it's a felt, it doesn't need a field read

    // Due to the way constrain is defined, all signals are read from inputs.
    if (auto blockArg = mlir::dyn_cast_or_null<mlir::BlockArgument>(val)) {
      // to use this constructor, the block arg must be a felt
      res.push_back(SignalUsage(blockArg.getArgNumber()));
    } else if (auto fieldRead = mlir::dyn_cast_or_null<FieldReadOp>(val.getDefiningOp())) {
      std::deque<FieldDefOp> fields;
      mlir::SymbolTableCollection tables;
      mlir::BlockArgument arg;
      FieldReadOp currRead = fieldRead;
      while (currRead != nullptr) {
        auto component = currRead.getComponent();
        auto res = currRead.getFieldDefOp(tables);
        if (mlir::failed(res)) {
          fieldRead.emitError() << "could not find field read\n";
          return mlir::failure();
        }
        fields.push_front(res->get());
        arg = mlir::dyn_cast_or_null<mlir::BlockArgument>(component);
        currRead = mlir::dyn_cast_or_null<FieldReadOp>(component.getDefiningOp());
      }
      if (arg == nullptr) {
        fieldRead.emitError() << "could not follow a read chain!\n";
        return mlir::failure();
      }
      // We only want to generate this if the end value is a felt
      res.push_back(
          SignalUsage(arg.getArgNumber(), std::vector<FieldDefOp>(fields.begin(), fields.end()))
      );
    } else if (val.getDefiningOp() != nullptr && mlir::isa<FeltConstantOp>(val.getDefiningOp())) {
      auto constFelt = mlir::dyn_cast<FeltConstantOp>(val.getDefiningOp());
      res.push_back(SignalUsage(constFelt));
    } else if (val.getDefiningOp() != nullptr && !mlir::isa<FuncOp>(val.getDefiningOp())) {
      // Fallback for non-function calls
      llvm::errs() << "fallback: " << *val.getDefiningOp() << "\n";
      for (auto operand : val.getDefiningOp()->getOperands()) {
        auto uses = SignalUsage::get(operand);
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

  SignalUsage(unsigned b) : blockArgIdx(b), fieldRefs({}), constFelt(nullptr) {}
  SignalUsage(unsigned b, std::vector<FieldDefOp> f)
      : blockArgIdx(b), fieldRefs(f), constFelt(nullptr) {}
  SignalUsage(FeltConstantOp c) : blockArgIdx(0), fieldRefs({}), constFelt(c) {}

  unsigned getInputNum() const { return blockArgIdx; }

  /// @brief Resolve
  /// @param other
  /// @return
  mlir::FailureOr<SignalUsage>
  translate(const SignalUsage &prefix, const SignalUsage &other) const {
    if (blockArgIdx != prefix.blockArgIdx || fieldRefs.size() < prefix.fieldRefs.size()) {
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

  void print(mlir::raw_ostream &os) const {
    if (constFelt) {
      os << "<const: " << const_cast<FeltConstantOp &>(constFelt) << ">";
    } else {
      os << "<input: " << getInputNum();
      for (auto f : fieldRefs) {
        os << ", field: " << f.getName();
      }
      os << ">";
    }
  }

  bool operator==(const SignalUsage &rhs) const {
    return blockArgIdx == rhs.blockArgIdx && fieldRefs == rhs.fieldRefs;
  }

  // required for EquivalenceClasses usage
  bool operator<(const SignalUsage &rhs) const {
    if (blockArgIdx < rhs.blockArgIdx) {
      return true;
    } else if (blockArgIdx > rhs.blockArgIdx) {
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
      size_t hash = std::hash<unsigned>{}(val.blockArgIdx);
      for (auto f : val.fieldRefs) {
        hash ^= OpHash<FieldDefOp>{}(f);
      }
      return hash;
    }
  };

private:
  /**
   * If the block arg is 0, then it refers to "self", meaning the signal is internal or an output
   * (public means an output) Otherwise, it is an input, either public or private.
   */
  unsigned blockArgIdx;
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