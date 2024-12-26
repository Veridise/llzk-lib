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

/// @brief A summary of constraints enforced by an LLZK struct.
class ConstraintSummary {
public:
  ConstraintSummary(StructDefOp s, mlir::DataFlowSolver &solver, mlir::AnalysisManager &am);

  void dump() const;
  void print(llvm::raw_ostream &os) const;

private:
  StructDefOp structDef;
  llvm::EquivalenceClasses<SignalUsage> constrainSets;
};

/// @brief An analysis wrapper around the ConstraintSummary that performs additional checks.
class ConstraintSummaryAnalysis {
  // Using a map to keep insertion order for iteration.
  using SummaryMap = std::map<StructDefOp, ConstraintSummary>;

public:
  ConstraintSummaryAnalysis(mlir::Operation *op, mlir::AnalysisManager &am);

  ConstraintSummary &getSummary(StructDefOp op) { return summaries.at(op); }
  const ConstraintSummary &getSummary(StructDefOp op) const { return summaries.at(op); }

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