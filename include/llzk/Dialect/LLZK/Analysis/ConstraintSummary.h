#pragma once

#include "llzk/Dialect/LLZK/Analysis/ConstrainRef.h"
#include "llzk/Dialect/LLZK/IR/Ops.h"
#include "llzk/Dialect/LLZK/Util/Compare.h"
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

using ConstrainRefRemappings = std::vector<std::pair<ConstrainRef, ConstrainRef>>;

/// @brief A summary of constraints enforced by an LLZK struct.
/// A constraint summary is a set of values that constrain one another through
/// one or more emit operations (`emit_in` or `emit_eq`). The summaries only
/// indicate that values are connected by constraints, but do not include information
/// about the type of computation that binds them together.
///
/// For example, a constraint summary of the form: {
///     {%arg1, %arg2[@foo], <constfelt: 1>}
/// }
/// Means that %arg1, field @foo of %arg2, and the constant felt 1 are connected
/// via some constraints. These constraints could take the form of (in Circom notation):
///     %arg1 + %arg2[@foo] === 1
/// Or
///     %arg1 === 1 / %arg2[@foo]
/// Or any other form of constraint including those values.
class ConstraintSummary {
public:
  /// @brief Compute a ConstraintSummary
  /// @param mod The LLZK-complaint module that is the parent of struct `s`.
  /// @param s The struct to compute the summary for.
  /// @param solver A pre-configured DataFlowSolver. The liveness of the struct must
  /// already be computed in this solver in order for the constraint analysis to run.
  /// @param am A module-level analysis manager. This analysis manager needs to originate
  /// from a module-level analysis (i.e., for the `mod` module) so that analyses
  /// for other constraints can be queried via the getChildAnalysis method.
  /// @return
  static mlir::FailureOr<ConstraintSummary> compute(
      mlir::ModuleOp mod, StructDefOp s, mlir::DataFlowSolver &solver, mlir::AnalysisManager &am
  );

  /// @brief Dumps the ConstraintSummary to stderr.
  void dump() const;
  /// @brief Print the constraintSummary to the specified output stream.
  /// @param os The LLVM/MLIR output stream.
  void print(mlir::raw_ostream &os) const;

  /// @brief Translate the ConstrainRefs in this summary to that of a different
  /// context. Used to translate a summary of a struct to a summary for a called subcomponent.
  /// @param translation A vector of mappings of current reference prefix -> translated reference
  /// prefix.
  /// @return A summary that contains only translated references. Non-constant references with
  /// no translation are omitted. This omissions allows calling components to ignore internal
  /// references within subcomponents that are inaccessible to the caller.
  ConstraintSummary translate(ConstrainRefRemappings translation);

  /// @brief Get the values that are connected to the given ref via emitted constraints.
  /// This method looks for constraints to the value in the ref and constraints to any
  /// prefix of this value.
  /// For example, if ref is an array element (foo[2]), this looks for constraints on
  /// foo[2] as well as foo, as arrays may be constrained in their entirity via emit_in operations.
  /// @param ref
  /// @return The set of references that are connected to ref via constraints.
  ConstrainRefSet getConstrainingValues(const ConstrainRef &ref) const;

  /*
  Rule of three, needed for the mlir::SymbolTableCollection, which has no copy constructor.
  Since the mlir::SymbolTableCollection is a caching mechanism, we simply allow default, empty
  construction for copies.
  */

  ConstraintSummary(const ConstraintSummary &other)
      : mod(other.mod), structDef(other.structDef), constraintSets(other.constraintSets), tables() {
  }
  ConstraintSummary &operator=(const ConstraintSummary &other) {
    mod = other.mod;
    structDef = other.structDef;
    constraintSets = other.constraintSets;
  }
  ~ConstraintSummary() = default;

private:
  mlir::ModuleOp mod;
  // Using mutable because many operations are not const by default, even for "const"-like
  // operations, like "getName()", and this reduces const_casts.
  mutable StructDefOp structDef;
  llvm::EquivalenceClasses<ConstrainRef> constraintSets;

  // Also mutable for caching within otherwise const lookup operations.
  mutable mlir::SymbolTableCollection tables;

  /// @brief Constructs an empty summary. The summary is populated using computeConstraints.
  /// @param m The parent LLZK-compliant module.
  /// @param s The struct to summarize.
  ConstraintSummary(mlir::ModuleOp m, StructDefOp s) : mod(m), structDef(s), constraintSets() {}

  /// @brief Runs the constraint analysis to compute a transitive closure over ConstrainRefs
  /// as operated over by emit operations.
  /// @param solver The pre-configured solver.
  /// @param am The module-level AnalysisManager.
  /// @return mlir::success() if no issues were encountered, mlir::failure() otherwise
  mlir::LogicalResult computeConstraints(mlir::DataFlowSolver &solver, mlir::AnalysisManager &am);

  /// @brief Update the constraintSets EquivalenceClasses based on the given
  /// emit operation. Relies on the caller to verify that `emitOp` is either
  /// an EmitEqualityOp or an EmitContainmentOp, as the logic for both is currently
  /// the same.
  /// @param solver The pre-configured solver.
  /// @param emitOp The emit operation that is creating a constraint.
  void walkConstrainOp(mlir::DataFlowSolver &solver, mlir::Operation *emitOp);
};

/// @brief A module-level analysis for constructing ConstraintSummary objects for
/// all structs in the given LLZK module.
class ConstraintSummaryModuleAnalysis {
  /// Using a map, not an unordered map, to control sorting order for iteration.
  using SummaryMap =
      std::map<StructDefOp, std::shared_ptr<ConstraintSummary>, OpLocationLess<StructDefOp>>;

public:
  /// @brief Computes ConstraintSummary objects for all structs contained within the
  /// given op, if the op is a module op.
  /// @param op The top-level op. If op is not an LLZK-compliant mlir::ModuleOp, the
  /// analysis will fail.
  /// @param am The analysis manager used to query sub-analyses per StructDefOperation.
  ConstraintSummaryModuleAnalysis(mlir::Operation *op, mlir::AnalysisManager &am);

  bool hasSummary(StructDefOp op) const { return summaries.find(op) != summaries.end(); }
  ConstraintSummary &getSummary(StructDefOp op) {
    ensureSummaryCreated(op);
    return *summaries.at(op);
  }
  const ConstraintSummary &getSummary(StructDefOp op) const {
    ensureSummaryCreated(op);
    return *summaries.at(op);
  }

  SummaryMap::iterator begin() { return summaries.begin(); }
  SummaryMap::iterator end() { return summaries.end(); }
  SummaryMap::const_iterator cbegin() const { return summaries.cbegin(); }
  SummaryMap::const_iterator cend() const { return summaries.cend(); }

private:
  SummaryMap summaries;

  /// @brief Ensures that the given struct has a summary.
  /// @param op The struct to ensure has a summary.
  void ensureSummaryCreated(StructDefOp op) const {
    debug::ensure(
        hasSummary(op),
        "constraint summary does not exist for StructDefOp " + mlir::Twine(op.getName())
    );
  }
};

} // namespace llzk
