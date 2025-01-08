#pragma once

#include "llzk/Dialect/LLZK/Analysis/ConstrainRef.h"
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

using ConstrainRefRemappings = std::vector<std::pair<ConstrainRef, ConstrainRef>>;

/// @brief A summary of constraints enforced by an LLZK struct.
/// The summary
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
  std::set<ConstrainRef> getConstrainingValues(const ConstrainRef &ref) const;

  /*
  Rule of three, needed for the mlir::SymbolTableCollection, which has no copy constructor.
  Since the mlir::SymbolTableCollection is a caching mechanism, we simply allow default, empty
  construction for copies.
  */

  /// Copy constructor.
  ConstraintSummary(const ConstraintSummary &other)
      : mod(other.mod), structDef(other.structDef), constraintSets(other.constraintSets), tables() {
  }
  /// Copy assignment.
  ConstraintSummary &operator=(const ConstraintSummary &other) {
    mod = other.mod;
    structDef = other.structDef;
    constraintSets = other.constraintSets;
  }
  /// Destructor. Just default.
  ~ConstraintSummary() = default;

private:
  mlir::ModuleOp mod;
  // Using mutable because many operations are not const by default, even for "const"-like
  // operations, like "getName()", and this reduces const_casts.
  mutable StructDefOp structDef;
  llvm::EquivalenceClasses<ConstrainRef> constraintSets;

  mutable mlir::SymbolTableCollection tables;

  StructDefOp getStructDef(StructType ty) const {
    auto sDef = ty.getDefinition(tables, mod);
    if (mlir::failed(sDef)) {
      llvm::report_fatal_error("could not find struct definition from struct type");
    }
    return sDef->get();
  }

  /// Try to create references out of a given operation.
  /// A single operation may contain multiple usages, e.g. addition of signals.
  mlir::FailureOr<std::vector<ConstrainRef>>
  getConstrainRefs(mlir::DataFlowSolver &solver, mlir::Value val);

  /// Produce all possible ConstraintRefs that are present from the struct's constrain function.
  std::vector<ConstrainRef> getAllConstrainRefs() const;

  /// Produce all possible ConstraintRefs that are present starting from the given BlockArgument.
  std::vector<ConstrainRef> getAllConstrainRefs(mlir::BlockArgument arg) const;

  /// Produce all possible ConstraintRefs that are present starting from the given
  /// BlockArgument and partially-specified indices into that object (fields).
  /// This produces refs for composite types (e.g., full structs and full arrays)
  /// as well as individual fields and constants.
  std::vector<ConstrainRef> getAllConstrainRefs(
      StructDefOp s, mlir::BlockArgument blockArg, std::vector<ConstrainRefIndex> fields = {}
  ) const;

  /// Produce all possible ConstraintRefs that are present starting from the given
  /// arrayField, originating from a given blockArg,
  /// and partially-specified indices into that object (fields).
  /// This produces refs for composite types (e.g., full structs and full arrays)
  /// as well as individual fields and constants.
  std::vector<ConstrainRef> getAllConstrainRefs(
      ArrayType arrayTy, mlir::BlockArgument blockArg, std::vector<ConstrainRefIndex> fields = {}
  ) const;

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
  struct StructDefOpLess {
    bool operator()(const StructDefOp &lhs, const StructDefOp &rhs) const {
      // Try sorting by location first, then name.
      auto lhsLoc = lhs->getLoc().dyn_cast<mlir::FileLineColLoc>();
      auto rhsLoc = rhs->getLoc().dyn_cast<mlir::FileLineColLoc>();
      if (lhsLoc && rhsLoc) {
        auto filenameCmp = lhsLoc.getFilename().compare(rhsLoc.getFilename());
        return filenameCmp < 0 || (filenameCmp == 0 && lhsLoc.getLine() < rhsLoc.getLine()) ||
               (filenameCmp == 0 && lhsLoc.getLine() == rhsLoc.getLine() &&
                lhsLoc.getColumn() < rhsLoc.getColumn());
      }

      auto lhsName = const_cast<StructDefOp &>(lhs).getName();
      auto rhsName = const_cast<StructDefOp &>(rhs).getName();
      return lhsName.compare(rhsName) < 0;
    }
  };
  /// Using a map, not an unordered map, to control sorting order for iteration.
  using SummaryMap = std::map<StructDefOp, std::shared_ptr<ConstraintSummary>, StructDefOpLess>;

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
    if (!hasSummary(op)) {
      llvm::report_fatal_error(
          "constraint summary does not exist for StructDefOp " + mlir::Twine(op.getName())
      );
    }
  }
};

} // namespace llzk