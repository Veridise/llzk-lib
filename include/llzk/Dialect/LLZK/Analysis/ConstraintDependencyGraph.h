#pragma once

#include "llzk/Dialect/LLZK/Analysis/ConstrainRef.h"
#include "llzk/Dialect/LLZK/IR/Ops.h"
#include "llzk/Dialect/LLZK/Util/Compare.h"
#include "llzk/Dialect/LLZK/Util/ErrorHelper.h"
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

class ConstrainRefLatticeValue;

using ConstrainRefRemappings = std::vector<std::pair<ConstrainRef, ConstrainRefLatticeValue>>;

/// @brief A dependency graph of constraints enforced by an LLZK struct.
///
/// Mathmatically speaking, a constraint dependency graph (CDG) is a transitive closure
/// of edges where there is an edge between signals `a` and `b`
/// iff `a` and `b` appear in the same constraint.
///
/// Less formally, a CDG is a set of signals that constrain one another through
/// one or more emit operations (`emit_in` or `emit_eq`). The CDG only
/// indicate that signals are connected by constraints, but do not include information
/// about the type of computation that binds them together.
///
/// For example, a CDG of the form: {
///     {%arg1, %arg2, %arg3[@foo]}
/// }
/// Means that %arg1, %arg2, and field @foo of %arg3, are connected
/// via some constraints. These constraints could take the form of (in Circom notation):
///     %arg1 + %arg3[@foo] === %arg2
/// Or
///     %arg2 === %arg2 / %arg3[@foo]
/// Or any other form of constraint including those values.
///
/// The CDG also records information about constant values (e.g., constfelt) that
/// are included in constraints, but does not compute a transitive closure over
/// constant values, as constant value usage in constraints does not imply any
/// dependency between signal values (e.g., constraints a + b === 0 and c + d === 0 both use
/// constant 0, but does not enforce a dependency between a, b, c, and d).
class ConstraintDependencyGraph {
public:
  /// @brief Compute a ConstraintDependencyGraph (CDG)
  /// @param mod The LLZK-complaint module that is the parent of struct `s`.
  /// @param s The struct to compute the CDG for.
  /// @param solver A pre-configured DataFlowSolver. The liveness of the struct must
  /// already be computed in this solver in order for the constraint analysis to run.
  /// @param am A module-level analysis manager. This analysis manager needs to originate
  /// from a module-level analysis (i.e., for the `mod` module) so that analyses
  /// for other constraints can be queried via the getChildAnalysis method.
  /// @return
  static mlir::FailureOr<ConstraintDependencyGraph> compute(
      mlir::ModuleOp mod, StructDefOp s, mlir::DataFlowSolver &solver, mlir::AnalysisManager &am
  );

  /// @brief Dumps the CDG to stderr.
  void dump() const;
  /// @brief Print the CDG to the specified output stream.
  /// @param os The LLVM/MLIR output stream.
  void print(mlir::raw_ostream &os) const;

  /// @brief Translate the ConstrainRefs in this CDG to that of a different
  /// context. Used to translate a CDG of a struct to a CDG for a called subcomponent.
  /// @param translation A vector of mappings of current reference prefix -> translated reference
  /// prefix.
  /// @return A CDG that contains only translated references. Non-constant references with
  /// no translation are omitted. This omissions allows calling components to ignore internal
  /// references within subcomponents that are inaccessible to the caller.
  ConstraintDependencyGraph translate(ConstrainRefRemappings translation);

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

  ConstraintDependencyGraph(const ConstraintDependencyGraph &other)
      : mod(other.mod), structDef(other.structDef), signalSets(other.signalSets),
        constantSets(other.constantSets), tables() {}
  ConstraintDependencyGraph &operator=(const ConstraintDependencyGraph &other) {
    mod = other.mod;
    structDef = other.structDef;
    signalSets = other.signalSets;
    constantSets = other.constantSets;
  }
  ~ConstraintDependencyGraph() = default;

private:
  mlir::ModuleOp mod;
  // Using mutable because many operations are not const by default, even for "const"-like
  // operations, like "getName()", and this reduces const_casts.
  mutable StructDefOp structDef;

  // Transitive closure only over signals.
  llvm::EquivalenceClasses<ConstrainRef> signalSets;
  // A simple set mapping of constants, as we do not want to compute a transitive closure over
  // constants.
  std::unordered_map<ConstrainRef, ConstrainRefSet, ConstrainRef::Hash> constantSets;

  // Also mutable for caching within otherwise const lookup operations.
  mutable mlir::SymbolTableCollection tables;

  /// @brief Constructs an empty CDG. The CDG is populated using computeConstraints.
  /// @param m The parent LLZK-compliant module.
  /// @param s The struct to analyze.
  ConstraintDependencyGraph(mlir::ModuleOp m, StructDefOp s) : mod(m), structDef(s), signalSets() {}

  /// @brief Runs the constraint analysis to compute a transitive closure over ConstrainRefs
  /// as operated over by emit operations.
  /// @param solver The pre-configured solver.
  /// @param am The module-level AnalysisManager.
  /// @return mlir::success() if no issues were encountered, mlir::failure() otherwise
  mlir::LogicalResult computeConstraints(mlir::DataFlowSolver &solver, mlir::AnalysisManager &am);

  /// @brief Update the signalSets EquivalenceClasses based on the given
  /// emit operation. Relies on the caller to verify that `emitOp` is either
  /// an EmitEqualityOp or an EmitContainmentOp, as the logic for both is currently
  /// the same.
  /// @param solver The pre-configured solver.
  /// @param emitOp The emit operation that is creating a constraint.
  void walkConstrainOp(mlir::DataFlowSolver &solver, mlir::Operation *emitOp);
};

/// @brief A module-level analysis for constructing ConstraintDependencyGraph objects for
/// all structs in the given LLZK module.
class ConstraintDependencyGraphModuleAnalysis {
  /// Using a map, not an unordered map, to control sorting order for iteration.
  using DependencyMap = std::map<
      StructDefOp, std::shared_ptr<ConstraintDependencyGraph>, OpLocationLess<StructDefOp>>;

public:
  /// @brief Computes ConstraintDependencyGraph objects for all structs contained within the
  /// given op, if the op is a module op.
  /// @param op The top-level op. If op is not an LLZK-compliant mlir::ModuleOp, the
  /// analysis will fail.
  /// @param am The analysis manager used to query sub-analyses per StructDefOperation.
  ConstraintDependencyGraphModuleAnalysis(mlir::Operation *op, mlir::AnalysisManager &am);

  bool hasCDG(StructDefOp op) const { return dependencies.find(op) != dependencies.end(); }
  ConstraintDependencyGraph &getCDG(StructDefOp op) {
    ensureCDGCreated(op);
    return *dependencies.at(op);
  }
  const ConstraintDependencyGraph &getCDG(StructDefOp op) const {
    ensureCDGCreated(op);
    return *dependencies.at(op);
  }

  DependencyMap::iterator begin() { return dependencies.begin(); }
  DependencyMap::iterator end() { return dependencies.end(); }
  DependencyMap::const_iterator cbegin() const { return dependencies.cbegin(); }
  DependencyMap::const_iterator cend() const { return dependencies.cend(); }

private:
  DependencyMap dependencies;

  /// @brief Ensures that the given struct has a CDG.
  /// @param op The struct to ensure has a CDG.
  void ensureCDGCreated(StructDefOp op) const {
    ensure(hasCDG(op), "CDG does not exist for StructDefOp " + mlir::Twine(op.getName()));
  }
};

} // namespace llzk
