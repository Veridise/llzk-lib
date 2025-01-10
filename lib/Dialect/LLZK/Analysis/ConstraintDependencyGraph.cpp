#include "llzk/Dialect/LLZK/Analysis/ConstraintDependencyGraph.h"
#include "llzk/Dialect/LLZK/Analysis/DenseAnalysis.h"
#include "llzk/Dialect/LLZK/Util/Hash.h"
#include "llzk/Dialect/LLZK/Util/SymbolHelper.h"

#include <mlir/Analysis/DataFlow/DeadCodeAnalysis.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Index/IR/IndexOps.h>
#include <mlir/IR/Value.h>

#include <numeric>
#include <unordered_set>

namespace llzk {

/*
Private Utilities:

These classes are defined here and not in the header as they are not designed
for use outside of this specific ConstraintDependencyGraph analysis.
*/

using ConstantMap = mlir::DenseMap<mlir::Value, mlir::APInt>;
using ConstrainRefSetMap = mlir::DenseMap<mlir::Value, ConstrainRefSet>;
using ArgumentMap = mlir::DenseMap<mlir::BlockArgument, ConstrainRefSet>;

/// A lattice for use in dense analysis.
class ConstrainRefLattice : public dataflow::AbstractDenseLattice {
public:
  using AbstractDenseLattice::AbstractDenseLattice;

  /* Static utilities */

  /// If val is the source of other values (i.e., a block argument from the function
  /// args or a constant), create the base reference to the val. Otherwise,
  /// return failure.
  /// Our lattice values must originate from somewhere.
  static mlir::FailureOr<ConstrainRef> getSourceRef(mlir::Value val) {
    if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(val)) {
      return ConstrainRef(blockArg);
    } else if (auto constFelt = mlir::dyn_cast<FeltConstantOp>(val.getDefiningOp())) {
      return ConstrainRef(constFelt);
    } else if (auto constIdx = mlir::dyn_cast<mlir::index::ConstantOp>(val.getDefiningOp())) {
      return ConstrainRef(constIdx);
    }
    return mlir::failure();
  }

  /* Required methods */

  /// Maximum upper bound
  mlir::ChangeResult join(const AbstractDenseLattice &rhs) override {
    if (auto *r = dynamic_cast<const ConstrainRefLattice *>(&rhs)) {
      return setValues(r->refSetMap);
    }
    llvm::report_fatal_error("invalid join lattice type");
    return mlir::ChangeResult::NoChange;
  }

  /// Minimum lower bound
  virtual mlir::ChangeResult meet(const AbstractDenseLattice &rhs) override {
    llvm::report_fatal_error("meet operation is not supported for ConstrainRefLattice");
    return mlir::ChangeResult::NoChange;
  }

  void print(mlir::raw_ostream &os) const override {
    os << "ConstrainRefLattice { ";
    for (auto mit = refSetMap.begin(); mit != refSetMap.end();) {
      auto &[v, refSet] = *mit;
      os << "\n    (" << v << ") => { ";
      for (auto it = refSet.begin(); it != refSet.end();) {
        it->print(os);
        it++;
        if (it != refSet.end()) {
          os << ", ";
        }
      }
      mit++;
      if (mit != refSetMap.end()) {
        os << " },";
      } else {
        os << " }\n";
      }
    }
    os << "}\n";
  }

  /* Update utility methods */

  mlir::ChangeResult setValues(const ConstrainRefSetMap &rhs) {
    auto res = mlir::ChangeResult::NoChange;

    for (auto &[v, s] : rhs) {
      res |= setValue(v, s);
    }
    return res;
  }

  mlir::ChangeResult setValue(mlir::Value v, const ConstrainRefSet &rhs) {
    auto res = mlir::ChangeResult::NoChange;

    for (auto &r : rhs) {
      auto [_, inserted] = refSetMap[v].insert(r);
      res = inserted ? mlir::ChangeResult::Change : res;
    }
    return res;
  }

  mlir::ChangeResult setValue(mlir::Value v, const ConstrainRef &ref) {
    auto [_, inserted] = refSetMap[v].insert(ref);
    return inserted ? mlir::ChangeResult::Change : mlir::ChangeResult::NoChange;
  }

  ConstrainRefSet getOrDefault(mlir::Value v) const {
    if (refSetMap.find(v) == refSetMap.end()) {
      auto sourceRef = getSourceRef(v);
      if (mlir::succeeded(sourceRef)) {
        return {sourceRef.value()};
      }
      return {};
    }
    return refSetMap.at(v);
  }

  ConstrainRefSet getReturnValue(unsigned i) const {
    auto op = this->getPoint().get<mlir::Operation *>();
    if (auto retOp = mlir::dyn_cast<ReturnOp>(op)) {
      if (i >= retOp.getNumOperands()) {
        llvm::report_fatal_error("return value requested is out of range");
      }
      return this->getOrDefault(retOp.getOperand(i));
    }
    return ConstrainRefSet();
  }

private:
  ConstrainRefSetMap refSetMap;
};

mlir::raw_ostream &operator<<(mlir::raw_ostream &os, const ConstrainRefLattice &v) {
  v.print(os);
  return os;
}

/// @brief The dataflow analysis that computes the set of references that
/// LLZK operations use and produce. The analysis is simple: any operation will
/// simply output a union of its input references, regardless of what type of
/// operation it performs, as the analysis is operator-insensitive.
class ConstrainRefAnalysis : public dataflow::DenseForwardDataFlowAnalysis<ConstrainRefLattice> {
public:
  using dataflow::DenseForwardDataFlowAnalysis<ConstrainRefLattice>::DenseForwardDataFlowAnalysis;

  void visitCallControlFlowTransfer(
      mlir::CallOpInterface call, dataflow::CallControlFlowAction action,
      const ConstrainRefLattice &before, ConstrainRefLattice *after
  ) override {

    auto fnOpRes = resolveCallable<FuncOp>(tables, call);
    debug::ensure(succeeded(fnOpRes), "could not resolve called function");

    auto fnOp = fnOpRes->get();
    if (fnOp.getName() == FUNC_NAME_CONSTRAIN || fnOp.getName() == FUNC_NAME_COMPUTE) {
      // Do nothing special.
      join(after, before);
      return;
    }
    /// `action == CallControlFlowAction::Enter` indicates that:
    ///   - `before` is the state before the call operation;
    ///   - `after` is the state at the beginning of the callee entry block;
    else if (action == dataflow::CallControlFlowAction::EnterCallee) {
      // Add all of the argument values to the lattice.
      auto calledFnRes = resolveCallable<FuncOp>(tables, call);
      debug::ensure(mlir::succeeded(calledFnRes), "could not resolve function call");
      auto calledFn = calledFnRes->get();

      auto updated = after->join(before);
      for (auto arg : calledFn->getRegion(0).getArguments()) {
        auto sourceRef = ConstrainRefLattice::getSourceRef(arg);
        debug::ensure(mlir::succeeded(sourceRef), "failed to get source ref");
        updated |= after->setValue(arg, sourceRef.value());
      }
      propagateIfChanged(after, updated);
    }
    /// `action == CallControlFlowAction::Exit` indicates that:
    ///   - `before` is the state at the end of a callee exit block;
    ///   - `after` is the state after the call operation.
    else if (action == dataflow::CallControlFlowAction::ExitCallee) {
      // Translate argument values based on the operands given at the call site.
      std::unordered_map<ConstrainRef, ConstrainRefSet, ConstrainRef::Hash> translation;
      auto funcOpRes = resolveCallable<FuncOp>(tables, call);
      debug::ensure(mlir::succeeded(funcOpRes), "could not lookup called function");
      auto funcOp = funcOpRes->get();

      auto callOp = mlir::dyn_cast<CallOp>(call.getOperation());
      debug::ensure(callOp, "call is not a llzk::CallOp");

      for (unsigned i = 0; i < funcOp.getNumArguments(); i++) {
        auto key = ConstrainRef(funcOp.getArgument(i));
        auto val = before.getOrDefault(callOp.getOperand(i));
        translation[key] = val;
      }

      mlir::ChangeResult updated = after->join(before);
      for (unsigned i = 0; i < callOp.getNumResults(); i++) {
        auto retRef = before.getReturnValue(i);
        ConstrainRefSet translated;
        for (auto &ref : retRef) {
          auto f = translation.find(ref);
          if (f != translation.end()) {
            auto &retVal = f->second;
            translated.insert(retVal.begin(), retVal.end());
          }
        }

        updated |= after->setValue(callOp->getResult(i), translated);
      }
      propagateIfChanged(after, updated);
    }
    // Note that `setToEntryState` may be a "partial fixpoint" for some
    // lattices, e.g., lattices that are lists of maps of other lattices will
    // only set fixpoint for "known" lattices.
    else if (action == mlir::dataflow::CallControlFlowAction::ExternalCallee) {
      setToEntryState(after);
    }
  }

  /// @brief Propagate constrain reference lattice values from operands to results.
  /// @param op
  /// @param before
  /// @param after
  void visitOperation(
      mlir::Operation *op, const ConstrainRefLattice &before, ConstrainRefLattice *after
  ) override {
    // Collect the references that are made by the operands to `op`.
    ConstrainRefSetMap operandVals;
    for (auto &operand : op->getOpOperands()) {
      operandVals[operand.get()] = before.getOrDefault(operand.get());
    }

    // Propagate existing state.
    join(after, before);

    // We will now join the the operand refs based on the type of operand.
    if (auto fieldRead = mlir::dyn_cast<FieldReadOp>(op)) {
      // In the readf case, the operand is indexed into by the read's fielddefop.
      assert(operandVals.size() == 1);
      assert(fieldRead->getNumResults() == 1);

      auto fieldOpRes = fieldRead.getFieldDefOp(tables);
      debug::ensure(mlir::succeeded(fieldOpRes), "could not find field read");

      auto res = fieldRead->getResult(0);
      const auto &ops = operandVals.at(fieldRead->getOpOperand(0).get());
      ConstrainRefSet fieldVals;
      for (auto &r : ops) {
        fieldVals.insert(r.createChild(ConstrainRefIndex(fieldOpRes.value())));
      }
      propagateIfChanged(after, after->setValue(res, fieldVals));
    } else if (auto arrayRead = mlir::dyn_cast<ReadArrayOp>(op)) {
      // In the readarr case, we index the first operand by all remaining indices
      assert(arrayRead->getNumResults() == 1);
      auto res = arrayRead->getResult(0);

      auto array = arrayRead.getOperand(0);
      auto currVals = operandVals[array];

      for (size_t i = 1; i < arrayRead.getNumOperands(); i++) {
        auto currentOp = arrayRead.getOperand(i);
        auto &idxVals = operandVals[currentOp];

        ConstrainRefSet newVals;
        if (idxVals.size() == 1 && idxVals.begin()->isConstantIndex()) {
          auto idxVal = *idxVals.begin();
          for (auto &r : currVals) {
            newVals.insert(r.createChild(idxVal));
          }
        } else {
          // Otherwise, assume any range is valid.
          auto arrayType = mlir::dyn_cast<ArrayType>(array.getType());
          auto lower = mlir::APInt::getZero(64);
          mlir::APInt upper(64, arrayType.getDimSize(i - 1));
          auto idxRange = ConstrainRefIndex(lower, upper);
          for (auto &r : currVals) {
            newVals.insert(r.createChild(idxRange));
          }
        }
        currVals = newVals;
      }

      propagateIfChanged(after, after->setValue(res, currVals));
    } else {
      // Standard union of operands into the results value.
      // TODO: Could perform constant computation/propagation here for, e.g., arithmetic
      // over constants, but such analysis may be better suited for a dedicated pass.
      propagateIfChanged(after, fallbackOpUpdate(op, operandVals, before, after));
    }
  }

protected:
  void setToEntryState(ConstrainRefLattice *lattice) override {
    // the entry state is empty, so do nothing.
  }

  // Perform a standard union of operands into the results value.
  mlir::ChangeResult fallbackOpUpdate(
      mlir::Operation *op, const ConstrainRefSetMap &operandVals, const ConstrainRefLattice &before,
      ConstrainRefLattice *after
  ) {
    auto updated = mlir::ChangeResult::NoChange;
    for (auto res : op->getResults()) {
      auto cur = before.getOrDefault(res);

      for (auto &[_, set] : operandVals) {
        cur.insert(set.begin(), set.end());
      }
      updated |= after->setValue(res, cur);
    }
    return updated;
  }

private:
  mlir::SymbolTableCollection tables;
};

/*
ConstraintDependencyGraphAnalysis

Needs to be declared before implementing the ConstraintDependencyGraph functions, as
they reference the ConstraintDependencyGraphAnalysis.
*/

/// @brief An analysis wrapper around the ConstraintDependencyGraph for a given struct.
/// This analysis is a StructDefOp-level analysis that should not be directly
/// interacted with---rather, it is a utility used by the ConstraintDependencyGraphModuleAnalysis
/// that helps use MLIR's AnalysisManager to cache summaries for sub-components.
class ConstraintDependencyGraphAnalysis {
public:
  ConstraintDependencyGraphAnalysis(mlir::Operation *op) {
    structDefOp = mlir::dyn_cast<StructDefOp>(op);
    if (!structDefOp) {
      auto error_message =
          "ConstraintDependencyGraphAnalysis expects provided op to be a StructDefOp!";
      op->emitError(error_message);
      llvm::report_fatal_error(error_message);
    }
    auto maybeModOp = getRootModule(op);
    if (mlir::failed(maybeModOp)) {
      auto error_message =
          "ConstraintDependencyGraphAnalysis could not find root module from StructDefOp!";
      op->emitError(error_message);
      llvm::report_fatal_error(error_message);
    }
    modOp = *maybeModOp;
  }

  /// @brief Construct a summary, using the module's analysis manager to query
  /// ConstraintDependencyGraph objects for nested components.
  mlir::LogicalResult
  constructSummary(mlir::DataFlowSolver &solver, mlir::AnalysisManager &moduleAnalysisManager) {
    auto summaryRes =
        ConstraintDependencyGraph::compute(modOp, structDefOp, solver, moduleAnalysisManager);
    if (mlir::failed(summaryRes)) {
      return mlir::failure();
    }
    summary = std::make_shared<ConstraintDependencyGraph>(*summaryRes);
    return mlir::success();
  }

  ConstraintDependencyGraph &getSummary() {
    ensureSummaryCreated();
    return *summary;
  }

  const ConstraintDependencyGraph &getSummary() const {
    ensureSummaryCreated();
    return *summary;
  }

private:
  mlir::ModuleOp modOp;
  StructDefOp structDefOp;
  std::shared_ptr<ConstraintDependencyGraph> summary;

  void ensureSummaryCreated() const {
    if (!summary) {
      llvm::report_fatal_error("constraint summary does not exist; must invoke constructSummary");
    }
  }

  friend class ConstraintDependencyGraphModuleAnalysis;
};

/* ConstraintDependencyGraph */

mlir::FailureOr<ConstraintDependencyGraph> ConstraintDependencyGraph::compute(
    mlir::ModuleOp m, StructDefOp s, mlir::DataFlowSolver &solver, mlir::AnalysisManager &am
) {
  ConstraintDependencyGraph summary(m, s);
  if (summary.computeConstraints(solver, am).failed()) {
    return mlir::failure();
  }
  return summary;
}

void ConstraintDependencyGraph::dump() const { print(llvm::errs()); }

/// Print all constraints. Any element that is unconstrained is omitted.
void ConstraintDependencyGraph::print(llvm::raw_ostream &os) const {
  // the EquivalenceClasses::iterator is sorted, but the EquivalenceClasses::member_iterator is
  // not guaranteed to be sorted. So, we will sort members before printing them.
  // We also want to add the constant values into the printing.
  std::set<std::set<ConstrainRef>> sortedSets;
  for (auto it = signalSets.begin(); it != signalSets.end(); it++) {
    if (!it->isLeader()) {
      continue;
    }

    std::set<ConstrainRef> sortedMembers;
    for (auto mit = signalSets.member_begin(it); mit != signalSets.member_end(); mit++) {
      sortedMembers.insert(*mit);
    }

    // We only want to print sets with a size > 1, because size == 1 means the
    // signal is not in a constraint.
    if (sortedMembers.size() > 1) {
      sortedSets.insert(sortedMembers);
    }
  }
  // Add the constants in separately.
  for (auto &[ref, constSet] : constantSets) {
    if (constSet.empty()) {
      continue;
    }
    std::set<ConstrainRef> sortedMembers(constSet.begin(), constSet.end());
    sortedMembers.insert(ref);
    sortedSets.insert(sortedMembers);
  }

  os << "ConstraintDependencyGraph { ";

  for (auto it = sortedSets.begin(); it != sortedSets.end();) {
    os << "\n    { ";
    for (auto mit = it->begin(); mit != it->end();) {
      os << *mit;
      mit++;
      if (mit != it->end()) {
        os << ", ";
      }
    }

    it++;
    if (it == sortedSets.end()) {
      os << " }\n";
    } else {
      os << " },";
    }
  }

  os << "}\n";
}

mlir::LogicalResult ConstraintDependencyGraph::computeConstraints(
    mlir::DataFlowSolver &solver, mlir::AnalysisManager &am
) {
  // Fetch the constrain function. This is a required feature for all LLZK structs.
  auto constrainFnOp = structDef.getConstrainFuncOp();
  debug::ensure(
      constrainFnOp,
      "malformed struct " + mlir::Twine(structDef.getName()) + " must define a constrain function"
  );

  /**
   * Now, given the analysis, construct the summary:
   * - Union all references based on solver results.
   * - Union all references based on nested summaries.
   */

  // - Union all constraints from the analysis
  // This requires iterating over all of the emit operations
  constrainFnOp.walk([this, &solver](EmitEqualityOp emitOp) {
    this->walkConstrainOp(solver, emitOp);
  });

  constrainFnOp.walk([this, &solver](EmitContainmentOp emitOp) {
    this->walkConstrainOp(solver, emitOp);
  });

  /**
   * Step two of the analysis is to traverse all of the constrain calls.
   * This is the nested analysis, basically.
   * Constrain functions don't return, so we don't need to compute "values" from
   * the call. We just need to see what constraints are generated here, and
   * add them to the transitive closures.
   */
  constrainFnOp.walk([this, &solver, &am](CallOp fnCall) mutable {
    auto res = resolveCallable<FuncOp>(tables, fnCall);
    debug::ensure(mlir::succeeded(res), "could not resolve constrain call");

    auto fn = res->get();
    if (fn.getName() != FUNC_NAME_CONSTRAIN) {
      return;
    }
    // Nested
    auto calledStruct = fn.getOperation()->getParentOfType<StructDefOp>();
    ConstrainRefRemappings translations;

    auto lattice = solver.lookupState<ConstrainRefLattice>(fnCall.getOperation());
    debug::ensure(lattice, "could not find lattice for call operation");

    // Map fn parameters to args in the call op
    for (unsigned i = 0; i < fn.getNumArguments(); i++) {
      auto prefix = ConstrainRef(fn.getArgument(i));

      for (auto &s : lattice->getOrDefault(fnCall.getOperand(i))) {
        translations.push_back({prefix, s});
      }
    }
    auto summary =
        am.getChildAnalysis<ConstraintDependencyGraphAnalysis>(calledStruct).getSummary();
    auto translatedSummary = summary.translate(translations);

    // Now, union sets based on the translation
    // We should be able to just merge what is in the translatedSummary to the current summary
    auto &tSets = translatedSummary.signalSets;
    for (auto lit = tSets.begin(); lit != tSets.end(); lit++) {
      if (!lit->isLeader()) {
        continue;
      }
      auto leader = lit->getData();
      for (auto mit = tSets.member_begin(lit); mit != tSets.member_end(); mit++) {
        signalSets.unionSets(leader, *mit);
      }
    }
    // And update the constant sets
    for (auto &[ref, constSet] : translatedSummary.constantSets) {
      constantSets[ref].insert(constSet.begin(), constSet.end());
    }
  });

  return mlir::success();
}

void ConstraintDependencyGraph::walkConstrainOp(
    mlir::DataFlowSolver &solver, mlir::Operation *emitOp
) {
  std::vector<ConstrainRef> signalUsages, constUsages;
  auto lattice = solver.lookupState<ConstrainRefLattice>(emitOp);
  debug::ensure(lattice, "failed to get lattice for emit operation");

  for (auto operand : emitOp->getOperands()) {
    auto refs = lattice->getOrDefault(operand);
    for (auto &ref : refs) {
      if (ref.isConstant()) {
        constUsages.push_back(ref);
      } else {
        signalUsages.push_back(ref);
      }
    }
  }

  // Compute a transitive closure over the signals.
  if (!signalUsages.empty()) {
    auto it = signalUsages.begin();
    auto leader = signalSets.getOrInsertLeaderValue(*it);
    for (it++; it != signalUsages.end(); it++) {
      signalSets.unionSets(leader, *it);
    }
  }
  // Also update constant references for each value.
  for (auto &sig : signalUsages) {
    constantSets[sig].insert(constUsages.begin(), constUsages.end());
  }
}

ConstraintDependencyGraph ConstraintDependencyGraph::translate(ConstrainRefRemappings translation) {
  ConstraintDependencyGraph res(mod, structDef);
  auto translate = [&translation](const ConstrainRef &elem
                   ) -> mlir::FailureOr<std::vector<ConstrainRef>> {
    std::vector<ConstrainRef> refs;
    for (auto &[prefix, replacement] : translation) {
      auto translated = elem.translate(prefix, replacement);
      if (mlir::succeeded(translated)) {
        refs.push_back(translated.value());
      }
    }
    if (refs.empty()) {
      return mlir::failure();
    }
    return refs;
  };
  for (auto leaderIt = signalSets.begin(); leaderIt != signalSets.end(); leaderIt++) {
    if (!leaderIt->isLeader()) {
      continue;
    }
    // translate everything in this set first
    std::vector<ConstrainRef> translatedSignals, translatedConsts;
    for (auto mit = signalSets.member_begin(leaderIt); mit != signalSets.member_end(); mit++) {
      auto member = translate(*mit);
      if (mlir::failed(member)) {
        continue;
      }
      for (auto &ref : *member) {
        if (ref.isConstant()) {
          translatedConsts.push_back(ref);
        } else {
          translatedSignals.push_back(ref);
        }
      }
      // Also add the constants from the original summary
      auto &origConstSet = constantSets[*mit];
      translatedConsts.insert(translatedConsts.end(), origConstSet.begin(), origConstSet.end());
    }

    if (translatedSignals.empty()) {
      continue;
    }

    // Now we can insert the translated signals
    auto it = translatedSignals.begin();
    auto leader = *it;
    res.signalSets.insert(leader);
    for (it++; it != translatedSignals.end(); it++) {
      res.signalSets.insert(*it);
      res.signalSets.unionSets(leader, *it);
    }

    // And update the constant references
    for (auto &ref : translatedSignals) {
      res.constantSets[ref].insert(translatedConsts.begin(), translatedConsts.end());
    }
  }
  return res;
}

ConstrainRefSet ConstraintDependencyGraph::getConstrainingValues(const ConstrainRef &ref) const {
  ConstrainRefSet res;
  auto currRef = mlir::FailureOr<ConstrainRef>(ref);
  while (mlir::succeeded(currRef)) {
    // Add signals
    auto it = signalSets.findLeader(currRef.value());
    for (; it != signalSets.member_end(); it++) {
      if (currRef.value() != *it) {
        res.insert(*it);
      }
    }
    // Add constants
    auto constIt = constantSets.find(*currRef);
    if (constIt != constantSets.end()) {
      res.insert(constIt->second.begin(), constIt->second.end());
    }
    // Go to parent
    currRef = currRef->getParentPrefix();
  }
  return res;
}

/* ConstraintDependencyGraphModuleAnalysis */

ConstraintDependencyGraphModuleAnalysis::ConstraintDependencyGraphModuleAnalysis(
    mlir::Operation *op, mlir::AnalysisManager &am
) {
  if (auto modOp = mlir::dyn_cast<mlir::ModuleOp>(op)) {
    mlir::DataFlowConfig config;
    mlir::DataFlowSolver solver(config);
    dataflow::markAllOpsAsLive(solver, modOp);

    // The analysis is run at the module level so that lattices are computed
    // for global functions as well.
    solver.load<ConstrainRefAnalysis>();
    auto res = solver.initializeAndRun(modOp);
    debug::ensure(res.succeeded(), "solver failed to run on module!");

    modOp.walk([this, &solver, &am](StructDefOp s) {
      auto &csa = am.getChildAnalysis<ConstraintDependencyGraphAnalysis>(s);
      if (mlir::failed(csa.constructSummary(solver, am))) {
        auto error_message = "ConstraintDependencyGraphAnalysis failed to compute summary for " +
                             mlir::Twine(s.getName());
        s->emitError(error_message);
        llvm::report_fatal_error(error_message);
      }
      summaries[s] = csa.summary;
    });
  } else {
    auto error_message =
        "ConstraintDependencyGraphModuleAnalysis expects provided op to be an mlir::ModuleOp!";
    op->emitError(error_message);
    llvm::report_fatal_error(error_message);
  }
}

} // namespace llzk
