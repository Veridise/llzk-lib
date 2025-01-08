#include "llzk/Dialect/LLZK/Analysis/ConstraintSummary.h"
#include "llzk/Dialect/LLZK/Analysis/DenseAnalysis.h"
#include "llzk/Dialect/LLZK/Util/Hash.h"
#include "llzk/Dialect/LLZK/Util/SymbolHelper.h"

#include <mlir/Analysis/DataFlow/DeadCodeAnalysis.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Index/IR/IndexOps.h>
#include <mlir/IR/Value.h>

#include <unordered_set>

namespace llzk {

/*
Private Utilities:

These classes are defined here and not in the header as they are not designed
for use outside of this specific ConstraintSummary analysis.
*/

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
    auto res = mlir::ChangeResult::NoChange;
    if (auto *r = dynamic_cast<const ConstrainRefLattice *>(&rhs)) {
      for (auto &[v, s] : r->refSetMap) {
        for (auto &ref : s) {
          auto [_, inserted] = refSetMap[v].insert(ref);
          res = inserted ? mlir::ChangeResult::Change : res;
        }
      }
    } else {
      llvm::report_fatal_error("invalid join lattice type");
    }
    return res;
  }

  /// Minimum lower bound
  virtual mlir::ChangeResult meet(const AbstractDenseLattice &rhs) override {
    llvm::report_fatal_error("what");
    mlir::ChangeResult::NoChange;
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
      for (auto &r : s) {
        auto [_, inserted] = refSetMap[v].insert(r);
        res = inserted ? mlir::ChangeResult::Change : res;
      }
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
    if (failed(fnOpRes)) {
      llvm::report_fatal_error("could not resolve called function");
    }

    join(after, before);

    auto fnOp = fnOpRes->get();
    if (fnOp.getName() == FUNC_NAME_CONSTRAIN) {
      // Do nothing special.
      return;
    }

    /// `action == CallControlFlowAction::Enter` indicates that:
    ///   - `before` is the state before the call operation;
    ///   - `after` is the state at the beginning of the callee entry block;
    if (action == dataflow::CallControlFlowAction::EnterCallee) {
      // Set up the transition function.

      // Here, we add all of the argument values to the lattice
      auto calledFnRes = resolveCallable<FuncOp>(tables, call);
      if (mlir::failed(calledFnRes)) {
        llvm::report_fatal_error("could not resolve function call");
      }
      auto calledFn = calledFnRes->get();

      auto updated = mlir::ChangeResult::NoChange;
      for (auto arg : calledFn->getRegion(0).getArguments()) {
        // llvm::errs() << "arg is " << arg << "\n";
        auto sourceRef = ConstrainRefLattice::getSourceRef(arg);
        if (mlir::failed(sourceRef)) {
          llvm::report_fatal_error("Failed to get source ref");
        }
        updated |= after->setValue(arg, sourceRef.value());
      }
      propagateIfChanged(after, updated);
    }

    /// `action == CallControlFlowAction::Exit` indicates that:
    ///   - `before` is the state at the end of a callee exit block;
    ///   - `after` is the state after the call operation.
    if (action == dataflow::CallControlFlowAction::ExitCallee) {
      // Set up the transition function.

      // Translate argument values
      std::unordered_map<ConstrainRef, ConstrainRefSet, ConstrainRef::Hash> translation;
      auto funcOpRes = resolveCallable<FuncOp>(tables, call);
      if (mlir::failed(funcOpRes)) {
        llvm::report_fatal_error("could not lookup called function");
      }
      auto funcOp = funcOpRes->get();

      auto callOp = mlir::dyn_cast<CallOp>(call.getOperation());
      if (!callOp) {
        llvm::report_fatal_error("call is not a call op!");
      }

      for (unsigned i = 0; i < funcOp.getNumArguments(); i++) {
        auto key = ConstrainRef(funcOp.getArgument(i));
        auto val = before.getOrDefault(callOp.getOperand(i));
        translation[key] = val;
      }

      mlir::ChangeResult updated = mlir::ChangeResult::NoChange;
      for (unsigned i = 0; i < callOp.getNumResults(); i++) {
        auto retRef = before.getReturnValue(i);
        ConstrainRefSet translated;
        for (auto &ref : retRef) {
          if (translation.find(ref) != translation.end()) {
            auto &retVal = translation.at(ref);
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
    if (action == mlir::dataflow::CallControlFlowAction::ExternalCallee) {
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
    // First, see if any of this operations operands are direct references,
    // or if we need to resolve function calls
    ConstrainRefSetMap operandVals;
    for (auto &operand : op->getOpOperands()) {
      operandVals[operand.get()] = before.getOrDefault(operand.get());
    }

    // Propagate existing state.
    join(after, before);

    // We will now join the the operand refs based on the type of operand.
    if (auto fieldRead = mlir::dyn_cast<FieldReadOp>(op)) {
      assert(operandVals.size() == 1);
      assert(fieldRead->getNumResults() == 1);

      auto fieldOpRes = fieldRead.getFieldDefOp(tables);
      if (mlir::failed(fieldOpRes)) {
        llvm::report_fatal_error("could not find field read\n");
      }

      auto res = fieldRead->getResult(0);
      const auto &ops = operandVals.at(fieldRead->getOpOperand(0).get());
      ConstrainRefSet fieldVals;
      for (auto &r : ops) {
        fieldVals.insert(r.createChild(ConstrainRefIndex(fieldOpRes->get())));
      }
      propagateIfChanged(after, after->setValue(res, fieldVals));
    } else if (auto arrayRead = mlir::dyn_cast<ReadArrayOp>(op)) {
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
      // Standard union of operands, unless all operands are constants,
      // in which case we can easily propagate more precise updates.
      auto updated = mlir::ChangeResult::NoChange;
      for (auto res : op->getResults()) {
        auto cur = before.getOrDefault(res);

        for (auto &[_, set] : operandVals) {
          cur.insert(set.begin(), set.end());
        }
        updated |= after->setValue(res, cur);
      }
      propagateIfChanged(after, updated);
    }
  }

protected:
  void setToEntryState(ConstrainRefLattice *lattice) override {
    // the entry state is empty, so do nothing.
  }

private:
  mlir::SymbolTableCollection tables;
};

/*
ConstraintSummaryAnalysis

Needs to be declared before implementing the ConstraintSummary functions, as
they reference the ConstraintSummaryAnalysis.
*/

/// @brief An analysis wrapper around the ConstraintSummary for a given struct.
/// This analysis is a StructDefOp-level analysis that should not be directly
/// interacted with---rather, it is a utility used by the ConstraintSummaryModuleAnalysis
/// that helps use MLIR's AnalysisManager to cache summaries for sub-components.
class ConstraintSummaryAnalysis {
public:
  ConstraintSummaryAnalysis(mlir::Operation *op) {
    structDefOp = mlir::dyn_cast<StructDefOp>(op);
    if (!structDefOp) {
      auto error_message = "ConstraintSummaryAnalysis expects provided op to be a StructDefOp!";
      op->emitError(error_message);
      llvm::report_fatal_error(error_message);
    }
    auto maybeModOp = getRootModule(op);
    if (mlir::failed(maybeModOp)) {
      auto error_message = "ConstraintSummaryAnalysis could not find root module from StructDefOp!";
      op->emitError(error_message);
      llvm::report_fatal_error(error_message);
    }
    modOp = *maybeModOp;
  }

  /// @brief Construct a summary, using the module's analysis manager to query
  /// ConstraintSummary objects for nested components.
  mlir::LogicalResult
  constructSummary(mlir::DataFlowSolver &solver, mlir::AnalysisManager &moduleAnalysisManager) {
    auto summaryRes = ConstraintSummary::compute(modOp, structDefOp, solver, moduleAnalysisManager);
    if (mlir::failed(summaryRes)) {
      return mlir::failure();
    }
    summary = std::make_shared<ConstraintSummary>(*summaryRes);
    return mlir::success();
  }

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

/* ConstraintSummary */

mlir::FailureOr<ConstraintSummary> ConstraintSummary::compute(
    mlir::ModuleOp m, StructDefOp s, mlir::DataFlowSolver &solver, mlir::AnalysisManager &am
) {
  ConstraintSummary summary(m, s);
  if (summary.computeConstraints(solver, am).failed()) {
    return mlir::failure();
  }
  return summary;
}

void ConstraintSummary::dump() const { print(llvm::errs()); }

/// NOTE: Only prints scalar elements. The constraintSets also contains
/// references to composite elements to help with construction and lookup, but
/// for printing clarity, we omit these elements.
/// For each reference within the struct, print the set that constrains that reference.
void ConstraintSummary::print(llvm::raw_ostream &os) const {
  // the EquivalenceClasses::iterator is sorted, but the EquivalenceClasses::member_iterator is
  // not guaranteed to be sorted. So, we will sort members before printing them.
  std::set<std::set<ConstrainRef>> sortedSets;
  for (auto it = constraintSets.begin(); it != constraintSets.end(); it++) {
    if (!it->isLeader()) {
      continue;
    }

    std::set<ConstrainRef> sortedMembers;
    for (auto mit = constraintSets.member_begin(it); mit != constraintSets.member_end(); mit++) {
      sortedMembers.insert(*mit);
    }

    sortedSets.insert(sortedMembers);
  }

  os << "ConstraintSummary { ";

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
      os << " },\n";
    }
  }

  os << "}\n";
}

mlir::LogicalResult
ConstraintSummary::computeConstraints(mlir::DataFlowSolver &solver, mlir::AnalysisManager &am) {
  // Fetch the constrain function. This is a required feature for all LLZK structs.
  auto constrainFnOp = structDef.getConstrainFuncOp();
  if (constrainFnOp == nullptr) {
    llvm::report_fatal_error(
        "malformed struct " + mlir::Twine(structDef.getName()) + " must define a constrain function"
    );
  }

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
    if (mlir::failed(res)) {
      fnCall.emitError() << "Could not resolve callable!\n";
      return;
    }
    auto fn = res->get();
    if (fn.getName() != FUNC_NAME_CONSTRAIN) {
      return;
    }
    // Nested
    StructDefOp calledStruct(fn.getOperation()->getParentOp());
    ConstrainRefRemappings translations;

    auto lattice = solver.lookupState<ConstrainRefLattice>(fnCall.getOperation());
    if (!lattice) {
      llvm::report_fatal_error("could not find lattice for call operation");
    }
    // Map fn parameters to args in the call op
    for (unsigned i = 0; i < fn.getNumArguments(); i++) {
      auto prefix = ConstrainRef(fn.getArgument(i));

      for (auto &s : lattice->getOrDefault(fnCall.getOperand(i))) {
        translations.push_back({prefix, s});
      }
    }
    auto summary = am.getChildAnalysis<ConstraintSummaryAnalysis>(calledStruct).getSummary();
    auto translatedSummary = summary.translate(translations);

    // Now, union sets based on the translation
    // We should be able to just merge what is in the translatedSummary to the current summary
    auto &tSets = translatedSummary.constraintSets;
    for (auto lit = tSets.begin(); lit != tSets.end(); lit++) {
      if (!lit->isLeader()) {
        continue;
      }
      auto leader = lit->getData();
      for (auto mit = tSets.member_begin(lit); mit != tSets.member_end(); mit++) {
        constraintSets.unionSets(leader, *mit);
      }
    }
  });

  return mlir::success();
}

void ConstraintSummary::walkConstrainOp(mlir::DataFlowSolver &solver, mlir::Operation *emitOp) {
  std::vector<ConstrainRef> usages;
  for (auto operand : emitOp->getOperands()) {
    auto lattice = solver.lookupState<ConstrainRefLattice>(emitOp);
    if (!lattice) {
      llvm::report_fatal_error("failed to get lattice for emit operation");
    }
    auto refs = lattice->getOrDefault(operand);
    usages.insert(usages.end(), refs.begin(), refs.end());
  }

  auto it = usages.begin();
  auto leader = constraintSets.getOrInsertLeaderValue(*it);
  for (it++; it != usages.end(); it++) {
    constraintSets.unionSets(leader, *it);
  }
}

ConstraintSummary ConstraintSummary::translate(ConstrainRefRemappings translation) {
  ConstraintSummary res(mod, structDef);
  auto translate = [&translation](const ConstrainRef &elem
                   ) -> mlir::FailureOr<std::vector<ConstrainRef>> {
    std::vector<ConstrainRef> res;
    for (auto &[prefix, replacement] : translation) {
      auto translated = elem.translate(prefix, replacement);
      if (mlir::succeeded(translated)) {
        res.push_back(translated.value());
      }
    }
    if (res.empty()) {
      return mlir::failure();
    }
    return res;
  };
  for (auto leaderIt = constraintSets.begin(); leaderIt != constraintSets.end(); leaderIt++) {
    if (!leaderIt->isLeader()) {
      continue;
    }
    // translate everything in this set first
    std::vector<ConstrainRef> translated;
    for (auto mit = constraintSets.member_begin(leaderIt); mit != constraintSets.member_end();
         mit++) {
      auto member = translate(*mit);
      if (mlir::succeeded(member)) {
        translated.insert(translated.end(), member->begin(), member->end());
      }
    }

    if (translated.empty()) {
      continue;
    }

    // Now we can insert
    auto it = translated.begin();
    auto leader = *it;
    res.constraintSets.insert(leader);
    for (it++; it != translated.end(); it++) {
      res.constraintSets.insert(*it);
      res.constraintSets.unionSets(leader, *it);
    }
  }
  return res;
}

ConstrainRefSet ConstraintSummary::getConstrainingValues(const ConstrainRef &ref) const {
  ConstrainRefSet res;
  auto currRef = mlir::FailureOr<ConstrainRef>(ref);
  while (mlir::succeeded(currRef)) {
    auto it = constraintSets.findLeader(currRef.value());
    for (; it != constraintSets.member_end(); it++) {
      if (currRef.value() != *it) {
        res.insert(*it);
      }
    }
    currRef = currRef->getParentPrefix();
  }
  return res;
}

/* ConstraintSummaryModuleAnalysis */

ConstraintSummaryModuleAnalysis::ConstraintSummaryModuleAnalysis(
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
    if (res.failed()) {
      llvm::report_fatal_error("solver failed to run on module!");
    }

    modOp.walk([this, &solver, &am](StructDefOp s) {
      auto &csa = am.getChildAnalysis<ConstraintSummaryAnalysis>(s);
      if (mlir::failed(csa.constructSummary(solver, am))) {
        auto error_message =
            "ConstraintSummaryAnalysis failed to compute summary for " + mlir::Twine(s.getName());
        s->emitError(error_message);
        llvm::report_fatal_error(error_message);
      }
      summaries[s] = csa.summary;
    });
  } else {
    auto error_message =
        "ConstraintSummaryModuleAnalysis expects provided op to be an mlir::ModuleOp!";
    op->emitError(error_message);
    llvm::report_fatal_error(error_message);
  }
}

} // namespace llzk