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

/// @brief Mark all operations from the top and included in the top operation
/// as live so the solver will perform dataflow analyses.
/// @param solver The solver.
/// @param top The top-level operation.
void makeLive(mlir::DataFlowSolver &solver, mlir::Operation *top) {
  for (mlir::Region &region : top->getRegions()) {
    for (mlir::Block &block : region) {
      (void)solver.getOrCreateState<mlir::dataflow::Executable>(&block)->setToLive();
      for (mlir::Operation &oper : block) {
        makeLive(solver, &oper);
      }
    }
  }
}

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

  // ConstrainRefLattice substituteArguments()

  // /// @brief Create child references based on a ref index
  // /// @param f
  // /// @return
  // ConstrainRefLatticeValue createChildren(ConstrainRefIndex r) const {
  //   ConstrainRefLatticeValue children;
  //   for (const auto &ref : signals) {
  //     children.signals.insert(ref.createChild(r));
  //   }
  //   return children;
  // }

  // ConstrainRefLatticeValue createChildren(FieldDefOp f) const {
  //   return createChildren(ConstrainRefIndex(f));
  // }

  // ConstrainRefLatticeValue createChildren(mlir::APInt i) const {
  //   return createChildren(ConstrainRefIndex(i));
  // }

  // ConstrainRefLatticeValue createChildren(mlir::APInt lower, mlir::APInt upper) const {
  //   return createChildren(ConstrainRefIndex(lower, upper));
  // }

  // /* Getter utility methods */

  // /// Get the single value of this lattice value if there is a single value,
  // /// returning a failure otherwise.
  // mlir::FailureOr<ConstrainRef> getSingleValue() const {
  //   if (signals.size() == 1) {
  //     return *signals.begin();
  //   }
  //   return mlir::failure();
  // }

  // /// @brief Return true if this contains a single, concrete value.
  // /// @return
  // bool isConcreteValue() const {
  //   auto singleValRes = getSingleValue();
  //   if (mlir::succeeded(singleValRes) && singleValRes->isConstant()) {
  //     return true;
  //   }
  //   return false;
  // }

  ConstrainRefSet &operator[](mlir::Value v) { return refSetMap[v]; }
  const ConstrainRefSet &operator[](mlir::Value v) const { return refSetMap.at(v); }

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

  static std::vector<ConstrainRef> getRefs(mlir::DataFlowSolver &solver, mlir::Value v) {
    std::vector<ConstrainRef> usages;
    // It may also be the case that the operand is just the argument value, which
    // won't have a lattice value.
    auto sourceValRes = ConstrainRefLattice::getSourceRef(v);
    if (mlir::succeeded(sourceValRes)) {
      usages.emplace_back(sourceValRes.value());
    } else {
      auto analysisRes = solver.lookupState<ConstrainRefLattice>(v);
      if (!analysisRes) {
        llvm::errs() << v << " is untraversed\n";
        llvm::report_fatal_error("untraversed value");
      }

      auto &signals = (*analysisRes)[v];
      usages.insert(usages.end(), signals.begin(), signals.end());
    }
    return usages;
  }

  void visitCallControlFlowTransfer(
      mlir::CallOpInterface call, dataflow::CallControlFlowAction action,
      const ConstrainRefLattice &before, ConstrainRefLattice *after
  ) override {
    // llvm::errs() << "CALL CALL CALL " << call << '\n';

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
      // llvm::report_fatal_error("enter todo");
      // setToEntryState(after);

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

      // llvm::errs() << "EnterCallee:\n";
      // llvm::errs() << before << "\n";
      // llvm::errs() << *after << "\n";
    }

    /// `action == CallControlFlowAction::Exit` indicates that:
    ///   - `before` is the state at the end of a callee exit block;
    ///   - `after` is the state after the call operation.
    if (action == dataflow::CallControlFlowAction::ExitCallee) {
      // Set up the transition function.
      // llvm::report_fatal_error("exit todo");

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
        // llvm::errs() << "key: " << key << "\n";
        // llvm::errs() << "val: " << callOp.getOperand(i) << "\n";
        auto val = before.getOrDefault(callOp.getOperand(i));

        // llvm::errs() << "  val size " << val.size() << "\n";
        // for (auto &v : val) {
        //   llvm::errs() << "    " << v << "\n";
        // }
        translation[key] = val;
      }

      mlir::ChangeResult updated = mlir::ChangeResult::NoChange;
      for (unsigned i = 0; i < callOp.getNumResults(); i++) {
        // llvm::errs() << "translating " << callOp << " return " << i << "\n";
        auto retRef = before.getReturnValue(i);
        ConstrainRefSet translated;
        // llvm::errs() << "     ret value " << callOp->getResult(i) << ":\n";
        for (auto &ref : retRef) {
          if (translation.find(ref) != translation.end()) {
            auto &retVal = translation.at(ref);
            translated.insert(retVal.begin(), retVal.end());
            // llvm::errs() << "        translated " << ref << " to:\n";
            // for (auto &q : retVal) {
            //   llvm::errs() << "            " << q << "\n";
            // }
          }
        }

        updated |= after->setValue(callOp->getResult(i), translated);
      }
      propagateIfChanged(after, updated);

      // llvm::errs() << "ExitCallee:\n";
      // llvm::errs() << before << "\n";
      // llvm::errs() << *after << "\n";
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

    // this will be set to entry state, so don't do anything
    if (auto fn = mlir::dyn_cast<FuncOp>(op)) {
      // llvm::errs() << "VISIT OPERATION: FUNC " << fn.getName() << "\n";
      // llvm::errs() << "    has " << op->getRegion(0).getNumArguments() << " arguments!\n";
      // // Seed the lattice with argument values.
      // auto updated = mlir::ChangeResult::NoChange;
      // for (auto arg : op->getRegion(0).getArguments()) {
      //   llvm::errs() << "arg is " << arg << "\n";
      //   auto sourceRef = getSourceRef(arg);
      //   if (mlir::failed(sourceRef)) {
      //     llvm::report_fatal_error("Failed to get source ref");
      //   }
      //   updated |= after->setValue(arg, sourceRef.value());
      // }
      // propagateIfChanged(after, updated);

      // Debugging calls
      // fn.walk([this](CallOp callOp) {
      //   llvm::errs() << "debugging call " << callOp << "\n";
      //   mlir::CallOpInterface call =
      //   mlir::dyn_cast<mlir::CallOpInterface>(callOp.getOperation());
      //   {
      //     auto callable =
      //       dyn_cast_if_present<mlir::CallableOpInterface>(call.resolveCallable());
      //     if (!callable) {
      //       llvm::errs() << "BEFORE callable is nullptr!\n";
      //     } else {
      //       llvm::errs() << "BEFORE callable is not nullptr!\n";
      //     }
      //   }

      //   auto res = llzk::resolveCallable<FuncOp>(tables, call);
      //   if (mlir::succeeded(res)) {
      //     auto fn = res.value().get();
      //     auto symRes = getPathFromRoot(fn);
      //     if (mlir::failed(symRes)) {
      //       llvm::report_fatal_error("cannot get path from root");
      //     }
      //     call.setCalleeFromCallable(symRes.value());
      //     callOp.setCalleeAttr(symRes.value());
      //   } else {
      //     llvm::errs() << "no symbol!\n";
      //   }

      //   {
      //     auto callable =
      //       dyn_cast_if_present<mlir::CallableOpInterface>(call.resolveCallable());
      //     if (!callable) {
      //       llvm::errs() << "AFTER callable is nullptr!\n";
      //     } else {
      //       llvm::errs() << "AFTER callable is not nullptr!\n";
      //     }
      //   }

      //   // llvm::errs() << (callable && callable.getCallableRegion()) << '\n';
      //   // llvm::errs() << callable << ", " << callable.getCallableRegion() << "\n";
      // });
    }

    // this->visitExternalCall
    // auto resolveCalls = [this, &solver, &am](CallOp fnCall) mutable {
    //   auto res = resolveCallable<FuncOp>(tables, fnCall);
    //   if (mlir::failed(res)) {
    //     fnCall.emitError() << "Could not resolve callable!\n";
    //     return;
    //   }
    //   auto fn = res->get();
    //   if (fn.getName() != FUNC_NAME_CONSTRAIN) {
    //     return;
    //   }
    //   // Nested
    //   StructDefOp calledStruct(fn.getOperation()->getParentOp());
    //   ConstrainRefRemappings translations;
    //   // Map fn parameters to args in the call op
    //   for (unsigned i = 0; i < fn.getNumArguments(); i++) {
    //     auto prefix = ConstrainRef(fn.getArgument(i));
    //     auto replacements = solver.lookupState<ConstrainRefLattice>(fnCall.getOperand(i));

    //     if (!replacements) {
    //       llvm::report_fatal_error("failed to look up replacement translation symbols");
    //     }

    //     for (auto &s : replacements->getValue().signals) {
    //       translations.push_back({prefix, s});
    //     }
    //   }
    //   auto summary = am.getChildAnalysis<ConstraintSummaryAnalysis>(calledStruct).getSummary();
    //   auto translatedSummary = summary.translate(translations);

    //   // Now, union sets based on the translation
    //   // We should be able to just merge what is in the translatedSummary to the current summary
    //   auto &tSets = translatedSummary.constraintSets;
    //   for (auto lit = tSets.begin(); lit != tSets.end(); lit++) {
    //     if (!lit->isLeader()) {
    //       continue;
    //     }
    //     auto leader = lit->getData();
    //     for (auto mit = tSets.member_begin(lit); mit != tSets.member_end(); mit++) {
    //       constraintSets.unionSets(leader, *mit);
    //     }
    //   }
    // };

    // First, see if any of this operations operands are direct references,
    // or if we need to resolve function calls
    ConstrainRefSetMap operandVals;
    for (auto &operand : op->getOpOperands()) {
      operandVals[operand.get()] = before.getOrDefault(operand.get());
      ;
    }

    // llvm::errs() << "running " << *op << "\n";
    // if (auto retOp = mlir::dyn_cast<ReturnOp>(op)) {
    //   llvm::errs() << "    return op has " << op->getResults().size() << " results\n";
    // }

    // for (auto &[v, set] : operandVals) {
    //   llvm::errs() << "    " << v << " => { ";
    //   for (auto &ref : set) {
    //     llvm::errs() << ref << ", ";
    //   }
    //   llvm::errs() << " }\n";
    // }

    /* Not doing this, we don't want the default values. */
    // auto updated = after->join(before);

    // updated |= after->setValues(operandVals);

    // simple union.
    // for (auto res : op->getResults()) {
    //   auto cur = before.getOrDefault(res);

    //   for (auto &[_, set] : operandVals) {
    //     cur.insert(set.begin(), set.end());
    //   }
    //   updated |= after->setValue(res, cur);
    // }

    // propagateIfChanged(after, updated);
    /* end */

    // Propagate existing state.
    join(after, before);

    // llvm::errs() << "BEFORE " << before;
    // llvm::errs() << "AFTER " << *after;

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

class GlobalFuncAnalysis {
public:
  GlobalFuncAnalysis(mlir::Operation *op) {
    funcOp = mlir::dyn_cast<FuncOp>(op);
    if (!funcOp) {
      auto error_message = "GlobalFuncAnalysis expects provided op to be a FuncOp!";
      op->emitError(error_message);
      llvm::report_fatal_error(error_message);
    }
    auto maybeModOp = getRootModule(op);
    if (mlir::failed(maybeModOp)) {
      auto error_message = "GlobalFuncAnalysis could not find root module from FuncOp!";
      op->emitError(error_message);
      llvm::report_fatal_error(error_message);
    }
    modOp = *maybeModOp;
  }

  mlir::LogicalResult
  computeAnalysis(mlir::DataFlowSolver &solver, mlir::AnalysisManager &moduleAnalysisManager) {
    if (solver.initializeAndRun(funcOp).failed()) {
      return mlir::failure();
    }

    retRefs.resize(funcOp.getNumResults());

    funcOp.walk([this, &solver](ReturnOp r) {
      for (unsigned i = 0; i < r.getNumOperands(); i++) {
        auto operand = r.getOperand(i);
        if (auto operandVals = solver.lookupState<ConstrainRefLattice>(operand)) {
          auto &refs = (*operandVals)[operand];

          retRefs[i].insert(refs.begin(), refs.end());
        }
      }
    });

    return mlir::success();
  }

  const std::vector<ConstrainRefSet> &getReturnVals() { return retRefs; }

  const ConstrainRefSet &getReturnVal(unsigned i) { return retRefs[i]; }

private:
  mlir::ModuleOp modOp;
  FuncOp funcOp;

  std::vector<ConstrainRefSet> retRefs;
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

mlir::FailureOr<std::vector<ConstrainRef>>
ConstraintSummary::getConstrainRefs(mlir::DataFlowSolver &solver, mlir::Value val) {
  llvm::report_fatal_error("get out of here");
  std::vector<ConstrainRef> res;
  // Due to the way constrain is defined, all signals are read from inputs.
  if (auto blockArg = mlir::dyn_cast_or_null<mlir::BlockArgument>(val)) {
    // to use this constructor, the block arg must be a felt
    res.push_back(ConstrainRef(blockArg));
  } else if (auto fieldRead = mlir::dyn_cast_or_null<FieldReadOp>(val.getDefiningOp())) {
    auto fieldOpRes = fieldRead.getFieldDefOp(tables);
    if (mlir::failed(fieldOpRes)) {
      fieldRead.emitError() << "could not find field read\n";
      return mlir::failure();
    }

    auto parentRes = getConstrainRefs(solver, fieldRead.getComponent());
    if (mlir::succeeded(parentRes)) {
      for (auto ref : parentRes.value()) {
        res.push_back(ref.createChild(ConstrainRefIndex(fieldOpRes->get())));
      }
    }
  } else if (auto arrayRead = mlir::dyn_cast_or_null<ReadArrayOp>(val.getDefiningOp())) {
    llvm::errs() << "array read " << arrayRead << "\n";
    auto array = arrayRead.getOperand(0);
    auto arrayRefs = getConstrainRefs(solver, array);
    if (mlir::succeeded(arrayRefs)) {
      for (auto idx : arrayRead.getIndices()) {
        llvm::errs() << "   idx " << idx << "\n";
      }
    }
    llvm::report_fatal_error("todo!");
  } else if (val.getDefiningOp() != nullptr && mlir::isa<FeltConstantOp>(val.getDefiningOp())) {
    auto constFelt = mlir::dyn_cast<FeltConstantOp>(val.getDefiningOp());
    res.emplace_back(constFelt);
  } else if (val.getDefiningOp() != nullptr) {
    // Fallback for every other type of operation
    // This also works for global func call ops, since we use an interprocedural dataflow solver
    for (auto operand : val.getDefiningOp()->getOperands()) {
      auto uses = getConstrainRefs(solver, operand);
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

std::vector<ConstrainRef> ConstraintSummary::getAllConstrainRefs(
    ArrayType arrayTy, mlir::BlockArgument blockArg, std::vector<ConstrainRefIndex> fields
) const {
  std::vector<ConstrainRef> res;
  // Add root item
  res.emplace_back(blockArg, fields);

  // Recurse into arrays by iterating over their elements
  int64_t maxSz = arrayTy.getDimSize(0);
  for (int64_t i = 0; i < maxSz; i++) {
    auto elemTy = arrayTy.getElementType();

    std::vector<ConstrainRefIndex> subFields = fields;
    subFields.emplace_back(i);

    if (auto arrayElemTy = mlir::dyn_cast<ArrayType>(elemTy)) {
      // recurse
      auto subRes = getAllConstrainRefs(arrayElemTy, blockArg, subFields);
      res.insert(res.end(), subRes.begin(), subRes.end());
    } else if (auto structTy = mlir::dyn_cast<StructType>(elemTy)) {
      // recurse into struct def
      auto subRes = getAllConstrainRefs(getStructDef(structTy), blockArg, subFields);
      res.insert(res.end(), subRes.begin(), subRes.end());
    } else {
      // scalar type
      res.emplace_back(blockArg, subFields);
    }
  }

  return res;
}

std::vector<ConstrainRef> ConstraintSummary::getAllConstrainRefs(
    StructDefOp s, mlir::BlockArgument blockArg, std::vector<ConstrainRefIndex> fields
) const {
  std::vector<ConstrainRef> res;
  // Add root item
  res.emplace_back(blockArg, fields);
  // Recurse into struct types by iterating over all their field definitions
  for (auto f : s.getOps<FieldDefOp>()) {
    std::vector<ConstrainRefIndex> subFields = fields;
    subFields.emplace_back(f);
    // Make a reference to the current field, regardless of if it is a composite
    // type or not.
    res.emplace_back(blockArg, subFields);
    if (auto structTy = mlir::dyn_cast<llzk::StructType>(f.getType())) {
      // Create refs for each field
      auto subRes = getAllConstrainRefs(getStructDef(structTy), blockArg, subFields);
      res.insert(res.end(), subRes.begin(), subRes.end());
    } else if (auto arrayTy = mlir::dyn_cast<llzk::ArrayType>(f.getType())) {
      // Create refs for each array element
      auto subRes = getAllConstrainRefs(arrayTy, blockArg, subFields);
      res.insert(res.end(), subRes.begin(), subRes.end());
    }
  }
  return res;
}

std::vector<ConstrainRef> ConstraintSummary::getAllConstrainRefs(mlir::BlockArgument arg) const {
  auto ty = arg.getType();
  std::vector<ConstrainRef> res;
  if (auto structTy = mlir::dyn_cast<llzk::StructType>(ty)) {
    // recurse over fields
    res = getAllConstrainRefs(getStructDef(structTy), arg);
  } else if (auto arrayType = mlir::dyn_cast<llzk::ArrayType>(ty)) {
    res = getAllConstrainRefs(arrayType, arg);
  } else if (mlir::isa<llzk::FeltType>(ty) || mlir::isa<mlir::IndexType>(ty)) {
    // Scalar type
    res.emplace_back(arg);
  } else {
    std::string msg = "unsupported type: ";
    llvm::raw_string_ostream ss(msg);
    ss << ty;
    llvm::report_fatal_error(ss.str().c_str());
  }
  return res;
}

std::vector<ConstrainRef> ConstraintSummary::getAllConstrainRefs() const {
  std::vector<ConstrainRef> res;
  auto constrainFnOp = structDef.getConstrainFuncOp();
  if (constrainFnOp == nullptr) {
    llvm::report_fatal_error(
        "malformed struct " + mlir::Twine(structDef.getName()) + " must define a constrain function"
    );
  }

  for (auto a : constrainFnOp.getArguments()) {
    auto argRes = getAllConstrainRefs(a);
    res.insert(res.end(), argRes.begin(), argRes.end());
  }
  return res;
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

  /// NOTE: do this at the module level
  // Run the analysis. Assumes the liveness analysis has already been performed.
  // solver.load<ConstrainRefAnalysis>();
  // auto res = solver.initializeAndRun(constrainFnOp);
  // if (res.failed()) {
  //   return mlir::failure();
  // }

  /**
   * Now, given the analysis, construct the summary:
   * - Union all references based on solver results.
   * - Union all referenced based on function calls.
   * - Union all references based on nested summaries.
   */

  /// NOTE: no?
  /// or will this work ONLY if we have some kind of op that uses the return val?
  /// Yes, s
  // insert solver state based on global functions
  // constrainFnOp.walk([this, &solver, &am](CallOp fnCall) mutable {
  //   auto res = resolveCallable<FuncOp>(tables, fnCall);
  //   if (mlir::failed(res)) {
  //     fnCall.emitError() << "Could not resolve callable!\n";
  //     return;
  //   }
  //   auto fn = res->get();
  //   if (fn.getName() == FUNC_NAME_CONSTRAIN) {
  //     return;
  //   }

  //   // recurse over function
  //   ConstrainRefRemappings translations;
  //   // Map fn parameters to args in the call op
  //   for (unsigned i = 0; i < fn.getNumArguments(); i++) {
  //     auto prefix = ConstrainRef(fn.getArgument(i));
  //     auto replacements = solver.lookupState<ConstrainRefLattice>(fnCall.getOperand(i));

  //     if (!replacements) {
  //       llvm::report_fatal_error("failed to look up replacement translation symbols");
  //     }

  //     for (auto &s : replacements->getValue().signals) {
  //       translations.push_back({prefix, s});
  //     }
  //   }
  //   auto retAnalysis = am.getChildAnalysis<GlobalFuncAnalysis>(fn);
  //   if (retAnalysis.computeAnalysis(solver, am).failed()) {
  //     llvm::report_fatal_error("failed to run global function analysis");
  //   }

  //   llvm::errs() << "Confirm we have no values\n";
  //   for (auto ret : fnCall.getResults()) {
  //     auto val = solver.lookupState<ConstrainRefLattice>(ret);
  //     llvm::errs() << "ret of " << ret << " is " << *val << "\n";
  //   }

  //   // Now, insert state into the solver values.
  //   for (unsigned i = 0; i < fnCall.getNumResults(); i++) {
  //     auto state = solver.lookupState<ConstrainRefLattice>(fnCall.getResult(i));
  //     if (!state) {
  //       llvm::report_fatal_error("state not created");
  //     }
  //     auto &untranslated = retAnalysis.getReturnVal(i);
  //     // auto &retState = solver.getProgramPoint
  //     for (const auto &ref : untranslated) {
  //       // state->getValue().
  //     }
  //   }

  //   llvm::report_fatal_error("todo! thingy");

  // });

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
        // llvm::errs() << "union:\n";
        // llvm::errs() << "  " << leader << "\n";
        // llvm::errs() << "  " << *mit << "\n";
        // llvm::errs() << "  --\n";
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
    // llvm::errs() << "unioning:\n  " << leader << "\n  " << *it << "\n  done\n";
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

std::set<ConstrainRef> ConstraintSummary::getConstrainingValues(const ConstrainRef &ref) const {
  std::set<ConstrainRef> res;
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
    makeLive(solver, modOp);

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