//===-- ConstraintDependencyGraph.cpp ---------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/LLZK/Analysis/ConstrainRefLattice.h"
#include "llzk/Dialect/LLZK/Analysis/ConstraintDependencyGraph.h"
#include "llzk/Dialect/LLZK/Analysis/DenseAnalysis.h"
#include "llzk/Dialect/LLZK/Util/Hash.h"
#include "llzk/Dialect/LLZK/Util/SymbolHelper.h"

#include <mlir/Analysis/DataFlow/DeadCodeAnalysis.h>
#include <mlir/IR/Value.h>

#include <llvm/Support/Debug.h>

#include <numeric>
#include <unordered_set>

#define DEBUG_TYPE "llzk-cdg"

namespace llzk {

/* ConstrainRefAnalysis */

void ConstrainRefAnalysis::visitCallControlFlowTransfer(
    mlir::CallOpInterface call, dataflow::CallControlFlowAction action,
    const ConstrainRefLattice &before, ConstrainRefLattice *after
) {
  LLVM_DEBUG(
      llvm::dbgs() << "ConstrainRefAnalysis::visitCallControlFlowTransfer: " << call << '\n'
  );
  auto fnOpRes = resolveCallable<FuncOp>(tables, call);
  ensure(succeeded(fnOpRes), "could not resolve called function");

  LLVM_DEBUG({
    llvm::dbgs().indent(4) << "parent op is ";
    if (auto s = call->getParentOfType<StructDefOp>()) {
      llvm::dbgs() << s.getName();
    } else if (auto p = call->getParentOfType<FuncOp>()) {
      llvm::dbgs() << p.getName();
    } else {
      llvm::dbgs() << "<UNKNOWN PARENT TYPE>";
    }
    llvm::dbgs() << '\n';
  });

  /// `action == CallControlFlowAction::Enter` indicates that:
  ///   - `before` is the state before the call operation;
  ///   - `after` is the state at the beginning of the callee entry block;
  if (action == dataflow::CallControlFlowAction::EnterCallee) {
    // We skip updating the incoming lattice for function calls,
    // as ConstrainRefs are relative to the containing function/struct, so we don't need to pollute
    // the callee with the callers values.
    // This also avoids a non-convergence scenario, as calling a
    // function from other contexts can cause the lattice values to oscillate and constantly
    // change (thus looping infinitely).

    setToEntryState(after);
  }
  /// `action == CallControlFlowAction::Exit` indicates that:
  ///   - `before` is the state at the end of a callee exit block;
  ///   - `after` is the state after the call operation.
  else if (action == dataflow::CallControlFlowAction::ExitCallee) {
    // Get the argument values of the lattice by getting the state as it would
    // have been for the callsite.
    dataflow::AbstractDenseLattice *beforeCall = nullptr;
    if (auto *prev = call->getPrevNode()) {
      beforeCall = getLattice(prev);
    } else {
      beforeCall = getLattice(call->getBlock());
    }
    ensure(beforeCall, "could not get prior lattice");

    // Translate argument values based on the operands given at the call site.
    std::unordered_map<ConstrainRef, ConstrainRefLatticeValue, ConstrainRef::Hash> translation;
    auto funcOpRes = resolveCallable<FuncOp>(tables, call);
    ensure(mlir::succeeded(funcOpRes), "could not lookup called function");
    auto funcOp = funcOpRes->get();

    auto callOp = mlir::dyn_cast<CallOp>(call.getOperation());
    ensure(callOp, "call is not a llzk::CallOp");

    for (unsigned i = 0; i < funcOp.getNumArguments(); i++) {
      auto key = ConstrainRef(funcOp.getArgument(i));
      auto val = before.getOrDefault(callOp.getOperand(i));
      translation[key] = val;
    }

    // The lattice at the return is the lattice before the call + translated
    // return values.
    mlir::ChangeResult updated = after->join(*beforeCall);
    for (unsigned i = 0; i < callOp.getNumResults(); i++) {
      auto retVal = before.getReturnValue(i);
      auto [translatedVal, _] = retVal.translate(translation);
      updated |= after->setValue(callOp->getResult(i), translatedVal);
    }
    propagateIfChanged(after, updated);
  }
  /// `action == CallControlFlowAction::External` indicates that:
  ///   - `before` is the state before the call operation.
  ///   - `after` is the state after the call operation, since there is no callee
  ///      body to enter into.
  else if (action == mlir::dataflow::CallControlFlowAction::ExternalCallee) {
    // For external calls, we propagate what information we already have from
    // before the call to after the call, since the external call won't invalidate
    // any of that information. It also, conservatively, makes no assumptions about
    // external calls and their computation, so CDG edges will not be computed over
    // input arguments to external functions.
    join(after, before);
  }
}

void ConstrainRefAnalysis::visitOperation(
    mlir::Operation *op, const ConstrainRefLattice &before, ConstrainRefLattice *after
) {
  LLVM_DEBUG(llvm::dbgs() << "ConstrainRefAnalysis::visitOperation: " << *op << '\n');
  // Collect the references that are made by the operands to `op`.
  ConstrainRefLattice::ValueMap operandVals;
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
    ensure(mlir::succeeded(fieldOpRes), "could not find field read");

    auto res = fieldRead->getResult(0);
    const auto &ops = operandVals.at(fieldRead->getOpOperand(0).get());
    auto [fieldVals, _] = ops.referenceField(fieldOpRes.value());

    propagateIfChanged(after, after->setValue(res, fieldVals));
  } else if (mlir::isa<ReadArrayOp>(op)) {
    arraySubdivisionOpUpdate(op, operandVals, before, after);
  } else if (auto createArray = mlir::dyn_cast<CreateArrayOp>(op)) {
    // Create an array using the operand values, if they exist.
    // Currently, the new array must either be fully initialized or uninitialized.

    auto newArrayVal = ConstrainRefLatticeValue(createArray.getType().getShape());
    // If the array is initialized, iterate through all operands and initialize the array value.
    for (unsigned i = 0; i < createArray.getNumOperands(); i++) {
      auto currentOp = createArray.getOperand(i);
      auto &opVals = operandVals[currentOp];
      (void)newArrayVal.getElemFlatIdx(i).setValue(opVals);
    }

    assert(createArray->getNumResults() == 1);
    auto res = createArray->getResult(0);

    propagateIfChanged(after, after->setValue(res, newArrayVal));
  } else if (auto extractArray = mlir::dyn_cast<ExtractArrayOp>(op)) {
    arraySubdivisionOpUpdate(op, operandVals, before, after);
  } else {
    // Standard union of operands into the results value.
    // TODO: Could perform constant computation/propagation here for, e.g., arithmetic
    // over constants, but such analysis may be better suited for a dedicated pass.
    propagateIfChanged(after, fallbackOpUpdate(op, operandVals, before, after));
  }
}

// Perform a standard union of operands into the results value.
mlir::ChangeResult ConstrainRefAnalysis::fallbackOpUpdate(
    mlir::Operation *op, const ConstrainRefLattice::ValueMap &operandVals,
    const ConstrainRefLattice &before, ConstrainRefLattice *after
) {
  auto updated = mlir::ChangeResult::NoChange;
  for (auto res : op->getResults()) {
    auto cur = before.getOrDefault(res);

    for (auto &[_, opVal] : operandVals) {
      (void)cur.update(opVal);
    }
    updated |= after->setValue(res, cur);
  }
  return updated;
}

// Perform the update for either a readarr op or an extractarr op, which
// operate very similarly: index into the first operand using a variable number
// of provided indices.
void ConstrainRefAnalysis::arraySubdivisionOpUpdate(
    mlir::Operation *op, const ConstrainRefLattice::ValueMap &operandVals,
    const ConstrainRefLattice &before, ConstrainRefLattice *after
) {
  ensure(mlir::isa<ReadArrayOp>(op) || mlir::isa<ExtractArrayOp>(op), "wrong type of op provided!");

  // We index the first operand by all remaining indices.
  assert(op->getNumResults() == 1);
  auto res = op->getResult(0);

  auto array = op->getOperand(0);
  auto it = operandVals.find(array);
  ensure(it != operandVals.end(), "improperly constructed operandVals map");
  auto currVals = it->second;

  std::vector<ConstrainRefIndex> indices;

  for (size_t i = 1; i < op->getNumOperands(); i++) {
    auto currentOp = op->getOperand(i);
    auto idxIt = operandVals.find(currentOp);
    ensure(idxIt != operandVals.end(), "improperly constructed operandVals map");
    auto &idxVals = idxIt->second;

    if (idxVals.isSingleValue() && idxVals.getSingleValue().isConstantIndex()) {
      ConstrainRefIndex idx(idxVals.getSingleValue().getConstantIndexValue());
      indices.push_back(idx);
    } else {
      // Otherwise, assume any range is valid.
      auto arrayType = mlir::dyn_cast<ArrayType>(array.getType());
      auto lower = mlir::APInt::getZero(64);
      mlir::APInt upper(64, arrayType.getDimSize(i - 1));
      auto idxRange = ConstrainRefIndex(lower, upper);
      indices.push_back(idxRange);
    }
  }

  auto [newVals, _] = currVals.extract(indices);

  propagateIfChanged(after, after->setValue(res, newVals));
}

/* ConstraintDependencyGraph */

mlir::FailureOr<ConstraintDependencyGraph> ConstraintDependencyGraph::compute(
    mlir::ModuleOp m, StructDefOp s, mlir::DataFlowSolver &solver, mlir::AnalysisManager &am
) {
  ConstraintDependencyGraph cdg(m, s);
  if (cdg.computeConstraints(solver, am).failed()) {
    return mlir::failure();
  }
  return cdg;
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
  ensure(
      constrainFnOp,
      "malformed struct " + mlir::Twine(structDef.getName()) + " must define a constrain function"
  );

  /**
   * Now, given the analysis, construct the CDG:
   * - Union all references based on solver results.
   * - Union all references based on nested dependencies.
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
    ensure(mlir::succeeded(res), "could not resolve constrain call");

    auto fn = res->get();
    if (!fn.isStructConstrain()) {
      return;
    }
    // Nested
    auto calledStruct = fn.getOperation()->getParentOfType<StructDefOp>();
    ConstrainRefRemappings translations;

    auto lattice = solver.lookupState<ConstrainRefLattice>(fnCall.getOperation());
    ensure(lattice, "could not find lattice for call operation");

    // Map fn parameters to args in the call op
    for (unsigned i = 0; i < fn.getNumArguments(); i++) {
      auto prefix = ConstrainRef(fn.getArgument(i));
      auto val = lattice->getOrDefault(fnCall.getOperand(i));
      translations.push_back({prefix, val});
    }
    auto &childAnalysis =
        am.getChildAnalysis<ConstraintDependencyGraphStructAnalysis>(calledStruct);
    if (!childAnalysis.constructed()) {
      ensure(
          mlir::succeeded(childAnalysis.runAnalysis(solver, am)),
          "could not construct CDG for child struct"
      );
    }
    auto translatedCDG = childAnalysis.getResult().translate(translations);

    // Now, union sets based on the translation
    // We should be able to just merge what is in the translatedCDG to the current CDG
    auto &tSets = translatedCDG.signalSets;
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
    for (auto &[ref, constSet] : translatedCDG.constantSets) {
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
  ensure(lattice, "failed to get lattice for emit operation");

  for (auto operand : emitOp->getOperands()) {
    auto latticeVal = lattice->getOrDefault(operand);
    for (auto &ref : latticeVal.foldToScalar()) {
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

ConstraintDependencyGraph ConstraintDependencyGraph::translate(ConstrainRefRemappings translation
) const {
  ConstraintDependencyGraph res(mod, structDef);
  auto translate = [&translation](const ConstrainRef &elem
                   ) -> mlir::FailureOr<std::vector<ConstrainRef>> {
    std::vector<ConstrainRef> refs;
    for (auto &[prefix, vals] : translation) {
      if (!elem.isValidPrefix(prefix)) {
        continue;
      }

      if (vals.isArray()) {
        // Try to index into the array
        auto suffix = elem.getSuffix(prefix);
        ensure(
            mlir::succeeded(suffix), "failure is nonsensical, we already checked for valid prefix"
        );

        auto [resolvedVals, _] = vals.extract(suffix.value());
        auto folded = resolvedVals.foldToScalar();
        refs.insert(refs.end(), folded.begin(), folded.end());
      } else {
        for (auto &replacement : vals.getScalarValue()) {
          auto translated = elem.translate(prefix, replacement);
          if (mlir::succeeded(translated)) {
            refs.push_back(translated.value());
          }
        }
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
      // Also add the constants from the original CDG
      if (auto it = constantSets.find(*mit); it != constantSets.end()) {
        auto &origConstSet = it->second;
        translatedConsts.insert(translatedConsts.end(), origConstSet.begin(), origConstSet.end());
      }
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
    for (auto it = signalSets.findLeader(*currRef); it != signalSets.member_end(); it++) {
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

/* ConstraintDependencyGraphStructAnalysis */

mlir::LogicalResult ConstraintDependencyGraphStructAnalysis::runAnalysis(
    mlir::DataFlowSolver &solver, mlir::AnalysisManager &moduleAnalysisManager
) {
  auto result =
      ConstraintDependencyGraph::compute(getModule(), getStruct(), solver, moduleAnalysisManager);
  if (mlir::failed(result)) {
    return mlir::failure();
  }
  setResult(std::move(*result));
  return mlir::success();
}

} // namespace llzk
