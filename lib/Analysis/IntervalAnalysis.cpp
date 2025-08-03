//===-- IntervalAnalysis.cpp - Interval analysis implementation -*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Analysis/IntervalAnalysis.h"
#include "llzk/Analysis/Matchers.h"
#include "llzk/Util/Debug.h"
#include "llzk/Util/StreamHelper.h"

#include <llvm/ADT/TypeSwitch.h>

using namespace mlir;

namespace llzk {

using namespace array;
using namespace boolean;
using namespace cast;
using namespace component;
using namespace constrain;
using namespace felt;
using namespace function;

/* ExpressionValue */

bool ExpressionValue::operator==(const ExpressionValue &rhs) const {
  if (expr == nullptr && rhs.expr == nullptr) {
    return i == rhs.i;
  }
  if (expr == nullptr || rhs.expr == nullptr) {
    return false;
  }
  return i == rhs.i && *expr == *rhs.expr;
}

ExpressionValue
intersection(llvm::SMTSolverRef solver, const ExpressionValue &lhs, const ExpressionValue &rhs) {
  Interval res = lhs.i.intersect(rhs.i);
  auto exprEq = solver->mkEqual(lhs.expr, rhs.expr);
  return ExpressionValue(exprEq, res);
}

ExpressionValue
add(llvm::SMTSolverRef solver, const ExpressionValue &lhs, const ExpressionValue &rhs) {
  ExpressionValue res;
  res.i = lhs.i + rhs.i;
  res.expr = solver->mkBVAdd(lhs.expr, rhs.expr);
  return res;
}

ExpressionValue
sub(llvm::SMTSolverRef solver, const ExpressionValue &lhs, const ExpressionValue &rhs) {
  ExpressionValue res;
  res.i = lhs.i - rhs.i;
  res.expr = solver->mkBVSub(lhs.expr, rhs.expr);
  return res;
}

ExpressionValue
mul(llvm::SMTSolverRef solver, const ExpressionValue &lhs, const ExpressionValue &rhs) {
  ExpressionValue res;
  res.i = lhs.i * rhs.i;
  res.expr = solver->mkBVMul(lhs.expr, rhs.expr);
  return res;
}

ExpressionValue
div(llvm::SMTSolverRef solver, DivFeltOp op, const ExpressionValue &lhs,
    const ExpressionValue &rhs) {
  ExpressionValue res;
  auto divRes = lhs.i / rhs.i;
  if (failed(divRes)) {
    op->emitWarning(
        "divisor is not restricted to non-zero values, leading to potential divide-by-zero error."
        " Range of division result will be treated as unbounded."
    );
    res.i = Interval::Entire(lhs.getField());
  } else {
    res.i = *divRes;
  }
  res.expr = solver->mkBVUDiv(lhs.expr, rhs.expr);
  return res;
}

ExpressionValue
mod(llvm::SMTSolverRef solver, const ExpressionValue &lhs, const ExpressionValue &rhs) {
  ExpressionValue res;
  res.i = lhs.i % rhs.i;
  res.expr = solver->mkBVURem(lhs.expr, rhs.expr);
  return res;
}

ExpressionValue
cmp(llvm::SMTSolverRef solver, CmpOp op, const ExpressionValue &lhs, const ExpressionValue &rhs) {
  ExpressionValue res;
  res.i = Interval::Boolean(lhs.getField());
  switch (op.getPredicate()) {
  case FeltCmpPredicate::EQ:
    res.expr = solver->mkEqual(lhs.expr, rhs.expr);
    res.i = lhs.i.intersect(rhs.i);
    break;
  case FeltCmpPredicate::NE:
    res.expr = solver->mkNot(solver->mkEqual(lhs.expr, rhs.expr));
    break;
  case FeltCmpPredicate::LT:
    res.expr = solver->mkBVUlt(lhs.expr, rhs.expr);
    break;
  case FeltCmpPredicate::LE:
    res.expr = solver->mkBVUle(lhs.expr, rhs.expr);
    break;
  case FeltCmpPredicate::GT:
    res.expr = solver->mkBVUgt(lhs.expr, rhs.expr);
    break;
  case FeltCmpPredicate::GE:
    res.expr = solver->mkBVUge(lhs.expr, rhs.expr);
    break;
  }
  return res;
}

ExpressionValue
boolAnd(llvm::SMTSolverRef solver, const ExpressionValue &lhs, const ExpressionValue &rhs) {
  ExpressionValue res;
  res.i = boolAnd(lhs.i, rhs.i);
  res.expr = solver->mkAnd(lhs.expr, rhs.expr);
  return res;
}

ExpressionValue
boolOr(llvm::SMTSolverRef solver, const ExpressionValue &lhs, const ExpressionValue &rhs) {
  ExpressionValue res;
  res.i = boolOr(lhs.i, rhs.i);
  res.expr = solver->mkOr(lhs.expr, rhs.expr);
  return res;
}

ExpressionValue
boolXor(llvm::SMTSolverRef solver, const ExpressionValue &lhs, const ExpressionValue &rhs) {
  ExpressionValue res;
  res.i = boolXor(lhs.i, rhs.i);
  // There's no Xor, so we do (L || R) && !(L && R)
  res.expr = solver->mkAnd(
      solver->mkOr(lhs.expr, rhs.expr), solver->mkNot(solver->mkAnd(lhs.expr, rhs.expr))
  );
  return res;
}

ExpressionValue fallbackBinaryOp(
    llvm::SMTSolverRef solver, Operation *op, const ExpressionValue &lhs, const ExpressionValue &rhs
) {
  ExpressionValue res;
  res.i = Interval::Entire(lhs.getField());
  res.expr = TypeSwitch<Operation *, llvm::SMTExprRef>(op)
                 .Case<AndFeltOp>([&](AndFeltOp _) { return solver->mkBVAnd(lhs.expr, rhs.expr); })
                 .Case<OrFeltOp>([&](OrFeltOp _) { return solver->mkBVOr(lhs.expr, rhs.expr); })
                 .Case<XorFeltOp>([&](XorFeltOp _) { return solver->mkBVXor(lhs.expr, rhs.expr); })
                 .Case<ShlFeltOp>([&](ShlFeltOp _) { return solver->mkBVShl(lhs.expr, rhs.expr); })
                 .Case<ShrFeltOp>([&](ShrFeltOp _) {
    return solver->mkBVLshr(lhs.expr, rhs.expr);
  }).Default([&](Operation *unsupported) {
    llvm::report_fatal_error(
        "no fallback provided for " + mlir::Twine(unsupported->getName().getStringRef())
    );
    return nullptr;
  });

  return res;
}

ExpressionValue neg(llvm::SMTSolverRef solver, const ExpressionValue &val) {
  ExpressionValue res;
  res.i = -val.i;
  res.expr = solver->mkBVNeg(val.expr);
  return res;
}

ExpressionValue notOp(llvm::SMTSolverRef solver, const ExpressionValue &val) {
  ExpressionValue res;
  // TODO: reason about this slightly better
  res.i = Interval::Entire(val.getField());
  res.expr = solver->mkBVNot(val.expr);
  return res;
}

ExpressionValue boolNot(llvm::SMTSolverRef solver, const ExpressionValue &val) {
  ExpressionValue res;
  res.i = boolNot(val.i);
  res.expr = solver->mkBVNot(val.expr);
  return res;
}

ExpressionValue
fallbackUnaryOp(llvm::SMTSolverRef solver, Operation *op, const ExpressionValue &val) {
  const Field &field = val.getField();
  ExpressionValue res;
  res.i = Interval::Entire(field);
  res.expr = TypeSwitch<Operation *, llvm::SMTExprRef>(op)
                 .Case<InvFeltOp>([&](InvFeltOp _) {
    // The definition of an inverse X^-1 is Y s.t. XY % prime = 1.
    // To create this expression, we create a new symbol for Y and add the
    // XY % prime = 1 constraint to the solver.
    std::string symName = buildStringViaInsertionOp(*op);
    llvm::SMTExprRef invSym = field.createSymbol(solver, symName.c_str());
    llvm::SMTExprRef one = solver->mkBitvector(field.one(), field.bitWidth());
    llvm::SMTExprRef prime = solver->mkBitvector(field.prime(), field.bitWidth());
    llvm::SMTExprRef mult = solver->mkBVMul(val.getExpr(), invSym);
    llvm::SMTExprRef mod = solver->mkBVURem(mult, prime);
    llvm::SMTExprRef constraint = solver->mkEqual(mod, one);
    solver->addConstraint(constraint);
    return invSym;
  }).Default([&](Operation *unsupported) {
    llvm::report_fatal_error(
        "no fallback provided for " + mlir::Twine(op->getName().getStringRef())
    );
    return nullptr;
  });

  return res;
}

void ExpressionValue::print(mlir::raw_ostream &os) const {
  if (expr) {
    expr->print(os);
  } else {
    os << "<null expression>";
  }

  os << " ( interval: " << i << " )";
}

/* IntervalAnalysisLattice */

ChangeResult IntervalAnalysisLattice::join(const AbstractDenseLattice &other) {
  const auto *rhs = dynamic_cast<const IntervalAnalysisLattice *>(&other);
  if (!rhs) {
    llvm::report_fatal_error("invalid join lattice type");
  }
  ChangeResult res = ChangeResult::NoChange;
  for (auto &[k, v] : rhs->valMap) {
    auto it = valMap.find(k);
    if (it == valMap.end() || it->second != v) {
      valMap[k] = v;
      res |= ChangeResult::Change;
    }
  }
  for (auto &v : rhs->constraints) {
    if (!constraints.contains(v)) {
      constraints.insert(v);
      res |= ChangeResult::Change;
    }
  }
  for (auto &[e, i] : rhs->intervals) {
    auto it = intervals.find(e);
    if (it == intervals.end() || it->second != i) {
      intervals[e] = i;
      res |= ChangeResult::Change;
    }
  }
  return res;
}

void IntervalAnalysisLattice::print(mlir::raw_ostream &os) const {
  os << "IntervalAnalysisLattice { ";
  for (auto &[ref, val] : valMap) {
    os << "\n    (valMap) " << ref << " := " << val;
  }
  for (auto &[expr, interval] : intervals) {
    os << "\n    (intervals) ";
    if (!expr) {
      os << "<null expr>";
    } else {
      expr->print(os);
    }
    os << " in " << interval;
  }
  if (!valMap.empty()) {
    os << '\n';
  }
  os << '}';
}

FailureOr<IntervalAnalysisLattice::LatticeValue> IntervalAnalysisLattice::getValue(Value v) const {
  auto it = valMap.find(v);
  if (it == valMap.end()) {
    return failure();
  }
  return it->second;
}

FailureOr<IntervalAnalysisLattice::LatticeValue>
IntervalAnalysisLattice::getValue(Value v, StringAttr f) const {
  auto it = fieldMap.find(v);
  if (it == fieldMap.end()) {
    return failure();
  }
  auto fit = it->second.find(f);
  if (fit == it->second.end()) {
    return failure();
  }
  return fit->second;
}

ChangeResult IntervalAnalysisLattice::setValue(Value v, ExpressionValue e) {
  LatticeValue val(e);
  if (valMap[v] == val) {
    return ChangeResult::NoChange;
  }
  valMap[v] = val;
  intervals[e.getExpr()] = e.getInterval();
  return ChangeResult::Change;
}

ChangeResult IntervalAnalysisLattice::setValue(Value v, StringAttr f, ExpressionValue e) {
  LatticeValue val(e);
  if (fieldMap[v][f] == val) {
    return ChangeResult::NoChange;
  }
  fieldMap[v][f] = val;
  intervals[e.getExpr()] = e.getInterval();
  return ChangeResult::Change;
}

ChangeResult IntervalAnalysisLattice::addSolverConstraint(ExpressionValue e) {
  if (!constraints.contains(e)) {
    constraints.insert(e);
    return ChangeResult::Change;
  }
  return ChangeResult::NoChange;
}

FailureOr<Interval> IntervalAnalysisLattice::findInterval(llvm::SMTExprRef expr) const {
  auto it = intervals.find(expr);
  if (it != intervals.end()) {
    return it->second;
  }
  return failure();
}

/* IntervalDataFlowAnalysis */

/// @brief The interval analysis is intraprocedural only for now, so this control
/// flow transfer function passes no data to the callee and sets the post-call
/// state to that of the pre-call state (i.e., calls are ignored).
void IntervalDataFlowAnalysis::visitCallControlFlowTransfer(
    CallOpInterface call, dataflow::CallControlFlowAction action,
    const IntervalAnalysisLattice &before, IntervalAnalysisLattice *after
) {
  /// `action == CallControlFlowAction::Enter` indicates that:
  ///   - `before` is the state before the call operation;
  ///   - `after` is the state at the beginning of the callee entry block;
  if (action == dataflow::CallControlFlowAction::EnterCallee) {
    // We skip updating the incoming lattice for function calls,
    // as values are relative to the containing function/struct, so we don't need to pollute
    // the callee with the callers values.
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

    // The lattice at the return is the lattice before the call
    propagateIfChanged(after, after->join(*beforeCall));
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

void IntervalDataFlowAnalysis::visitOperation(
    Operation *op, const Lattice &before, Lattice *after
) {
  ChangeResult changed = after->join(before);

  llvm::SmallVector<LatticeValue> operandVals;

  auto constrainRefLattice = _dataflowSolver.lookupState<ConstrainRefLattice>(op);
  ensure(constrainRefLattice, "failed to get lattice");

  for (OpOperand &operand : op->getOpOperands()) {
    Value val = operand.get();
    // First, lookup the operand value in the before state.
    auto priorState = before.getValue(val);
    if (succeeded(priorState) && priorState->getScalarValue().getExpr() != nullptr) {
      operandVals.push_back(*priorState);
      continue;
    }

    // Else, look up the stored value by constrain ref.
    // We only care about scalar type values, so we ignore composite types, which
    // are currently limited to non-Signal structs and arrays.
    Type valTy = val.getType();
    if (mlir::isa<ArrayType, StructType>(valTy) && !isSignalType(valTy)) {
      operandVals.push_back(LatticeValue());
      continue;
    }

    ConstrainRefLatticeValue refSet = constrainRefLattice->getOrDefault(val);
    ensure(refSet.isScalar(), "should have ruled out array values already");

    if (refSet.getScalarValue().empty()) {
      // If we can't compute the reference, then there must be some unsupported
      // op the reference analysis cannot handle. We emit a warning and return
      // early, since there's no meaningful computation we can do for this op.
      op->emitWarning() << "state of " << val
                        << " is empty; defining operation is unsupported by constrain ref analysis";
      propagateIfChanged(after, changed);
      return;
    } else if (!refSet.isSingleValue()) {
      std::string warning;
      debug::Appender(warning) << "operand " << val << " is not a single value " << refSet
                               << ", overapproximating";
      op->emitWarning(warning);
      // Here, we will override the prior lattice value with a new symbol, representing
      // "any" value, then use that value for the operands.
      ExpressionValue anyVal(field.get(), createFeltSymbol(val));
      changed |= after->setValue(val, anyVal);
      operandVals.emplace_back(anyVal);
    } else {
      auto ref = refSet.getSingleValue();
      ExpressionValue exprVal(field.get(), getOrCreateSymbol(ref));
      if (succeeded(priorState)) {
        exprVal = exprVal.withInterval(priorState->getScalarValue().getInterval());
      }
      changed |= after->setValue(val, exprVal);
      operandVals.emplace_back(exprVal);
    }
  }

  // Now, the way we update is dependent on the type of the operation.
  if (!isConsideredOp(op)) {
    op->emitWarning("unconsidered operation type, analysis may be incomplete");
  }

  if (isConstOp(op)) {
    auto constVal = getConst(op);
    auto expr = createConstBitvectorExpr(constVal);
    ExpressionValue latticeVal(field.get(), expr, constVal);
    changed |= after->setValue(op->getResult(0), latticeVal);
  } else if (isArithmeticOp(op)) {
    ensure(operandVals.size() <= 2, "arithmetic op with the wrong number of operands");
    ExpressionValue result;
    if (operandVals.size() == 2) {
      result = performBinaryArithmetic(op, operandVals[0], operandVals[1]);
    } else {
      result = performUnaryArithmetic(op, operandVals[0]);
    }

    changed |= after->setValue(op->getResult(0), result);
  } else if (EmitEqualityOp emitEq = mlir::dyn_cast<EmitEqualityOp>(op)) {
    ensure(operandVals.size() == 2, "constraint op with the wrong number of operands");
    Value lhsVal = emitEq.getLhs(), rhsVal = emitEq.getRhs();
    ExpressionValue lhsExpr = operandVals[0].getScalarValue();
    ExpressionValue rhsExpr = operandVals[1].getScalarValue();

    // Special handling for generalized (s - c0) * (s - c1) * ... * (s - cN) = 0 patterns.
    // These patterns enforce that s is one of c0, ..., cN.
    auto res = getGeneralizedDecompInterval(constrainRefLattice, lhsVal, rhsVal);
    if (succeeded(res)) {
      for (Value signalVal : res->first) {
        changed |= applyInterval(emitEq, after, signalVal, res->second);
      }
    }

    ExpressionValue constraint = intersection(smtSolver, lhsExpr, rhsExpr);
    // Update the LHS and RHS to the same value, but restricted intervals
    // based on the constraints
    changed |= applyInterval(emitEq, after, lhsVal, constraint.getInterval());
    changed |= applyInterval(emitEq, after, rhsVal, constraint.getInterval());
    changed |= after->addSolverConstraint(constraint);
  } else if (AssertOp assertOp = mlir::dyn_cast<AssertOp>(op)) {
    ensure(operandVals.size() == 1, "assert op with the wrong number of operands");
    // assert enforces that the operand is true. So we apply an interval of [1, 1]
    // to the operand.
    changed |= applyInterval(
        assertOp, after, assertOp.getCondition(),
        Interval::Degenerate(field.get(), field.get().one())
    );
    // Also add the solver constraint that the expression must be true.
    auto assertExpr = operandVals[0].getScalarValue();
    changed |= after->addSolverConstraint(assertExpr);
  } else if (auto readf = mlir::dyn_cast<FieldReadOp>(op)) {
    if (isSignalType(readf.getComponent().getType())) {
      // The reg value read from the signal type is equal to the value of the Signal
      // struct overall.
      changed |= after->setValue(readf.getVal(), operandVals[0].getScalarValue());
    } else if (auto storedVal =
                   before.getValue(readf.getComponent(), readf.getFieldNameAttr().getAttr());
               succeeded(storedVal)) {
      // The result value is the value previously written to this field.
      changed |= after->setValue(readf.getVal(), storedVal->getScalarValue());
    }
  } else if (auto writef = mlir::dyn_cast<FieldWriteOp>(op)) {
    // Update values stored in a field
    changed |= after->setValue(
        writef.getComponent(), writef.getFieldNameAttr().getAttr(), operandVals[1].getScalarValue()
    );
  } else if (isa<IntToFeltOp, FeltToIndexOp>(op)) {
    // Casts don't modify the intervals
    changed |= after->setValue(op->getResult(0), operandVals[0].getScalarValue());
  } else if (
      // We do not need to explicitly handle read ops since they are resolved at the operand value
      // step where constrain refs are queries (with the exception of the Signal struct, see above).
      !isReadOp(op)
      // We do not currently handle return ops as the analysis is currently limited to constrain
      // functions, which return no value.
      && !isReturnOp(op)
      // The analysis ignores definition ops.
      && !isDefinitionOp(op)
      // We do not need to analyze the creation of structs.
      && !mlir::isa<CreateStructOp>(op)
  ) {
    op->emitWarning("unhandled operation, analysis may be incomplete");
  }

  propagateIfChanged(after, changed);
}

llvm::SMTExprRef IntervalDataFlowAnalysis::getOrCreateSymbol(const ConstrainRef &r) {
  auto it = refSymbols.find(r);
  if (it != refSymbols.end()) {
    return it->second;
  }
  auto sym = createFeltSymbol(r);
  refSymbols[r] = sym;
  return sym;
}

llvm::SMTExprRef IntervalDataFlowAnalysis::createFeltSymbol(const ConstrainRef &r) const {
  return createFeltSymbol(buildStringViaPrint(r).c_str());
}

llvm::SMTExprRef IntervalDataFlowAnalysis::createFeltSymbol(Value v) const {
  return createFeltSymbol(buildStringViaPrint(v).c_str());
}

llvm::SMTExprRef IntervalDataFlowAnalysis::createFeltSymbol(const char *name) const {
  return field.get().createSymbol(smtSolver, name);
}

llvm::APSInt IntervalDataFlowAnalysis::getConst(Operation *op) const {
  ensure(isConstOp(op), "op is not a const op");

  llvm::APInt fieldConst =
      TypeSwitch<Operation *, llvm::APInt>(op)
          .Case<FeltConstantOp>([&](FeltConstantOp feltConst) {
    llvm::APSInt constOpVal(feltConst.getValueAttr().getValue());
    return field.get().reduce(constOpVal);
  })
          .Case<arith::ConstantIndexOp>([&](arith::ConstantIndexOp indexConst) {
    return llvm::APInt(field.get().bitWidth(), indexConst.value());
  })
          .Case<arith::ConstantIntOp>([&](arith::ConstantIntOp intConst) {
    return llvm::APInt(field.get().bitWidth(), intConst.value());
  }).Default([](Operation *illegalOp) {
    std::string err;
    debug::Appender(err) << "unhandled getConst case: " << *illegalOp;
    llvm::report_fatal_error(Twine(err));
    return llvm::APInt();
  });
  return llvm::APSInt(fieldConst);
}

ExpressionValue IntervalDataFlowAnalysis::performBinaryArithmetic(
    Operation *op, const LatticeValue &a, const LatticeValue &b
) {
  ensure(isArithmeticOp(op), "is not arithmetic op");

  auto lhs = a.getScalarValue(), rhs = b.getScalarValue();
  ensure(lhs.getExpr(), "cannot perform arithmetic over null lhs smt expr");
  ensure(rhs.getExpr(), "cannot perform arithmetic over null rhs smt expr");

  auto res = TypeSwitch<Operation *, ExpressionValue>(op)
                 .Case<AddFeltOp>([&](AddFeltOp _) { return add(smtSolver, lhs, rhs); })
                 .Case<SubFeltOp>([&](SubFeltOp _) { return sub(smtSolver, lhs, rhs); })
                 .Case<MulFeltOp>([&](MulFeltOp _) { return mul(smtSolver, lhs, rhs); })
                 .Case<DivFeltOp>([&](DivFeltOp divOp) { return div(smtSolver, divOp, lhs, rhs); })
                 .Case<ModFeltOp>([&](ModFeltOp _) { return mod(smtSolver, lhs, rhs); })
                 .Case<CmpOp>([&](CmpOp cmpOp) { return cmp(smtSolver, cmpOp, lhs, rhs); })
                 .Case<AndBoolOp>([&](AndBoolOp _) { return boolAnd(smtSolver, lhs, rhs); })
                 .Case<OrBoolOp>([&](OrBoolOp _) { return boolOr(smtSolver, lhs, rhs); })
                 .Case<XorBoolOp>([&](XorBoolOp _) {
    return boolXor(smtSolver, lhs, rhs);
  }).Default([&](Operation *unsupported) {
    unsupported->emitWarning(
        "unsupported binary arithmetic operation, defaulting to over-approximated intervals"
    );
    return fallbackBinaryOp(smtSolver, unsupported, lhs, rhs);
  });

  ensure(res.getExpr(), "arithmetic produced null smt expr");
  return res;
}

ExpressionValue
IntervalDataFlowAnalysis::performUnaryArithmetic(Operation *op, const LatticeValue &a) {
  ensure(isArithmeticOp(op), "is not arithmetic op");

  auto val = a.getScalarValue();
  ensure(val.getExpr(), "cannot perform arithmetic over null smt expr");

  auto res = TypeSwitch<Operation *, ExpressionValue>(op)
                 .Case<NegFeltOp>([&](NegFeltOp _) { return neg(smtSolver, val); })
                 .Case<NotFeltOp, NotBoolOp>([&](Operation *_) {
    return notOp(smtSolver, val);
  }).Default([&](Operation *unsupported) {
    unsupported->emitWarning(
        "unsupported unary arithmetic operation, defaulting to over-approximated interval"
    );
    return fallbackUnaryOp(smtSolver, unsupported, val);
  });

  ensure(res.getExpr(), "arithmetic produced null smt expr");
  return res;
}

ChangeResult IntervalDataFlowAnalysis::applyInterval(
    Operation *originalOp, Lattice *after, Value val, Interval newInterval
) {
  auto latValRes = after->getValue(val);
  if (failed(latValRes)) {
    // visitOperation didn't add val to the lattice, so there's nothing to do
    return ChangeResult::NoChange;
  }
  ExpressionValue newLatticeVal = latValRes->getScalarValue().withInterval(newInterval);
  ChangeResult res = after->setValue(val, newLatticeVal);
  // To allow the dataflow analysis to do its fixed-point iteration, we need to
  // add the new expression to val's lattice as well.
  Lattice *valLattice = nullptr;
  if (auto valOp = val.getDefiningOp()) {
    // Getting the lattice at valOp gives us the "after" lattice, but we want to
    // update the "before" lattice so that the inputs to visitOperation will be
    // changed.
    if (auto prev = valOp->getPrevNode()) {
      valLattice = getOrCreate<Lattice>(prev);
    } else {
      valLattice = getOrCreate<Lattice>(valOp->getBlock());
    }
  } else if (auto blockArg = mlir::dyn_cast<BlockArgument>(val)) {
    Operation *owningOp = blockArg.getOwner()->getParentOp();
    // Apply the interval from the constrain function inputs to the compute function inputs
    if (auto fnOp = dyn_cast<FuncDefOp>(owningOp); fnOp && fnOp.isStructConstrain() &&
                                                   blockArg.getArgNumber() > 0 &&
                                                   !newInterval.isEntire()) {
      auto structOp = fnOp->getParentOfType<StructDefOp>();
      FuncDefOp computeFn = structOp.getComputeFuncOp();
      Operation *computeEntry = &computeFn.getRegion().front().front();
      BlockArgument computeArg = computeFn.getArgument(blockArg.getArgNumber() - 1);
      Lattice *computeEntryLattice = getOrCreate<Lattice>(computeEntry);
      auto entryLatticeVal = computeEntryLattice->getValue(computeArg);
      ExpressionValue newArgVal;
      if (succeeded(entryLatticeVal)) {
        newArgVal = entryLatticeVal->getScalarValue().withInterval(newInterval);
      } else {
        // We store the interval with an empty expression so that when the operation
        // is visited, the expressions can be properly generated with an existing
        // interval.
        newArgVal = ExpressionValue(nullptr, newInterval);
      }
      ChangeResult computeRes = computeEntryLattice->setValue(computeArg, newArgVal);
      propagateIfChanged(computeEntryLattice, computeRes);
    }
    valLattice = getOrCreate<Lattice>(blockArg.getOwner());
  } else {
    valLattice = getOrCreate<Lattice>(val);
  }
  ensure(valLattice, "val should have a lattice");
  if (valLattice != after) {
    propagateIfChanged(valLattice, valLattice->setValue(val, newLatticeVal));
  }

  // Now we descend into val's operands, if it has any.
  Operation *definingOp = val.getDefiningOp();
  if (!definingOp) {
    return res;
  }

  const Field &f = field.get();

  // This is a rules-based operation. If we have a rule for a given operation,
  // then we can make some kind of update, otherwise we leave the intervals
  // as is.
  // - First we'll define all the rules so the type switch can be less messy

  // cmp.<pred> restricts each side of the comparison if the result is known.
  auto cmpCase = [&](CmpOp cmpOp) {
    // Cmp output range is [0, 1], so in order to do something, we must have newInterval
    // either "true" (1) or "false" (0)
    ensure(
        newInterval.isBoolean(),
        "new interval for CmpOp outside of allowed boolean range or is empty"
    );
    if (!newInterval.isDegenerate()) {
      // The comparison result is unknown, so we can't update the operand ranges
      return ChangeResult::NoChange;
    }

    bool cmpTrue = newInterval.rhs() == f.one();

    Value lhs = cmpOp->getOperand(0), rhs = cmpOp->getOperand(1);
    auto lhsLatValRes = after->getValue(lhs), rhsLatValRes = after->getValue(rhs);
    if (failed(lhsLatValRes) || failed(rhsLatValRes)) {
      return ChangeResult::NoChange;
    }
    ExpressionValue lhsExpr = lhsLatValRes->getScalarValue(),
                    rhsExpr = rhsLatValRes->getScalarValue();

    Interval newLhsInterval, newRhsInterval;
    const Interval &lhsInterval = lhsExpr.getInterval();
    const Interval &rhsInterval = rhsExpr.getInterval();

    FeltCmpPredicate pred = cmpOp.getPredicate();
    // predicate cases
    auto eqCase = [&]() {
      return (pred == FeltCmpPredicate::EQ && cmpTrue) ||
             (pred == FeltCmpPredicate::NE && !cmpTrue);
    };
    auto neCase = [&]() {
      return (pred == FeltCmpPredicate::NE && cmpTrue) ||
             (pred == FeltCmpPredicate::EQ && !cmpTrue);
    };
    auto ltCase = [&]() {
      return (pred == FeltCmpPredicate::LT && cmpTrue) ||
             (pred == FeltCmpPredicate::GE && !cmpTrue);
    };
    auto leCase = [&]() {
      return (pred == FeltCmpPredicate::LE && cmpTrue) ||
             (pred == FeltCmpPredicate::GT && !cmpTrue);
    };
    auto gtCase = [&]() {
      return (pred == FeltCmpPredicate::GT && cmpTrue) ||
             (pred == FeltCmpPredicate::LE && !cmpTrue);
    };
    auto geCase = [&]() {
      return (pred == FeltCmpPredicate::GE && cmpTrue) ||
             (pred == FeltCmpPredicate::LT && !cmpTrue);
    };

    // new intervals based on case
    if (eqCase()) {
      newLhsInterval = newRhsInterval = lhsInterval.intersect(rhsInterval);
    } else if (neCase()) {

      if (lhsInterval.isDegenerate() && rhsInterval.isDegenerate() && lhsInterval == rhsInterval) {
        // In this case, we know lhs and rhs cannot satisfy this assertion, so they have
        // an empty value range.
        newLhsInterval = newRhsInterval = Interval::Empty(f);
      } else if (lhsInterval.isDegenerate()) {
        // rhs must not overlap with lhs
        newLhsInterval = lhsInterval;
        newRhsInterval = rhsInterval.difference(lhsInterval);
      } else if (rhsInterval.isDegenerate()) {
        // lhs must not overlap with rhs
        newLhsInterval = lhsInterval.difference(rhsInterval);
        newRhsInterval = rhsInterval;
      } else {
        // Leave unchanged
        newLhsInterval = lhsInterval;
        newRhsInterval = rhsInterval;
      }
    } else if (ltCase()) {
      newLhsInterval = lhsInterval.toUnreduced().computeLTPart(rhsInterval.toUnreduced()).reduce(f);
      newRhsInterval = rhsInterval.toUnreduced().computeGEPart(lhsInterval.toUnreduced()).reduce(f);
    } else if (leCase()) {
      newLhsInterval = lhsInterval.toUnreduced().computeLEPart(rhsInterval.toUnreduced()).reduce(f);
      newRhsInterval = rhsInterval.toUnreduced().computeGTPart(lhsInterval.toUnreduced()).reduce(f);
    } else if (gtCase()) {
      newLhsInterval = lhsInterval.toUnreduced().computeGTPart(rhsInterval.toUnreduced()).reduce(f);
      newRhsInterval = rhsInterval.toUnreduced().computeLEPart(lhsInterval.toUnreduced()).reduce(f);
    } else if (geCase()) {
      newLhsInterval = lhsInterval.toUnreduced().computeGEPart(rhsInterval.toUnreduced()).reduce(f);
      newRhsInterval = rhsInterval.toUnreduced().computeLTPart(lhsInterval.toUnreduced()).reduce(f);
    } else {
      cmpOp->emitWarning("unhandled cmp predicate");
      return ChangeResult::NoChange;
    }

    // Now we recurse to each operand
    return applyInterval(originalOp, after, lhs, newLhsInterval) |
           applyInterval(originalOp, after, rhs, newRhsInterval);
  };

  // If the result of a multiplication is non-zero, then both operands must be
  // non-zero.
  auto mulCase = [&](MulFeltOp mulOp) {
    auto zeroInt = Interval::Degenerate(f, f.zero());
    if (newInterval.intersect(zeroInt).isNotEmpty()) {
      // The multiplication may be zero, so we can't reduce the operands to be non-zero
      return ChangeResult::NoChange;
    }

    Value lhs = mulOp->getOperand(0), rhs = mulOp->getOperand(1);
    auto lhsLatValRes = after->getValue(lhs), rhsLatValRes = after->getValue(rhs);
    if (failed(lhsLatValRes) || failed(rhsLatValRes)) {
      return ChangeResult::NoChange;
    }
    ExpressionValue lhsExpr = lhsLatValRes->getScalarValue(),
                    rhsExpr = rhsLatValRes->getScalarValue();
    Interval newLhsInterval = lhsExpr.getInterval().difference(zeroInt);
    Interval newRhsInterval = rhsExpr.getInterval().difference(zeroInt);
    return applyInterval(originalOp, after, lhs, newLhsInterval) |
           applyInterval(originalOp, after, rhs, newRhsInterval);
  };

  // We have a special case for the Signal struct: if this value is created
  // from reading a Signal struct's reg field, we also apply the interval to
  // the struct itself.
  auto readfCase = [&](FieldReadOp readfOp) {
    Value comp = readfOp.getComponent();
    if (isSignalType(comp.getType())) {
      return applyInterval(originalOp, after, comp, newInterval);
    }
    return ChangeResult::NoChange;
  };

  // - Apply the rules given the op.
  // NOTE: disabling clang-format for this because it makes the last case statement
  // look ugly.
  // clang-format off
  res |= TypeSwitch<Operation *, ChangeResult>(definingOp)
            .Case<CmpOp>([&](CmpOp op) { return cmpCase(op); })
            .Case<MulFeltOp>([&](MulFeltOp op) { return mulCase(op); })
            .Case<FieldReadOp>([&](FieldReadOp op){ return readfCase(op); })
            .Default([&](Operation *_) { return ChangeResult::NoChange; });
  // clang-format on

  return res;
}

FailureOr<std::pair<DenseSet<Value>, Interval>>
IntervalDataFlowAnalysis::getGeneralizedDecompInterval(
    const ConstrainRefLattice *constrainRefLattice, Value lhs, Value rhs
) {
  auto isZeroConst = [this](Value v) {
    Operation *op = v.getDefiningOp();
    if (!op) {
      return false;
    }
    if (!isConstOp(op)) {
      return false;
    }
    llvm::APSInt c = getConst(op);
    return c == field.get().zero();
  };
  bool lhsIsZero = isZeroConst(lhs), rhsIsZero = isZeroConst(rhs);
  Value exprTree = nullptr;
  if (lhsIsZero && !rhsIsZero) {
    exprTree = rhs;
  } else if (!lhsIsZero && rhsIsZero) {
    exprTree = lhs;
  } else {
    return failure();
  }

  // We now explore the expression tree for multiplications of subtractions/signal values.
  std::optional<ConstrainRef> signalRef = std::nullopt;
  DenseSet<Value> signalVals;
  SmallVector<APSInt> consts;
  SmallVector<Value> frontier {exprTree};
  while (!frontier.empty()) {
    Value v = frontier.back();
    frontier.pop_back();
    Operation *op = v.getDefiningOp();

    FeltConstantOp c;
    Value signalVal;
    auto handleRefValue = [&constrainRefLattice, &signalRef, &signalVal, &signalVals]() {
      ConstrainRefLatticeValue refSet = constrainRefLattice->getOrDefault(signalVal);
      if (!refSet.isScalar() || !refSet.isSingleValue()) {
        return failure();
      }
      ConstrainRef r = refSet.getSingleValue();
      if (signalRef.has_value() && signalRef.value() != r) {
        return failure();
      } else if (!signalRef.has_value()) {
        signalRef = r;
      }
      signalVals.insert(signalVal);
      return success();
    };

    auto subPattern = m_CommutativeOp<SubFeltOp>(m_RefValue(&signalVal), m_Constant(&c));
    if (op && matchPattern(op, subPattern)) {
      if (failed(handleRefValue())) {
        return failure();
      }
      auto constInt = APSInt(c.getValueAttr().getValue());
      consts.push_back(field.get().reduce(constInt));
      continue;
    } else if (m_RefValue(&signalVal).match(v)) {
      if (failed(handleRefValue())) {
        return failure();
      }
      consts.push_back(field.get().zero());
      continue;
    }

    Value a, b;
    auto mulPattern = m_CommutativeOp<MulFeltOp>(matchers::m_Any(&a), matchers::m_Any(&b));
    if (op && matchPattern(op, mulPattern)) {
      frontier.push_back(a);
      frontier.push_back(b);
      continue;
    }

    return failure();
  }

  // Now, we aggregate the Interval. If we have sparse values (e.g., 0, 2, 4),
  // we will create a larger range of [0, 4], since we don't support multiple intervals.
  std::sort(consts.begin(), consts.end());
  Interval iv = Interval::TypeA(field.get(), consts.front(), consts.back());
  return std::make_pair(std::move(signalVals), iv);
}

/* StructIntervals */

LogicalResult
StructIntervals::computeIntervals(mlir::DataFlowSolver &solver, IntervalAnalysisContext &ctx) {

  auto computeIntervalsImpl = [&solver, &ctx, this](
                                  FuncDefOp fn,
                                  llvm::MapVector<ConstrainRef, Interval> &fieldRanges,
                                  llvm::SetVector<ExpressionValue> &solverConstraints
                              ) {
    // Get the lattice at the end of the function.
    Operation *fnEnd = fn.getRegion().back().getTerminator();

    const IntervalAnalysisLattice *lattice = solver.lookupState<IntervalAnalysisLattice>(fnEnd);

    solverConstraints = lattice->getConstraints();

    for (const auto &ref : ConstrainRef::getAllConstrainRefs(structDef, fn)) {
      // We only want to compute intervals for field elements and not composite types,
      // with the exception of the Signal struct.
      if (!ref.isScalar() && !ref.isSignal()) {
        continue;
      }
      // We also don't want to show the interval for a Signal and its internal reg.
      if (auto parentOr = ref.getParentPrefix(); succeeded(parentOr) && parentOr->isSignal()) {
        continue;
      }
      auto symbol = ctx.getSymbol(ref);
      auto intervalRes = lattice->findInterval(symbol);
      if (succeeded(intervalRes)) {
        fieldRanges[ref] = *intervalRes;
      } else {
        fieldRanges[ref] = Interval::Entire(ctx.field);
      }
    }
  };

  computeIntervalsImpl(structDef.getComputeFuncOp(), computeFieldRanges, computeSolverConstraints);
  computeIntervalsImpl(
      structDef.getConstrainFuncOp(), constrainFieldRanges, constrainSolverConstraints
  );

  return success();
}

void StructIntervals::print(mlir::raw_ostream &os, bool withConstraints, bool printCompute) const {
  auto writeIntervals =
      [&os, &withConstraints](
          const char *fnName, const llvm::MapVector<ConstrainRef, Interval> &fieldRanges,
          const llvm::SetVector<ExpressionValue> &solverConstraints, bool printName
      ) {
    int indent = 4;
    if (printName) {
      os << '\n';
      os.indent(indent) << fnName << " {";
      indent += 4;
    }

    if (fieldRanges.empty()) {
      os << "}\n";
      return;
    }

    for (auto &[ref, interval] : fieldRanges) {
      os << '\n';
      os.indent(indent) << ref << " in " << interval;
    }

    if (withConstraints) {
      os << "\n\n";
      os.indent(indent) << "Solver Constraints { ";
      if (solverConstraints.empty()) {
        os << "}\n";
      } else {
        for (const auto &e : solverConstraints) {
          os << '\n';
          os.indent(indent + 4);
          e.getExpr()->print(os);
        }
        os << '\n';
        os.indent(indent) << '}';
      }
    }

    if (printName) {
      os << '\n';
      os.indent(indent - 4) << '}';
    }
  };

  os << "StructIntervals { ";
  if (constrainFieldRanges.empty() && (!printCompute || computeFieldRanges.empty())) {
    os << "}\n";
    return;
  }

  if (printCompute) {
    writeIntervals(FUNC_NAME_COMPUTE, computeFieldRanges, computeSolverConstraints, printCompute);
  }
  writeIntervals(
      FUNC_NAME_CONSTRAIN, constrainFieldRanges, constrainSolverConstraints, printCompute
  );

  os << "\n}\n";
}

} // namespace llzk
