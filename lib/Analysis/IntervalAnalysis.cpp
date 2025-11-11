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

#include <mlir/Dialect/SCF/IR/SCF.h>

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
boolToFelt(llvm::SMTSolverRef solver, const ExpressionValue &expr, unsigned bitwidth) {
  llvm::SMTExprRef zero = solver->mkBitvector(mlir::APSInt::get(0), bitwidth);
  llvm::SMTExprRef one = solver->mkBitvector(mlir::APSInt::get(1), bitwidth);
  llvm::SMTExprRef boolToFeltConv = solver->mkIte(expr.getExpr(), one, zero);
  return expr.withExpression(boolToFeltConv);
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
    )
        .report();
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
bitAnd(llvm::SMTSolverRef solver, const ExpressionValue &lhs, const ExpressionValue &rhs) {
  ExpressionValue res;
  res.i = lhs.i & rhs.i;
  res.expr = solver->mkBVAnd(lhs.expr, rhs.expr);
  return res;
}

ExpressionValue
shiftLeft(llvm::SMTSolverRef solver, const ExpressionValue &lhs, const ExpressionValue &rhs) {
  ExpressionValue res;
  res.i = lhs.i << rhs.i;
  res.expr = solver->mkBVShl(lhs.expr, rhs.expr);
  return res;
}

ExpressionValue
shiftRight(llvm::SMTSolverRef solver, const ExpressionValue &lhs, const ExpressionValue &rhs) {
  ExpressionValue res;
  res.i = lhs.i >> rhs.i;
  res.expr = solver->mkBVLshr(lhs.expr, rhs.expr);
  return res;
}

ExpressionValue
cmp(llvm::SMTSolverRef solver, CmpOp op, const ExpressionValue &lhs, const ExpressionValue &rhs) {
  ExpressionValue res;
  const Field &f = lhs.getField();
  // Default result is any boolean output for when we are unsure about the comparison result.
  res.i = Interval::Boolean(f);
  switch (op.getPredicate()) {
  case FeltCmpPredicate::EQ:
    res.expr = solver->mkEqual(lhs.expr, rhs.expr);
    if (lhs.i.isDegenerate() && rhs.i.isDegenerate()) {
      res.i = lhs.i == rhs.i ? Interval::True(f) : Interval::False(f);
    } else if (lhs.i.intersect(rhs.i).isEmpty()) {
      res.i = Interval::False(f);
    }
    break;
  case FeltCmpPredicate::NE:
    res.expr = solver->mkNot(solver->mkEqual(lhs.expr, rhs.expr));
    if (lhs.i.isDegenerate() && rhs.i.isDegenerate()) {
      res.i = lhs.i != rhs.i ? Interval::True(f) : Interval::False(f);
    } else if (lhs.i.intersect(rhs.i).isEmpty()) {
      res.i = Interval::True(f);
    }
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
                 .Case<OrFeltOp>([&](auto) { return solver->mkBVOr(lhs.expr, rhs.expr); })
                 .Case<XorFeltOp>([&](auto) {
    return solver->mkBVXor(lhs.expr, rhs.expr);
  }).Default([&](auto *unsupported) {
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
  res.i = ~val.i;
  res.expr = solver->mkBVNot(val.expr);
  return res;
}

ExpressionValue boolNot(llvm::SMTSolverRef solver, const ExpressionValue &val) {
  ExpressionValue res;
  res.i = boolNot(val.i);
  res.expr = solver->mkNot(val.expr);
  return res;
}

ExpressionValue
fallbackUnaryOp(llvm::SMTSolverRef solver, Operation *op, const ExpressionValue &val) {
  const Field &field = val.getField();
  ExpressionValue res;
  res.i = Interval::Entire(field);
  res.expr = TypeSwitch<Operation *, llvm::SMTExprRef>(op)
                 .Case<InvFeltOp>([&](auto) {
    // The definition of an inverse X^-1 is Y s.t. XY % prime = 1.
    // To create this expression, we create a new symbol for Y and add the
    // XY % prime = 1 constraint to the solver.
    std::string symName = buildStringViaInsertionOp(*op);
    llvm::SMTExprRef invSym = field.createSymbol(solver, symName.c_str());
    llvm::SMTExprRef one = solver->mkBitvector(APSInt::get(1), field.bitWidth());
    llvm::SMTExprRef prime = solver->mkBitvector(toAPSInt(field.prime()), field.bitWidth());
    llvm::SMTExprRef mult = solver->mkBVMul(val.getExpr(), invSym);
    llvm::SMTExprRef mod = solver->mkBVURem(mult, prime);
    llvm::SMTExprRef constraint = solver->mkEqual(mod, one);
    solver->addConstraint(constraint);
    return invSym;
  }).Default([](Operation *unsupported) {
    llvm::report_fatal_error(
        "no fallback provided for " + mlir::Twine(unsupported->getName().getStringRef())
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

ChangeResult IntervalAnalysisLattice::setValue(Value v, const LatticeValue &val) {
  if (valMap[v] == val) {
    return ChangeResult::NoChange;
  }
  valMap[v] = val;
  ExpressionValue e = val.foldToScalar();
  intervals[e.getExpr()] = e.getInterval();
  return ChangeResult::Change;
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

ChangeResult IntervalAnalysisLattice::setInterval(llvm::SMTExprRef expr, const Interval &i) {
  auto it = intervals.find(expr);
  if (it != intervals.end() && it->second == i) {
    return ChangeResult::NoChange;
  }
  intervals[expr] = i;
  return ChangeResult::Change;
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
    const dataflow::AbstractDenseLattice *beforeCall = getLattice(getProgramPointBefore(call));
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

const SourceRefLattice *
IntervalDataFlowAnalysis::getSourceRefLattice(Operation *baseOp, Value val) {
  ProgramPoint *pp = _dataflowSolver.getProgramPointAfter(baseOp);
  auto defaultSourceRefLattice = _dataflowSolver.lookupState<SourceRefLattice>(pp);
  ensure(defaultSourceRefLattice, "failed to get lattice");
  if (Operation *defOp = val.getDefiningOp()) {
    ProgramPoint *defPoint = _dataflowSolver.getProgramPointAfter(defOp);
    auto sourceRefLattice = _dataflowSolver.lookupState<SourceRefLattice>(defPoint);
    ensure(sourceRefLattice, "failed to get SourceRefLattice for value");
    return sourceRefLattice;
  }
  return defaultSourceRefLattice;
}

mlir::LogicalResult
IntervalDataFlowAnalysis::visitOperation(Operation *op, const Lattice &before, Lattice *after) {
  // We only perform the visitation on operations within functions
  FuncDefOp fn = op->getParentOfType<FuncDefOp>();
  if (!fn) {
    return success();
  }

  ChangeResult changed = ChangeResult::NoChange;
  // We always propagate the values of the function args from the function
  // entry as the function context; if the input values are changed, this will
  // force the recomputation of intervals throughout the function.
  for (BlockArgument blockArg : fn.getArguments()) {
    auto blockArgLookupRes = before.getValue(blockArg);
    if (succeeded(blockArgLookupRes)) {
      changed |= after->setValue(blockArg, *blockArgLookupRes);
    }
  }

  auto getAfter = [&](Value val) {
    if (Operation *defOp = val.getDefiningOp()) {
      return getLattice(getProgramPointAfter(defOp));
    } else if (auto blockArg = dyn_cast<BlockArgument>(val)) {
      Operation *blockEntry = &blockArg.getOwner()->front();
      return getLattice(getProgramPointBefore(blockEntry));
    }
    return getLattice(getProgramPointBefore(op));
  };

  llvm::SmallVector<LatticeValue> operandVals;
  llvm::SmallVector<std::optional<SourceRef>> operandRefs;
  for (OpOperand &operand : op->getOpOperands()) {
    Value val = operand.get();
    SourceRefLatticeValue refSet = getSourceRefLattice(op, val)->getOrDefault(val);
    if (refSet.isSingleValue()) {
      operandRefs.push_back(refSet.getSingleValue());
    } else {
      operandRefs.push_back(std::nullopt);
    }
    // First, lookup the operand value after it is initialized
    Lattice *valLattice = getAfter(val);
    auto priorState = valLattice->getValue(val);
    if (succeeded(priorState) && priorState->getScalarValue().getExpr() != nullptr) {
      operandVals.push_back(*priorState);
      changed |= after->setValue(val, *priorState);
      continue;
    }

    // Else, look up the stored value by `SourceRef`.
    // We only care about scalar type values, so we ignore composite types, which
    // are currently limited to non-Signal structs and arrays.
    Type valTy = val.getType();
    if (llvm::isa<ArrayType, StructType>(valTy)) {
      LatticeValue empty;
      operandVals.push_back(empty);
      changed |= after->setValue(val, empty);
      continue;
    }

    ensure(refSet.isScalar(), "should have ruled out array values already");

    if (refSet.getScalarValue().empty()) {
      // If we can't compute the reference, then there must be some unsupported
      // op the reference analysis cannot handle. We emit a warning and return
      // early, since there's no meaningful computation we can do for this op.
      op->emitWarning()
          .append(
              "state of ", val, " is empty; defining operation is unsupported by SourceRef analysis"
          )
          .report();
      propagateIfChanged(after, changed);
      // We still return success so we can return overapproximated and partial
      // results to the user.
      return success();
    } else if (!refSet.isSingleValue()) {
      std::string warning;
      debug::Appender(warning) << "operand " << val << " is not a single value " << refSet
                               << ", overapproximating";
      op->emitWarning(warning).report();
      // Here, we will override the prior lattice value with a new symbol, representing
      // "any" value, then use that value for the operands.
      ExpressionValue anyVal(field.get(), createFeltSymbol(val));
      changed |= after->setValue(val, anyVal);
      operandVals.emplace_back(anyVal);
    } else {
      const SourceRef &ref = refSet.getSingleValue();
      ExpressionValue exprVal(field.get(), getOrCreateSymbol(ref));
      if (succeeded(priorState)) {
        exprVal = exprVal.withInterval(priorState->getScalarValue().getInterval());
      }
      changed |= after->setValue(val, exprVal);
      operandVals.emplace_back(exprVal);
    }

    // Since we initialized a value that was not found in the before lattice,
    // update that value in the lattice so we can find it later, but we don't
    // need to propagate the changes, since we already have what we need.
    auto res = after->getValue(val);
    ensure(succeeded(res), "expected precondition is that value is set");
    (void)valLattice->setValue(val, *res);
  }

  // Now, the way we update is dependent on the type of the operation.
  if (isConstOp(op)) {
    llvm::DynamicAPInt constVal = getConst(op);
    llvm::SMTExprRef expr = createConstBitvectorExpr(constVal);
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
  } else if (EmitEqualityOp emitEq = llvm::dyn_cast<EmitEqualityOp>(op)) {
    ensure(operandVals.size() == 2, "constraint op with the wrong number of operands");
    Value lhsVal = emitEq.getLhs(), rhsVal = emitEq.getRhs();
    ExpressionValue lhsExpr = operandVals[0].getScalarValue();
    ExpressionValue rhsExpr = operandVals[1].getScalarValue();

    // Special handling for generalized (s - c0) * (s - c1) * ... * (s - cN) = 0 patterns.
    // These patterns enforce that s is one of c0, ..., cN.
    auto res = getGeneralizedDecompInterval(op, lhsVal, rhsVal);
    if (succeeded(res)) {
      for (Value signalVal : res->first) {
        changed |= applyInterval(emitEq, after, getAfter(signalVal), signalVal, res->second);
      }
    }

    ExpressionValue constraint = intersection(smtSolver, lhsExpr, rhsExpr);
    // Update the LHS and RHS to the same value, but restricted intervals
    // based on the constraints.
    const Interval &constrainInterval = constraint.getInterval();
    changed |= applyInterval(emitEq, after, getAfter(lhsVal), lhsVal, constrainInterval);
    changed |= applyInterval(emitEq, after, getAfter(rhsVal), rhsVal, constrainInterval);
    changed |= after->addSolverConstraint(constraint);
  } else if (AssertOp assertOp = llvm::dyn_cast<AssertOp>(op)) {
    ensure(operandVals.size() == 1, "assert op with the wrong number of operands");
    // assert enforces that the operand is true. So we apply an interval of [1, 1]
    // to the operand.
    changed |= applyInterval(
        assertOp, after, after, assertOp.getCondition(),
        Interval::Degenerate(field.get(), field.get().one())
    );
    // Also add the solver constraint that the expression must be true.
    auto assertExpr = operandVals[0].getScalarValue();
    changed |= after->addSolverConstraint(assertExpr);
  } else if (auto readf = llvm::dyn_cast<FieldReadOp>(op)) {
    Value cmp = readf.getComponent();
    auto storedVal = getAfter(cmp)->getValue(cmp, readf.getFieldNameAttr().getAttr());
    if (succeeded(storedVal)) {
      // The result value is the value previously written to this field.
      changed |= after->setValue(readf.getVal(), storedVal->getScalarValue());
    } else if (operandRefs[0].has_value()) {
      // Initialize the value
      auto fieldDefRes = readf.getFieldDefOp(tables);
      if (succeeded(fieldDefRes)) {
        SourceRef ref = operandRefs[0]->createChild(SourceRefIndex(*fieldDefRes));
        ExpressionValue exprVal(field.get(), getOrCreateSymbol(ref));
        changed |= after->setValue(readf.getVal(), exprVal);
      }
    }
  } else if (auto writef = llvm::dyn_cast<FieldWriteOp>(op)) {
    // Update values stored in a field
    ExpressionValue writeVal = operandVals[1].getScalarValue();
    auto cmp = writef.getComponent();
    changed |= after->setValue(cmp, writef.getFieldNameAttr().getAttr(), writeVal);
    // We also need to update the interval on the assigned symbol
    SourceRefLatticeValue refSet = getSourceRefLattice(op, cmp)->getOrDefault(cmp);
    if (refSet.isSingleValue()) {
      auto fieldDefRes = writef.getFieldDefOp(tables);
      if (succeeded(fieldDefRes)) {
        SourceRefIndex idx(fieldDefRes.value());
        SourceRef fieldRef = refSet.getSingleValue().createChild(idx);
        llvm::SMTExprRef expr = getOrCreateSymbol(fieldRef);
        changed |= after->setInterval(expr, writeVal.getInterval());
      }
    }
  } else if (isa<IntToFeltOp, FeltToIndexOp>(op)) {
    // Casts don't modify the intervals, but they do modify the SMT types.
    ExpressionValue expr = operandVals[0].getScalarValue();
    // We treat all ints and indexes as felts with the exception of comparison
    // results, which are bools. So if `expr` is a bool, this cast needs to
    // upcast to a felt.
    if (expr.isBoolSort(smtSolver)) {
      expr = boolToFelt(smtSolver, expr, field.get().bitWidth());
    }
    changed |= after->setValue(op->getResult(0), expr);
  } else if (auto yieldOp = dyn_cast<scf::YieldOp>(op)) {
    // Fetch the lattice for after the parent operation so we can propagate
    // the yielded value to subsequent operations.
    Operation *parent = op->getParentOp();
    ensure(parent, "yield operation must have parent operation");
    auto postYieldLattice = getLattice(getProgramPointAfter(parent));
    ensure(postYieldLattice, "could not fetch post-yield lattice");
    // Bind the operand values to the result values of the parent
    for (unsigned idx = 0; idx < yieldOp.getResults().size(); ++idx) {
      Value parentRes = parent->getResult(idx);
      // Merge with the existing value, if present (e.g., another branch)
      // has possible value that must be merged.
      auto exprValRes = postYieldLattice->getValue(parentRes);
      ExpressionValue newResVal = operandVals[idx].getScalarValue();
      if (succeeded(exprValRes)) {
        ExpressionValue existingVal = exprValRes->getScalarValue();
        newResVal =
            existingVal.withInterval(existingVal.getInterval().join(newResVal.getInterval()));
      } else {
        newResVal = ExpressionValue(createFeltSymbol(parentRes), newResVal.getInterval());
      }
      changed |= after->setValue(parentRes, newResVal);
    }

    propagateIfChanged(postYieldLattice, postYieldLattice->join(*after));
  } else if (
      // We do not need to explicitly handle read ops since they are resolved at the operand value
      // step where `SourceRef`s are queries (with the exception of the Signal struct, see above).
      !isReadOp(op)
      // We do not currently handle return ops as the analysis is currently limited to constrain
      // functions, which return no value.
      && !isReturnOp(op)
      // The analysis ignores definition ops.
      && !isDefinitionOp(op)
      // We do not need to analyze the creation of structs.
      && !llvm::isa<CreateStructOp>(op)
  ) {
    op->emitWarning("unhandled operation, analysis may be incomplete").report();
  }

  propagateIfChanged(after, changed);
  return success();
}

llvm::SMTExprRef IntervalDataFlowAnalysis::getOrCreateSymbol(const SourceRef &r) {
  auto it = refSymbols.find(r);
  if (it != refSymbols.end()) {
    return it->second;
  }
  llvm::SMTExprRef sym = createFeltSymbol(r);
  refSymbols[r] = sym;
  return sym;
}

llvm::SMTExprRef IntervalDataFlowAnalysis::createFeltSymbol(const SourceRef &r) const {
  return createFeltSymbol(buildStringViaPrint(r).c_str());
}

llvm::SMTExprRef IntervalDataFlowAnalysis::createFeltSymbol(Value v) const {
  return createFeltSymbol(buildStringViaPrint(v).c_str());
}

llvm::SMTExprRef IntervalDataFlowAnalysis::createFeltSymbol(const char *name) const {
  return field.get().createSymbol(smtSolver, name);
}

llvm::DynamicAPInt IntervalDataFlowAnalysis::getConst(Operation *op) const {
  ensure(isConstOp(op), "op is not a const op");

  llvm::DynamicAPInt fieldConst =
      TypeSwitch<Operation *, llvm::DynamicAPInt>(op)
          .Case<FeltConstantOp>([&](FeltConstantOp feltConst) {
    llvm::APSInt constOpVal(feltConst.getValue());
    return field.get().reduce(constOpVal);
  })
          .Case<arith::ConstantIndexOp>([&](arith::ConstantIndexOp indexConst) {
    return DynamicAPInt(indexConst.value());
  })
          .Case<arith::ConstantIntOp>([&](arith::ConstantIntOp intConst) {
    return DynamicAPInt(intConst.value());
  }).Default([](Operation *illegalOp) {
    std::string err;
    debug::Appender(err) << "unhandled getConst case: " << *illegalOp;
    llvm::report_fatal_error(Twine(err));
    return llvm::DynamicAPInt();
  });
  return fieldConst;
}

ExpressionValue IntervalDataFlowAnalysis::performBinaryArithmetic(
    Operation *op, const LatticeValue &a, const LatticeValue &b
) {
  ensure(isArithmeticOp(op), "is not arithmetic op");

  auto lhs = a.getScalarValue(), rhs = b.getScalarValue();
  ensure(lhs.getExpr(), "cannot perform arithmetic over null lhs smt expr");
  ensure(rhs.getExpr(), "cannot perform arithmetic over null rhs smt expr");

  auto res = TypeSwitch<Operation *, ExpressionValue>(op)
                 .Case<AddFeltOp>([&](auto _) { return add(smtSolver, lhs, rhs); })
                 .Case<SubFeltOp>([&](auto _) { return sub(smtSolver, lhs, rhs); })
                 .Case<MulFeltOp>([&](auto _) { return mul(smtSolver, lhs, rhs); })
                 .Case<DivFeltOp>([&](auto divOp) { return div(smtSolver, divOp, lhs, rhs); })
                 .Case<ModFeltOp>([&](auto _) { return mod(smtSolver, lhs, rhs); })
                 .Case<AndFeltOp>([&](auto _) { return bitAnd(smtSolver, lhs, rhs); })
                 .Case<ShlFeltOp>([&](auto _) { return shiftLeft(smtSolver, lhs, rhs); })
                 .Case<ShrFeltOp>([&](auto _) { return shiftRight(smtSolver, lhs, rhs); })
                 .Case<CmpOp>([&](auto cmpOp) { return cmp(smtSolver, cmpOp, lhs, rhs); })
                 .Case<AndBoolOp>([&](auto _) { return boolAnd(smtSolver, lhs, rhs); })
                 .Case<OrBoolOp>([&](auto _) { return boolOr(smtSolver, lhs, rhs); })
                 .Case<XorBoolOp>([&](auto _) {
    return boolXor(smtSolver, lhs, rhs);
  }).Default([&](auto *unsupported) {
    unsupported
        ->emitWarning(
            "unsupported binary arithmetic operation, defaulting to over-approximated intervals"
        )
        .report();
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
                 .Case<NegFeltOp>([&](auto _) { return neg(smtSolver, val); })
                 .Case<NotFeltOp>([&](auto _) { return notOp(smtSolver, val); })
                 .Case<NotBoolOp>([&](auto _) { return boolNot(smtSolver, val); })
                 // The inverse op is currently overapproximated
                 .Case<InvFeltOp>([&](auto inv) {
    return fallbackUnaryOp(smtSolver, inv, val);
  }).Default([&](auto *unsupported) {
    unsupported
        ->emitWarning(
            "unsupported unary arithmetic operation, defaulting to over-approximated interval"
        )
        .report();
    return fallbackUnaryOp(smtSolver, unsupported, val);
  });

  ensure(res.getExpr(), "arithmetic produced null smt expr");
  return res;
}

ChangeResult IntervalDataFlowAnalysis::applyInterval(
    Operation *originalOp, Lattice *originalLattice, Lattice *after, Value val, Interval newInterval
) {
  auto latValRes = after->getValue(val);
  if (failed(latValRes)) {
    // visitOperation didn't add val to the lattice, so there's nothing to do
    return ChangeResult::NoChange;
  }
  ExpressionValue newLatticeVal = latValRes->getScalarValue().withInterval(newInterval);
  propagateIfChanged(after, after->setValue(val, newLatticeVal));
  ChangeResult res = originalLattice->setValue(val, newLatticeVal);
  // To allow the dataflow analysis to do its fixed-point iteration, we need to
  // add the new expression to val's lattice as well.
  Lattice *valLattice = nullptr;
  if (Operation *valOp = val.getDefiningOp()) {
    valLattice = getLattice(getProgramPointAfter(valOp));
  } else if (auto blockArg = llvm::dyn_cast<BlockArgument>(val)) {
    auto fnOp = dyn_cast<FuncDefOp>(blockArg.getOwner()->getParentOp());
    Operation *blockEntry = &blockArg.getOwner()->front();

    // Apply the interval from the constrain function inputs to the compute function inputs
    if (propagateInputConstraints && fnOp && fnOp.isStructConstrain() &&
        blockArg.getArgNumber() > 0 && !newInterval.isEntire()) {
      auto structOp = fnOp->getParentOfType<StructDefOp>();
      FuncDefOp computeFn = structOp.getComputeFuncOp();
      Operation *computeEntry = &computeFn.getRegion().front().front();
      BlockArgument computeArg = computeFn.getArgument(blockArg.getArgNumber() - 1);
      Lattice *computeEntryLattice = getLattice(getProgramPointBefore(computeEntry));

      SourceRef ref(computeArg);
      ExpressionValue newArgVal(getOrCreateSymbol(ref), newInterval);
      ChangeResult computeRes = computeEntryLattice->setValue(computeArg, newArgVal);
      propagateIfChanged(computeEntryLattice, computeRes);
    }

    valLattice = getLattice(getProgramPointBefore(blockEntry));
  } else {
    valLattice = getLattice(val);
  }

  ensure(valLattice, "val should have a lattice");
  auto setNewVal = [&valLattice, &val, &newLatticeVal, &res, this]() {
    propagateIfChanged(valLattice, valLattice->setValue(val, newLatticeVal));
    return res;
  };

  // Now we descend into val's operands, if it has any.
  Operation *definingOp = val.getDefiningOp();
  if (!definingOp) {
    return setNewVal();
  }
  Lattice *definingOpLattice = getLattice(getProgramPointAfter(definingOp));
  auto getOperandLattice = [&](Value operand) {
    if (Operation *defOp = operand.getDefiningOp()) {
      return getLattice(getProgramPointAfter(defOp));
    } else if (auto blockArg = dyn_cast<BlockArgument>(operand)) {
      Operation *blockEntry = &blockArg.getOwner()->front();
      return getLattice(getProgramPointBefore(blockEntry));
    }
    return definingOpLattice;
  };
  auto getOperandLatticeVal = [&](Value operand) {
    return getOperandLattice(operand)->getValue(operand);
  };

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

    Value lhs = cmpOp.getLhs(), rhs = cmpOp.getRhs();
    auto lhsLatValRes = getOperandLatticeVal(lhs), rhsLatValRes = getOperandLatticeVal(rhs);
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
      cmpOp->emitWarning("unhandled cmp predicate").report();
      return ChangeResult::NoChange;
    }

    // Now we recurse to each operand
    return applyInterval(originalOp, originalLattice, getOperandLattice(lhs), lhs, newLhsInterval) |
           applyInterval(originalOp, originalLattice, getOperandLattice(rhs), rhs, newRhsInterval);
  };

  // If the result of a multiplication is non-zero, then both operands must be
  // non-zero.
  auto mulCase = [&](MulFeltOp mulOp) {
    auto zeroInt = Interval::Degenerate(f, f.zero());
    if (newInterval.intersect(zeroInt).isNotEmpty()) {
      // The multiplication may be zero, so we can't reduce the operands to be non-zero
      return ChangeResult::NoChange;
    }

    Value lhs = mulOp.getLhs(), rhs = mulOp.getRhs();
    auto lhsLatValRes = getOperandLatticeVal(lhs), rhsLatValRes = getOperandLatticeVal(rhs);
    if (failed(lhsLatValRes) || failed(rhsLatValRes)) {
      return ChangeResult::NoChange;
    }
    ExpressionValue lhsExpr = lhsLatValRes->getScalarValue(),
                    rhsExpr = rhsLatValRes->getScalarValue();
    Interval newLhsInterval = lhsExpr.getInterval().difference(zeroInt);
    Interval newRhsInterval = rhsExpr.getInterval().difference(zeroInt);
    return applyInterval(originalOp, originalLattice, getOperandLattice(lhs), lhs, newLhsInterval) |
           applyInterval(originalOp, originalLattice, getOperandLattice(rhs), rhs, newRhsInterval);
  };

  // For casts, just pass the interval along to the cast's operand.
  auto castCase = [&](Operation *op) {
    Value operand = op->getOperand(0);
    return applyInterval(
        originalOp, originalLattice, getOperandLattice(operand), operand, newInterval
    );
  };

  // - Apply the rules given the op.
  // NOTE: disabling clang-format for this because it makes the last case statement
  // look ugly.
  // clang-format off
  res |= TypeSwitch<Operation *, ChangeResult>(definingOp)
            .Case<CmpOp>([&](auto op) { return cmpCase(op); })
            .Case<MulFeltOp>([&](auto op) { return mulCase(op); })
            .Case<IntToFeltOp, FeltToIndexOp>([&](auto op) { return castCase(op); })
            .Default([&](Operation *) { return ChangeResult::NoChange; });
  // clang-format on

  // Set the new val after recursion to avoid having recursive calls unset the value.
  return setNewVal();
}

FailureOr<std::pair<DenseSet<Value>, Interval>>
IntervalDataFlowAnalysis::getGeneralizedDecompInterval(Operation *baseOp, Value lhs, Value rhs) {
  auto isZeroConst = [this](Value v) {
    Operation *op = v.getDefiningOp();
    if (!op) {
      return false;
    }
    if (!isConstOp(op)) {
      return false;
    }
    return getConst(op) == field.get().zero();
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
  std::optional<SourceRef> signalRef = std::nullopt;
  DenseSet<Value> signalVals;
  SmallVector<DynamicAPInt> consts;
  SmallVector<Value> frontier {exprTree};
  while (!frontier.empty()) {
    Value v = frontier.back();
    frontier.pop_back();
    Operation *op = v.getDefiningOp();

    FeltConstantOp c;
    Value signalVal;
    auto handleRefValue = [this, &baseOp, &signalRef, &signalVal, &signalVals]() {
      SourceRefLatticeValue refSet =
          getSourceRefLattice(baseOp, signalVal)->getOrDefault(signalVal);
      if (!refSet.isScalar() || !refSet.isSingleValue()) {
        return failure();
      }
      SourceRef r = refSet.getSingleValue();
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
      auto constInt = APSInt(c.getValue());
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

static void getReversedOps(Region *r, llvm::SmallVector<Operation *> &opList) {
  for (Block &b : llvm::reverse(*r)) {
    for (Operation &op : llvm::reverse(b)) {
      for (Region &nested : llvm::reverse(op.getRegions())) {
        getReversedOps(&nested, opList);
      }
      opList.push_back(&op);
    }
  }
}

LogicalResult StructIntervals::computeIntervals(
    mlir::DataFlowSolver &solver, const IntervalAnalysisContext &ctx
) {

  auto computeIntervalsImpl = [&solver, &ctx, this](
                                  FuncDefOp fn, llvm::MapVector<SourceRef, Interval> &fieldRanges,
                                  llvm::SetVector<ExpressionValue> &solverConstraints
                              ) {
    // Since every lattice value does not contain every value, we will traverse
    // the function backwards (from most up-to-date to least-up-to-date lattices)
    // searching for the source refs. Once a source ref is found, we remove it
    // from the search set.

    SourceRefSet searchSet;
    for (const auto &ref : SourceRef::getAllSourceRefs(structDef, fn)) {
      // We only want to compute intervals for field elements and not composite types.
      if (!ref.isScalar()) {
        continue;
      }
      searchSet.insert(ref);
    }

    // Get all ops in reverse order, including nested ops.
    llvm::SmallVector<Operation *> opList;
    getReversedOps(&fn.getBody(), opList);

    // Also traverse the function op itself
    opList.push_back(fn);

    for (Operation *op : opList) {
      ProgramPoint *pp = solver.getProgramPointAfter(op);
      const IntervalAnalysisLattice *lattice = solver.lookupState<IntervalAnalysisLattice>(pp);
      const auto &c = lattice->getConstraints();
      solverConstraints.insert(c.begin(), c.end());

      SourceRefSet newSearchSet;
      for (const auto &ref : searchSet) {
        auto symbol = ctx.getSymbol(ref);
        auto intervalRes = lattice->findInterval(symbol);
        if (succeeded(intervalRes)) {
          fieldRanges[ref] = *intervalRes;
        } else {
          newSearchSet.insert(ref);
        }
      }
      searchSet = newSearchSet;
    }

    // For all unfound refs, default to the entire range.
    for (const auto &ref : searchSet) {
      fieldRanges[ref] = Interval::Entire(ctx.getField());
    }

    // Sort the outputs since we assembled things out of order.
    llvm::sort(fieldRanges, [](auto a, auto b) { return std::get<0>(a) < std::get<0>(b); });
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
          const char *fnName, const llvm::MapVector<SourceRef, Interval> &fieldRanges,
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
