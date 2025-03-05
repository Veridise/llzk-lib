#include "llzk/Dialect/LLZK/Analysis/IntervalAnalysis.h"

namespace llzk {

/* Field */

Field::Field(std::string_view primeStr) : primeMod(llvm::APSInt(primeStr)) {
  halfPrime = (primeMod + felt(1)) / felt(2);
}

const Field &Field::getField(const char *fieldName) {
  static llvm::DenseMap<llvm::StringRef, Field> knownFields;
  static std::once_flag fieldsInit;
  std::call_once(fieldsInit, initKnownFields, knownFields);

  if (auto it = knownFields.find(fieldName); it != knownFields.end()) {
    return it->second;
  }
  llvm::report_fatal_error("field \"" + mlir::Twine(fieldName) + "\" is unsupported");
}

void Field::initKnownFields(llvm::DenseMap<llvm::StringRef, Field> &knownFields) {
  // bn128/254, default for circom
  knownFields.try_emplace("bn128", Field("21888242871839275222246405745257275088696311157297823662689037894645226208583"));
  knownFields.try_emplace("bn254", knownFields.at("bn128"));
  // 15 * 2^27 + 1, default for zirgen
  knownFields.try_emplace("babybear", Field("2013265921"));
  // 2^64 - 2^32 + 1, used for plonky2
  knownFields.try_emplace("goldilocks", Field("18446744069414584321"));
  // 2^31 - 1, used for Plonky3
  knownFields.try_emplace("mersenne31", Field("2147483647"));
}

llvm::APSInt Field::reduce(llvm::APSInt i) const {
  auto maxBits = std::max(i.getBitWidth(), bitWidth());
  auto m = (i.zext(maxBits).urem(prime().zext(maxBits))).trunc(bitWidth());
  if (m.isNegative()) {
    return prime() + llvm::APSInt(m);
  }
  return llvm::APSInt(m);
}

llvm::APSInt Field::reduce(unsigned i) const {
  auto ap = llvm::APSInt(llvm::APInt(bitWidth(), i));
  return reduce(ap);
}

/* UnreducedInterval */

Interval UnreducedInterval::reduce(const Field &field) const {
  if (a > b) {
    return Interval::Empty(field);
  }
  if (width().trunc(field.bitWidth()) >= field.prime()) {
    return Interval::Entire(field);
  }

  auto lhs = field.reduce(a), rhs = field.reduce(b);

  if ((rhs - lhs).isZero()) {
    return Interval::Degenerate(field, lhs);
  }

  const auto &half = field.half();
  if (lhs.ule(rhs)) {
    if (lhs.ult(half) && rhs.ult(half)) {
      return Interval::TypeA(field, lhs, rhs);
    } else if (lhs.ult(half)) {
      return Interval::TypeC(field, lhs, rhs);
    } else {
      return Interval::TypeB(field, lhs, rhs);
    }
  } else {
    if (lhs.uge(half) && rhs.ult(half)) {
      return Interval::TypeF(field, lhs, rhs);
    } else {
      return Interval::Entire(field);
    }
  }
}

UnreducedInterval UnreducedInterval::intersect(const UnreducedInterval &rhs) const {
  auto &lhs = *this;
  return UnreducedInterval(std::max(lhs.a, rhs.a), std::min(lhs.b, rhs.b));
}

UnreducedInterval UnreducedInterval::doUnion(const UnreducedInterval &rhs) const {
  auto &lhs = *this;
  return UnreducedInterval(std::min(lhs.a, rhs.a), std::max(lhs.b, rhs.b));
}

UnreducedInterval UnreducedInterval::lt(const UnreducedInterval &rhs) const {
  auto one = llvm::APSInt(llvm::APInt(a.getBitWidth(), 1));
  auto bound = rhs.b - one;
  return UnreducedInterval(std::min(a, bound), std::min(b, bound));
}

UnreducedInterval UnreducedInterval::le(const UnreducedInterval &rhs) const {
  return UnreducedInterval(std::min(a, rhs.b), std::min(b, rhs.b));
}

UnreducedInterval UnreducedInterval::gt(const UnreducedInterval &rhs) const {
  auto one = llvm::APSInt(llvm::APInt(a.getBitWidth(), 1));
  auto bound = rhs.a + one;
  return UnreducedInterval(std::max(a, bound), std::max(b, bound));
}

UnreducedInterval UnreducedInterval::ge(const UnreducedInterval &rhs) const {
  return UnreducedInterval(std::max(a, rhs.a), std::max(b, rhs.a));
}

UnreducedInterval UnreducedInterval::operator-() const { return UnreducedInterval(-b, -a); }

UnreducedInterval operator+(const UnreducedInterval &lhs, const UnreducedInterval &rhs) {
  return UnreducedInterval(lhs.a + rhs.a, lhs.b + rhs.b);
}

UnreducedInterval operator-(const UnreducedInterval &lhs, const UnreducedInterval &rhs) {
  return lhs + (-rhs);
}

UnreducedInterval operator*(const UnreducedInterval &lhs, const UnreducedInterval &rhs) {
  auto v1 = lhs.a * rhs.a;
  auto v2 = lhs.a * rhs.b;
  auto v3 = lhs.b * rhs.a;
  auto v4 = lhs.b * rhs.b;

  auto minVal = std::min({v1, v2, v3, v4});
  auto maxVal = std::max({v1, v2, v3, v4});

  return UnreducedInterval(minVal, maxVal);
}

bool UnreducedInterval::overlaps(const UnreducedInterval &rhs) const {
  auto &lhs = *this;
  return lhs.b >= rhs.a || lhs.a <= rhs.b;
}

std::strong_ordering operator<=>(const UnreducedInterval &lhs, const UnreducedInterval &rhs) {
  if (lhs.a < rhs.a || (lhs.a == rhs.a && lhs.b < rhs.b)) {
    return std::strong_ordering::less;
  }
  if (lhs.a > rhs.a || (lhs.a == rhs.a && lhs.b > rhs.b)) {
    return std::strong_ordering::greater;
  }
  return std::strong_ordering::equal;
}

/* Interval */

UnreducedInterval Interval::toUnreduced() const {
  if (isEmpty()) {
    return UnreducedInterval(field.get().zero(), field.get().zero());
  }
  if (isEntire()) {
    return UnreducedInterval(field.get().zero(), field.get().maxVal());
  }
  return UnreducedInterval(a, b);
}

UnreducedInterval Interval::firstUnreduced() const {
  if (isOneOf<Type::TypeF>()) {
    return UnreducedInterval(field.get().prime() - a, b);
  }
  return toUnreduced();
}

UnreducedInterval Interval::secondUnreduced() const {
  ensure(isOneOf<Type::TypeA, Type::TypeB, Type::TypeC>(), "unsupported range type");
  return UnreducedInterval(a - field.get().prime(), b - field.get().prime());
}

Interval Interval::join(const Interval &rhs) const {
  auto &lhs = *this;

  // Trivial cases
  if (lhs.isEntire() || rhs.isEntire()) {
    return Interval::Empty(field.get());
  }
  if (lhs.isEmpty()) {
    return rhs;
  }
  if (rhs.isEmpty()) {
    return rhs;
  }
  if (lhs.isDegenerate() || rhs.isDegenerate()) {
    return lhs.toUnreduced().doUnion(rhs.toUnreduced()).reduce(field.get());
  }

  // More complex cases
  if (areOneOf<
          {Type::TypeA, Type::TypeA}, {Type::TypeB, Type::TypeB}, {Type::TypeC, Type::TypeC},
          {Type::TypeA, Type::TypeC}, {Type::TypeB, Type::TypeC}>(lhs, rhs)) {
    return Interval(rhs.ty, field.get(), std::min(lhs.a, rhs.a), std::max(lhs.b, rhs.b));
  }
  if (areOneOf<{Type::TypeA, Type::TypeB}>(lhs, rhs)) {
    auto lhsUnred = lhs.firstUnreduced();
    auto opt1 = rhs.firstUnreduced().doUnion(lhsUnred);
    auto opt2 = rhs.secondUnreduced().doUnion(lhsUnred);
    if (opt1.width() <= opt2.width()) {
      return opt1.reduce(field.get());
    }
    return opt2.reduce(field.get());
  }
  if (areOneOf<{Type::TypeF, Type::TypeF}, {Type::TypeA, Type::TypeF}>(lhs, rhs)) {
    return lhs.firstUnreduced().doUnion(rhs.firstUnreduced()).reduce(field.get());
  }
  if (areOneOf<{Type::TypeB, Type::TypeF}>(lhs, rhs)) {
    return lhs.secondUnreduced().doUnion(rhs.firstUnreduced()).reduce(field.get());
  }
  if (areOneOf<{Type::TypeC, Type::TypeF}>(lhs, rhs)) {
    return Interval::Entire(field.get());
  }
  if (areOneOf<
          {Type::TypeB, Type::TypeA}, {Type::TypeC, Type::TypeA}, {Type::TypeC, Type::TypeB},
          {Type::TypeF, Type::TypeA}, {Type::TypeF, Type::TypeB}, {Type::TypeF, Type::TypeC}>(
          lhs, rhs
      )) {
    return rhs.join(lhs);
  }
  llvm::report_fatal_error("unhandled join case");
  return Interval::Entire(field.get());
}

Interval Interval::intersect(const Interval &rhs) const {
  auto &lhs = *this;

  // Trivial cases
  if (lhs.isEmpty() || rhs.isEmpty()) {
    return Interval::Empty(field.get());
  }
  if (lhs.isEntire()) {
    return rhs;
  }
  if (rhs.isEntire()) {
    return lhs;
  }
  if (lhs.isDegenerate() || rhs.isDegenerate()) {
    return lhs.toUnreduced().intersect(rhs.toUnreduced()).reduce(field.get());
  }

  // More complex cases
  if (areOneOf<
          {Type::TypeA, Type::TypeA}, {Type::TypeB, Type::TypeB}, {Type::TypeC, Type::TypeC},
          {Type::TypeA, Type::TypeC}, {Type::TypeB, Type::TypeC}>(lhs, rhs)) {
    auto maxA = std::max(lhs.a, rhs.a);
    auto minB = std::min(lhs.b, rhs.b);
    if (maxA <= minB) {
      return Interval(lhs.ty, field.get(), maxA, minB);
    } else {
      return Interval::Empty(field.get());
    }
  }
  if (areOneOf<{Type::TypeA, Type::TypeB}>(lhs, rhs)) {
    return Interval::Empty(field.get());
  }
  if (areOneOf<{Type::TypeF, Type::TypeF}, {Type::TypeA, Type::TypeF}>(lhs, rhs)) {
    return lhs.firstUnreduced().intersect(rhs.firstUnreduced()).reduce(field.get());
  }
  if (areOneOf<{Type::TypeB, Type::TypeF}>(lhs, rhs)) {
    return lhs.secondUnreduced().intersect(rhs.firstUnreduced()).reduce(field.get());
  }
  if (areOneOf<{Type::TypeC, Type::TypeF}>(lhs, rhs)) {
    auto rhsUnred = rhs.firstUnreduced();
    auto opt1 = lhs.firstUnreduced().intersect(rhsUnred).reduce(field.get());
    auto opt2 = lhs.secondUnreduced().intersect(rhsUnred).reduce(field.get());
    ensure(!opt1.isEntire() && !opt2.isEntire(), "impossible intersection");
    if (opt1.isEmpty()) {
      return opt2;
    }
    if (opt2.isEmpty()) {
      return opt1;
    }
    return opt1.join(opt2);
  }
  if (areOneOf<
          {Type::TypeB, Type::TypeA}, {Type::TypeC, Type::TypeA}, {Type::TypeC, Type::TypeB},
          {Type::TypeF, Type::TypeA}, {Type::TypeF, Type::TypeB}, {Type::TypeF, Type::TypeC}>(
          lhs, rhs
      )) {
    return rhs.intersect(lhs);
  }
  return Interval::Empty(field.get());
}

Interval Interval::operator-() const { return (-firstUnreduced()).reduce(field.get()); }

Interval operator+(const Interval &lhs, const Interval &rhs) {
  ensure(lhs.field.get() == rhs.field.get(), "cannot add intervals in different fields");
  return (lhs.firstUnreduced() + rhs.firstUnreduced()).reduce(lhs.field.get());
}

Interval operator-(const Interval &lhs, const Interval &rhs) { return lhs + (-rhs); }

Interval operator*(const Interval &lhs, const Interval &rhs) {
  ensure(lhs.field.get() == rhs.field.get(), "cannot multiply intervals in different fields");
  const auto &field = lhs.field.get();
  auto zeroInterval = Interval::Degenerate(field, field.zero());
  if (lhs == zeroInterval || rhs == zeroInterval) {
    return zeroInterval;
  }
  if (lhs.isEmpty() || rhs.isEmpty()) {
    return Interval::Empty(field);
  }
  if (lhs.isEntire() || rhs.isEntire()) {
    return Interval::Entire(field);
  }

  if (Interval::areOneOf<{Interval::Type::TypeB, Interval::Type::TypeB}>(lhs, rhs)) {
    return (lhs.secondUnreduced() * rhs.secondUnreduced()).reduce(field);
  }
  return (lhs.firstUnreduced() * rhs.firstUnreduced()).reduce(field);
}

Interval operator/(const Interval &lhs, const Interval &rhs) {
  const auto &field = rhs.getField();
  if (rhs.width() > field.one()) {
    return Interval::Entire(field);
  }
  if (rhs.a.isZero()) {
    return lhs;
  }
  return UnreducedInterval(lhs.a / rhs.a, lhs.b / rhs.a).reduce(field);
}

Interval operator%(const Interval &lhs, const Interval &rhs) {
  const auto &field = rhs.getField();
  return UnreducedInterval(field.zero(), rhs.b).reduce(field);
}

void Interval::print(mlir::raw_ostream &os) const {
  os << TypeName(ty);
  if (isOneOf<Type::Degenerate>()) {
    os << '(' << a << ')';
  } else if (!isOneOf<Type::Entire, Type::Empty>()) {
    os << ":[ " << a << ", " << b << " ]";
  }
}

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
div(llvm::SMTSolverRef solver, const ExpressionValue &lhs, const ExpressionValue &rhs) {
  ExpressionValue res;
  res.i = lhs.i / rhs.i;
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

ExpressionValue fallbackBinaryOp(
    llvm::SMTSolverRef solver, mlir::Operation *op, const ExpressionValue &lhs,
    const ExpressionValue &rhs
) {
  ExpressionValue res;
  res.i = Interval::Entire(lhs.getField());
  if (mlir::isa<AndFeltOp>(op)) {
    res.expr = solver->mkBVAnd(lhs.expr, rhs.expr);
  } else if (mlir::isa<OrFeltOp>(op)) {
    res.expr = solver->mkBVOr(lhs.expr, rhs.expr);
  } else if (mlir::isa<XorFeltOp>(op)) {
    res.expr = solver->mkBVXor(lhs.expr, lhs.expr);
  } else if (mlir::isa<ShlFeltOp>(op)) {
    res.expr = solver->mkBVShl(lhs.expr, rhs.expr);
  } else if (mlir::isa<ShrFeltOp>(op)) {
    res.expr = solver->mkBVLshr(lhs.expr, rhs.expr);
  } else {
    llvm::report_fatal_error(
        "no fallback provided for " + mlir::Twine(op->getName().getStringRef())
    );
  }
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
  const auto &f = val.getField();
  if (val.i.isDegenerate()) {
    if (val.i == Interval::Degenerate(f, f.zero())) {
      res.i = Interval::Degenerate(f, f.one());
    } else {
      res.i = Interval::Degenerate(f, f.zero());
    }
  }
  res.i = Interval::Boolean(f);
  res.expr = solver->mkBVNot(val.expr);
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

mlir::ChangeResult IntervalAnalysisLattice::join(const AbstractDenseLattice &other) {
  const auto *rhs = dynamic_cast<const IntervalAnalysisLattice *>(&other);
  if (!rhs) {
    llvm::report_fatal_error("invalid join lattice type");
  }
  mlir::ChangeResult res = mlir::ChangeResult::NoChange;
  for (auto &[k, v] : rhs->valMap) {
    auto it = valMap.find(k);
    if (it == valMap.end() || it->second != v) {
      valMap[k] = v;
      res |= mlir::ChangeResult::Change;
    }
  }
  for (auto &v : rhs->constraints) {
    if (!constraints.contains(v)) {
      constraints.insert(v);
      res |= mlir::ChangeResult::Change;
    }
  }
  for (auto &[e, i] : rhs->intervals) {
    auto it = intervals.find(e);
    if (it == intervals.end() || it->second != i) {
      intervals[e] = i;
      res |= mlir::ChangeResult::Change;
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
    expr->print(os);
    os << " in " << interval;
  }
  if (!valMap.empty()) {
    os << '\n';
  }
  os << '}';
}

mlir::FailureOr<IntervalAnalysisLattice::LatticeValue>
IntervalAnalysisLattice::getValue(mlir::Value v) const {
  auto it = valMap.find(v);
  if (it == valMap.end()) {
    return mlir::failure();
  }
  return it->second;
}

mlir::ChangeResult IntervalAnalysisLattice::setValue(mlir::Value v, ExpressionValue e) {
  LatticeValue val(e);
  if (valMap[v] == val) {
    return mlir::ChangeResult::NoChange;
  }
  valMap[v] = val;
  intervals[e.getExpr()] = e.getInterval();
  return mlir::ChangeResult::Change;
}

mlir::ChangeResult IntervalAnalysisLattice::addSolverConstraint(ExpressionValue e) {
  if (!constraints.contains(e)) {
    constraints.insert(e);
    return mlir::ChangeResult::Change;
  }
  return mlir::ChangeResult::NoChange;
}

mlir::FailureOr<Interval> IntervalAnalysisLattice::findInterval(llvm::SMTExprRef expr) const {
  auto it = intervals.find(expr);
  if (it != intervals.end()) {
    return it->second;
  }
  return mlir::failure();
}

/* IntervalDataFlowAnalysis */

void IntervalDataFlowAnalysis::visitOperation(
    mlir::Operation *op, const Lattice &before, Lattice *after
) {
  mlir::ChangeResult changed = after->join(before);

  llvm::SmallVector<LatticeValue> operandVals;

  auto constrainRefLattice = dataflowSolver.lookupState<ConstrainRefLattice>(op);
  ensure(constrainRefLattice, "failed to get lattice");

  for (auto &operand : op->getOpOperands()) {
    auto val = operand.get();
    // First, lookup the operand value in the before state.
    auto priorState = before.getValue(val);
    if (mlir::succeeded(priorState)) {
      operandVals.push_back(*priorState);
      continue;
    }
    // Else, look up the stored value by constrain ref.
    // We only care about scalar type values, which is currently limited to:
    // felt, index, etc.
    if (!mlir::isa<FeltType>(val.getType())) {
      operandVals.push_back(LatticeValue());
      continue;
    }

    auto refSet = constrainRefLattice->getOrDefault(val);
    if (!refSet.isSingleValue()) {
      std::string valStr;
      llvm::raw_string_ostream ss(valStr);
      val.print(ss);

      op->emitWarning(
          "operand " + mlir::Twine(valStr) + " is not a single value, overapproximating"
      );
      operandVals.push_back(LatticeValue());
    } else {
      auto ref = refSet.getSingleValue();
      auto exprVal = ExpressionValue(field.get(), getOrCreateSymbol(ref));
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
    auto latticeVal = ExpressionValue(field.get(), expr, constVal);
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
  } else if (mlir::isa<EmitEqualityOp>(op)) {
    ensure(operandVals.size() == 2, "constraint op with the wrong number of operands");
    auto lhsVal = op->getOperand(0);
    auto rhsVal = op->getOperand(1);
    auto lhsExpr = operandVals[0].getScalarValue();
    auto rhsExpr = operandVals[1].getScalarValue();

    auto constraint = intersection(smtSolver, lhsExpr, rhsExpr);
    // Update the LHS and RHS to the same value, but restricted intervals
    // based on the constraints
    changed |= after->setValue(lhsVal, lhsExpr.withInterval(constraint.getInterval()));
    changed |= after->setValue(rhsVal, rhsExpr.withInterval(constraint.getInterval()));
    changed |= after->addSolverConstraint(constraint);

  } else if (isAssertOp(op)) {
    ensure(operandVals.size() == 1, "assert op with the wrong number of operands");
    // First, just add the solver constraint that the expression must be true.
    auto assertExpr = operandVals[0].getScalarValue();
    changed |= after->addSolverConstraint(assertExpr);
    // Then, we want to do a lookback at the boolean expression that composes
    // the assert expression and restrict the range on those values to be
    // ranges that make the constraint hold.

    // TODO: this currently only works for simple comparison cases and not
    // conjunctions/disjunctions of comparisons.
    if (auto cmpOp = mlir::dyn_cast<CmpOp>(op->getOperand(0).getDefiningOp())) {
      auto lhs = cmpOp->getOperand(0);
      auto rhs = cmpOp->getOperand(1);
      auto lhsLatticeVal = before.getValue(lhs);
      auto rhsLatticeVal = before.getValue(rhs);
      ensure(
          mlir::succeeded(lhsLatticeVal) && mlir::succeeded(rhsLatticeVal),
          "no values for assert predecessors"
      );
      auto lhsExpr = lhsLatticeVal->getScalarValue();
      auto rhsExpr = rhsLatticeVal->getScalarValue();

      Interval newLhsInterval, newRhsInterval;
      switch (cmpOp.getPredicate()) {
      case FeltCmpPredicate::EQ: {
        newLhsInterval = newRhsInterval = lhsExpr.getInterval().intersect(rhsExpr.getInterval());
        break;
      }
      case FeltCmpPredicate::NE: {
        newLhsInterval = newRhsInterval = lhsExpr.getInterval().join(rhsExpr.getInterval());
        break;
      }
      case FeltCmpPredicate::LT: {
        newLhsInterval = lhsExpr.getInterval()
                             .toUnreduced()
                             .lt(rhsExpr.getInterval().toUnreduced())
                             .reduce(field.get());
        newRhsInterval = rhsExpr.getInterval()
                             .toUnreduced()
                             .ge(lhsExpr.getInterval().toUnreduced())
                             .reduce(field.get());
        break;
      }
      case FeltCmpPredicate::LE: {
        newLhsInterval = lhsExpr.getInterval()
                             .toUnreduced()
                             .le(rhsExpr.getInterval().toUnreduced())
                             .reduce(field.get());
        newRhsInterval = rhsExpr.getInterval()
                             .toUnreduced()
                             .gt(lhsExpr.getInterval().toUnreduced())
                             .reduce(field.get());
        break;
      }
      case FeltCmpPredicate::GT: {
        newLhsInterval = lhsExpr.getInterval()
                             .toUnreduced()
                             .gt(rhsExpr.getInterval().toUnreduced())
                             .reduce(field.get());
        newRhsInterval = rhsExpr.getInterval()
                             .toUnreduced()
                             .le(lhsExpr.getInterval().toUnreduced())
                             .reduce(field.get());
        break;
      }
      case FeltCmpPredicate::GE: {
        newLhsInterval = lhsExpr.getInterval()
                             .toUnreduced()
                             .ge(rhsExpr.getInterval().toUnreduced())
                             .reduce(field.get());
        newRhsInterval = rhsExpr.getInterval()
                             .toUnreduced()
                             .lt(lhsExpr.getInterval().toUnreduced())
                             .reduce(field.get());
        break;
      }
      }

      changed |= after->setValue(lhs, lhsExpr.withInterval(newLhsInterval));
      changed |= after->setValue(rhs, rhsExpr.withInterval(newRhsInterval));
    }
  } else if (!isReadOp(op)          /* We do not need to explicitly handle read ops
                      since they are resolved at the operand value step where constrain refs are
                      queries */
             && !isReturnOp(op)     /* We do not currently handle return ops as the analysis
                 is currently limited to constrain functions, which return no value. */
             && !isDefinitionOp(op) /* The analysis ignores field, struct, function definitions. */
             &&
             !mlir::isa<CreateStructOp>(op) /* We do not need to analyze the creation of structs. */
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
  std::string symbolName;
  llvm::raw_string_ostream ss(symbolName);
  r.print(ss);

  return createFeltSymbol(symbolName.c_str());
}

llvm::SMTExprRef IntervalDataFlowAnalysis::createFeltSymbol(mlir::Value val) const {
  std::string symbolName;
  llvm::raw_string_ostream ss(symbolName);
  val.print(ss);

  return createFeltSymbol(symbolName.c_str());
}

llvm::SMTExprRef IntervalDataFlowAnalysis::createFeltSymbol(const char *name) const {
  return smtSolver->mkSymbol(name, smtSolver->getBitvectorSort(field.get().bitWidth()));
}

llvm::APSInt IntervalDataFlowAnalysis::getConst(mlir::Operation *op) const {
  ensure(isConstOp(op), "op is not a const op");
  auto fieldConst =
      mlir::dyn_cast<FeltConstantOp>(op).getValueAttr().getValue().zext(field.get().bitWidth());
  return llvm::APSInt(fieldConst);
}

ExpressionValue IntervalDataFlowAnalysis::performBinaryArithmetic(
    mlir::Operation *op, const LatticeValue &a, const LatticeValue &b
) {
  ensure(isArithmeticOp(op), "is not arithmetic op");

  auto lhs = a.getScalarValue(), rhs = b.getScalarValue();

  if (mlir::isa<AddFeltOp>(op)) {
    return add(smtSolver, lhs, rhs);
  } else if (mlir::isa<SubFeltOp>(op)) {
    return sub(smtSolver, lhs, rhs);
  } else if (mlir::isa<MulFeltOp>(op)) {
    return mul(smtSolver, lhs, rhs);
  } else if (mlir::isa<DivFeltOp>(op)) {
    return div(smtSolver, lhs, rhs);
  } else if (mlir::isa<ModFeltOp>(op)) {
    return mod(smtSolver, lhs, rhs);
  } else if (auto cmpOp = mlir::dyn_cast<CmpOp>(op)) {
    return cmp(smtSolver, cmpOp, lhs, rhs);
  } else {
    op->emitWarning(
        "unsupported binary arithmetic operation, defaulting to over-approximated intervals"
    );
    return fallbackBinaryOp(smtSolver, op, lhs, rhs);
  }
}

ExpressionValue
IntervalDataFlowAnalysis::performUnaryArithmetic(mlir::Operation *op, const LatticeValue &a) {
  ensure(isArithmeticOp(op), "is not arithmetic op");

  auto val = a.getScalarValue();

  if (mlir::isa<NegFeltOp>(op)) {
    return neg(smtSolver, val);
  } else if (mlir::isa<NotFeltOp>(op)) {
    return notOp(smtSolver, val);
  } else {
    llvm::report_fatal_error(
        "unsupported unary arithmetic operation " + mlir::Twine(op->getName().getStringRef())
    );
    return ExpressionValue();
  }
}

/* StructIntervals */

mlir::LogicalResult StructIntervals::computeIntervals(
    mlir::DataFlowSolver &solver, mlir::AnalysisManager &am, IntervalAnalysisContext &ctx
) {
  // Get the lattice at the end of the constrain function.
  ReturnOp constrainEnd;
  structDef.getConstrainFuncOp().walk([&constrainEnd](ReturnOp r) mutable { constrainEnd = r; });

  auto constrainLattice = solver.lookupState<IntervalAnalysisLattice>(constrainEnd);

  constrainSolverConstraints = constrainLattice->getConstraints();

  for (const auto &ref : ConstrainRef::getAllConstrainRefs(structDef)) {
    // Don't compute ranges for structs, that doesn't make any sense.
    if (!ref.isScalar()) {
      continue;
    }
    auto symbol = ctx.getSymbol(ref);
    auto constrainInterval = constrainLattice->findInterval(symbol);
    if (mlir::succeeded(constrainInterval)) {
      constrainFieldRanges[ref] = *constrainInterval;
    } else {
      constrainFieldRanges[ref] = Interval::Entire(ctx.field);
    }
  }

  return mlir::success();
}

void StructIntervals::print(mlir::raw_ostream &os, bool withConstraints) const {
  os << "StructIntervals { ";
  if (constrainFieldRanges.empty()) {
    os << "}\n";
    return;
  }

  for (auto &[ref, interval] : constrainFieldRanges) {
    os << "\n    " << ref << " in " << interval;
  }

  if (withConstraints) {
    os << "\n\n    Solver Constraints { ";
    if (constrainSolverConstraints.empty()) {
      os << "}\n";
    } else {
      for (const auto &e : constrainSolverConstraints) {
        os << "\n        ";
        e.getExpr()->print(os);
      }
      os << "\n    }";
    }
  }

  os << "\n}\n";
}

} // namespace llzk
