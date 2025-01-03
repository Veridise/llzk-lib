#include "llzk/Dialect/LLZK/Analysis/ConstraintSummary.h"
#include "llzk/Dialect/LLZK/Util/Hash.h"
#include "llzk/Dialect/LLZK/Util/SymbolHelper.h"

#include <mlir/Analysis/DataFlow/DeadCodeAnalysis.h>
#include <mlir/Analysis/DataFlow/SparseAnalysis.h>
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

/// Tracks the references that operations use and produce.
/// See lattice value requirements here:
/// https://github.com/llvm/llvm-project/blob/77c2b005539c4b0c0e2b7edeefd5f57b95019bc9/mlir/include/mlir/Analysis/DataFlow/SparseAnalysis.h#L83
struct ConstrainRefLatticeValue {
  ConstrainRefLatticeValue() = default;
  explicit ConstrainRefLatticeValue(const ConstrainRefSet &s) : signals(s) {}

  /* Required methods */

  static ConstrainRefLatticeValue
  join(const ConstrainRefLatticeValue &lhs, const ConstrainRefLatticeValue &rhs) {
    ConstrainRefLatticeValue combined;
    combined.signals.insert(lhs.signals.begin(), lhs.signals.end());
    combined.signals.insert(rhs.signals.begin(), rhs.signals.end());
    return combined;
  }

  bool operator==(const ConstrainRefLatticeValue &rhs) const { return signals == rhs.signals; }

  void print(mlir::raw_ostream &os) const {
    os << "{";
    for (auto it = signals.begin(); it != signals.end();) {
      it->print(os);
      it++;
      if (it != signals.end()) {
        os << ", ";
      }
    }
    os << "}";
  }

  /* Update utility methods */

  /// @brief Create child references based on a ref index
  /// @param f
  /// @return
  ConstrainRefLatticeValue createChildren(ConstrainRefIndex r) const {
    ConstrainRefLatticeValue children;
    for (const auto &ref : signals) {
      children.signals.insert(ref.createChild(r));
    }
    return children;
  }

  ConstrainRefLatticeValue createChildren(FieldDefOp f) const {
    return createChildren(ConstrainRefIndex(f));
  }

  ConstrainRefLatticeValue createChildren(mlir::APInt i) const {
    return createChildren(ConstrainRefIndex(i));
  }

  ConstrainRefLatticeValue createChildren(mlir::APInt lower, mlir::APInt upper) const {
    return createChildren(ConstrainRefIndex(lower, upper));
  }

  /* Getter utility methods */

  /// Get the single value of this lattice value if there is a single value,
  /// returning a failure otherwise.
  mlir::FailureOr<ConstrainRef> getSingleValue() const {
    if (signals.size() == 1) {
      return *signals.begin();
    }
    return mlir::failure();
  }

  /// @brief Return true if this contains a single, concrete value.
  /// @return
  bool isConcreteValue() const {
    auto singleValRes = getSingleValue();
    if (mlir::succeeded(singleValRes) && singleValRes->isConstant()) {
      return true;
    }
    return false;
  }

  ConstrainRefSet signals;
};

mlir::raw_ostream &operator<<(mlir::raw_ostream &os, const ConstrainRefLatticeValue &v) {
  v.print(os);
  return os;
}

using ConstrainRefLattice = mlir::dataflow::Lattice<ConstrainRefLatticeValue>;

/// @brief The dataflow analysis that computes the set of references that
/// LLZK operations use and produce. The analysis is simple: any operation will
/// simply output a union of its input references, regardless of what type of
/// operation it performs, as the analysis is operator-insensitive.
class ConstrainRefAnalysis
    : public mlir::dataflow::SparseForwardDataFlowAnalysis<ConstrainRefLattice> {
public:
  using mlir::dataflow::SparseForwardDataFlowAnalysis<
      ConstrainRefLattice>::SparseForwardDataFlowAnalysis;

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

  static std::vector<ConstrainRef> getRefs(mlir::DataFlowSolver &solver, mlir::Value v) {
    std::vector<ConstrainRef> usages;
    // It may also be the case that the operand is just the argument value, which
    // won't have a lattice value.
    auto sourceValRes = getSourceRef(v);
    if (mlir::succeeded(sourceValRes)) {
      usages.emplace_back(sourceValRes.value());
    } else {
      auto analysisRes = solver.lookupState<ConstrainRefLattice>(v);
      if (!analysisRes) {
        llvm::report_fatal_error("untraversed value");
      }

      auto &signals = analysisRes->getValue().signals;
      usages.insert(usages.end(), signals.begin(), signals.end());
    }
    return usages;
  }

  /// @brief Propagate constrain reference lattice values from operands to results.
  /// @param op
  /// @param operands
  /// @param results
  void visitOperation(
      mlir::Operation *op, mlir::ArrayRef<const ConstrainRefLattice *> operands,
      mlir::ArrayRef<ConstrainRefLattice *> results
  ) override {
    // First, see if any of this operations operands are direct references.
    std::vector<ConstrainRefLatticeValue> operandVals;
    for (unsigned i = 0; i < op->getOpOperands().size(); i++) {
      auto refs = operands[i]->getValue().signals;
      auto &operand = op->getOpOperands()[i];

      auto sourceVal = getSourceRef(operand.get());
      if (mlir::succeeded(sourceVal)) {
        refs.emplace(sourceVal.value());
      }
      operandVals.emplace_back(refs);
    }

    // We will now join the the operand refs based on the type of operand.

    auto isJoiningOp = [op]() -> bool {
      return mlir::isa<AddFeltOp>(op) // TODO: we can be fancy here
             || mlir::isa<FeltConstantOp>(op) || mlir::isa<mlir::index::ConstantOp>(op);
    };

    if (auto fieldRead = mlir::dyn_cast<FieldReadOp>(op)) {
      assert(operandVals.size() == 1);
      assert(results.size() == 1);

      auto fieldOpRes = fieldRead.getFieldDefOp(tables);
      if (mlir::failed(fieldOpRes)) {
        llvm::report_fatal_error("could not find field read\n");
      }

      auto &res = results.front();
      const auto &ops = operandVals.front();
      auto childOps = ops.createChildren(fieldOpRes->get());

      propagateIfChanged(res, res->join(childOps));
    } else if (auto arrayRead = mlir::dyn_cast<ReadArrayOp>(op)) {
      assert(results.size() == 1);
      auto &res = results.front();

      auto array = arrayRead.getOperand(0);
      auto currVals = operandVals[0];

      for (size_t i = 1; i < arrayRead.getNumOperands(); i++) {
        auto &idxVals = operandVals[i];

        auto singleIdxVal = idxVals.getSingleValue();
        if (mlir::succeeded(singleIdxVal) && singleIdxVal->isConstantIndex()) {
          currVals = currVals.createChildren(singleIdxVal->getConstantIndexValue());
        } else {
          // Otherwise, assume any range is valid.
          auto arrayType = mlir::dyn_cast<ArrayType>(array.getType());
          auto lower = mlir::APInt::getZero(64);
          mlir::APInt upper(64, arrayType.getDimSize(i - 1));
          currVals = currVals.createChildren(lower, upper);
        }
      }

      propagateIfChanged(res, res->join(currVals));
    } else if (isJoiningOp()) {
      // Standard union of operands, unless all operands are constants,
      // in which case we can easily propagate more precise updates.
      for (auto res : results) {
        mlir::ChangeResult changed = mlir::ChangeResult::NoChange;
        for (auto operand : operandVals) {
          changed |= res->join(operand);
        }
        propagateIfChanged(res, changed);
      }
    } else {
      llvm::errs() << "op is " << *op << "\n";
      llvm::report_fatal_error("todo!");
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

  // Run the analysis. Assumes the liveness analysis has already been performed.
  solver.load<ConstrainRefAnalysis>();
  auto res = solver.initializeAndRun(constrainFnOp);
  if (res.failed()) {
    return mlir::failure();
  }

  /**
   * Now, given the analysis, construct the summary:
   * 1. Union all references based on solver results.
   * 2. Union all references based on nested summaries.
   */

  // 1. Collect all references.
  // Thanks to the self argument, we can just do everything through the
  // blocks arguments to constrain.
  // for (auto a : constrainFnOp.getArguments()) {
  //   for (auto sigUsage : getAllConstrainRefs(a)) {
  //     constraintSets.insert(sigUsage);
  //   }
  // }

  // 1. Union all constraints from the analysis
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
    if (fn.getName() == FUNC_NAME_CONSTRAIN) {
      // Nested
      StructDefOp calledStruct(fn.getOperation()->getParentOp());
      ConstrainRefRemappings translations;
      // Map fn parameters to args in the call op
      for (unsigned i = 0; i < fn.getNumArguments(); i++) {
        auto prefix = ConstrainRef(fn.getArgument(i));
        auto replacements = solver.lookupState<ConstrainRefLattice>(fnCall.getOperand(i));

        if (!replacements) {
          llvm::report_fatal_error("failed to look up replacement translation symbols");
        }

        for (auto &s : replacements->getValue().signals) {
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
    }
  });

  return mlir::success();
}

void ConstraintSummary::walkConstrainOp(mlir::DataFlowSolver &solver, mlir::Operation *emitOp) {
  std::vector<ConstrainRef> usages;
  for (auto operand : emitOp->getOperands()) {
    auto refs = ConstrainRefAnalysis::getRefs(solver, operand);
    usages.insert(usages.end(), refs.begin(), refs.end());
  }

  auto it = usages.begin();
  auto leader = *it;
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
    mlir::DataFlowSolver solver;
    makeLive(solver, modOp);

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

void ConstraintSummaryModuleAnalysis::makeLive(mlir::DataFlowSolver &solver, mlir::Operation *top) {
  for (mlir::Region &region : top->getRegions()) {
    for (mlir::Block &block : region) {
      (void)solver.getOrCreateState<mlir::dataflow::Executable>(&block)->setToLive();
      for (mlir::Operation &oper : block) {
        makeLive(solver, &oper);
      }
    }
  }
}

} // namespace llzk