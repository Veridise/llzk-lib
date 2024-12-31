#include "llzk/Dialect/LLZK/Analysis/ConstraintSummary.h"
#include "llzk/Dialect/LLZK/Util/Hash.h"
#include "llzk/Dialect/LLZK/Util/SymbolHelper.h"

#include <mlir/Analysis/DataFlow/DeadCodeAnalysis.h>
#include <mlir/Analysis/DataFlow/SparseAnalysis.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/Value.h>

#include <unordered_set>

namespace llzk {

/*
Private Utilities:

These classes are defined here and not in the header as they are not designed
for use outside of this specific ConstraintSummary analysis.
*/

/// Tracks the references that operations use. See value requirements:
/// https://github.com/llvm/llvm-project/blob/77c2b005539c4b0c0e2b7edeefd5f57b95019bc9/mlir/include/mlir/Analysis/DataFlow/SparseAnalysis.h#L83
struct ConstrainRefLatticeValue {
  ConstrainRefLatticeValue() = default;

  // required
  static ConstrainRefLatticeValue
  join(const ConstrainRefLatticeValue &lhs, const ConstrainRefLatticeValue &rhs) {
    ConstrainRefLatticeValue combined;
    combined.signals.insert(lhs.signals.begin(), lhs.signals.end());
    combined.signals.insert(rhs.signals.begin(), rhs.signals.end());
    return combined;
  }

  // required
  bool operator==(const ConstrainRefLatticeValue &rhs) const { return signals == rhs.signals; }

  // required
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

  std::unordered_set<ConstrainRef, ConstrainRef::Hash> signals;
};

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

  /// @brief Propagate constrain reference lattice values from operands to results.
  /// @param op
  /// @param operands
  /// @param results
  void visitOperation(
      mlir::Operation *op, mlir::ArrayRef<const ConstrainRefLattice *> operands,
      mlir::ArrayRef<ConstrainRefLattice *> results
  ) override {
    // First, see if any of this operations operands are direct references.
    ConstrainRefLatticeValue operandVals;
    for (auto &operand : op->getOpOperands()) {
      auto res = ConstrainRef::get(operand.get());
      if (mlir::succeeded(res)) {
        operandVals.signals.insert(res->begin(), res->end());
      }
    }

    for (auto *res : results) {
      // add in the values from the operands, if they have are references
      mlir::ChangeResult changed = res->join(operandVals);

      // also merge the lattice values from the operands
      for (const auto *operand : operands) {
        changed |= res->join(*operand);
      }

      // propagate changes to the result lattice value
      propagateIfChanged(res, changed);
    }
  }

protected:
  void setToEntryState(ConstrainRefLattice *lattice) override {
    // the entry state is empty, so do nothing.
  }
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

void ConstraintSummary::print(llvm::raw_ostream &os) const {
  os << "ConstraintSummary {\n";
  // the EquivalenceClasses::iterator is sorted, but the EquivalenceClasses::member_iterator is
  // not guaranteed to be sorted. So, we will sort members before printing them, but otherwise
  // go in leader iterator order.
  for (auto leaderIt = constraintSets.begin(); leaderIt != constraintSets.end(); leaderIt++) {
    if (!leaderIt->isLeader()) {
      continue;
    }

    std::set<ConstrainRef> sortedMembers;
    for (auto mit = constraintSets.member_begin(leaderIt); mit != constraintSets.member_end();
         mit++) {
      sortedMembers.insert(*mit);
    }

    os << "    { ";
    for (auto mit = sortedMembers.begin(); mit != sortedMembers.end();) {
      mit->print(os);
      mit++;
      if (mit != sortedMembers.end()) {
        os << ", ";
      }
    }
    os << " }\n";
  }
  os << "}\n";
}

/// Produce all possible ConstraintRefs that are present starting from the given
/// BlockArgument and partially-specified indices into that object (fields).
std::vector<ConstrainRef> getAllConstrainRefs(
    mlir::ModuleOp mod, StructDefOp s, mlir::BlockArgument blockArg,
    std::vector<FieldDefOp> fields = {}
) {
  std::vector<ConstrainRef> res;
  // Recurse into struct types by iterating over all their field definitions
  for (auto f : s.getOps<FieldDefOp>()) {
    std::vector<FieldDefOp> subFields = fields;
    subFields.push_back(f);
    if (auto structTy = mlir::dyn_cast<llzk::StructType>(f.getType())) {
      mlir::SymbolTableCollection tables;
      auto sDef = structTy.getDefinition(tables, mod);
      if (mlir::failed(sDef)) {
        llvm::report_fatal_error("could not find struct definition from struct type");
      }
      auto subRes = getAllConstrainRefs(mod, sDef->get(), blockArg, subFields);
      res.insert(res.end(), subRes.begin(), subRes.end());
    } else {
      res.push_back(ConstrainRef(blockArg, subFields));
    }
  }
  return res;
}

/// Produce all possible ConstraintRefs that are present starting from the given BlockArgument.
std::vector<ConstrainRef> getAllConstrainRefs(mlir::ModuleOp mod, mlir::BlockArgument arg) {
  auto ty = arg.getType();
  std::vector<ConstrainRef> res;
  if (auto structTy = mlir::dyn_cast<llzk::StructType>(ty)) {
    mlir::SymbolTableCollection tables;
    auto sDef = structTy.getDefinition(tables, mod);
    if (mlir::failed(sDef)) {
      llvm::report_fatal_error("could not find struct definition from struct type");
    }
    res = getAllConstrainRefs(mod, sDef->get(), arg);
  } else if (mlir::isa<llzk::FeltType>(ty) || mlir::isa<mlir::IndexType>(ty) ||
             mlir::isa<llzk::ArrayType>(ty)) {
    // Scalar type
    res.push_back(ConstrainRef(arg));
  } else {
    std::string msg = "unsupported type: ";
    llvm::raw_string_ostream ss(msg);
    ss << ty;
    llvm::report_fatal_error(ss.str().c_str());
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
   * 1. Add all possible references to the constraintSets. This way, unconstrained
   * values will be present as singletons.
   * 2. Union all references based on solver results.
   * 3. Union all references based on nested summaries.
   */

  // 1. Collect all references.
  // Thanks to the self argument, we can just do everything through the
  // blocks arguments to constrain.
  for (auto a : constrainFnOp.getArguments()) {
    for (auto sigUsage : getAllConstrainRefs(mod, a)) {
      constraintSets.insert(sigUsage);
    }
  }

  // 2. Union all constraints from the analysis
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
  mlir::SymbolTableCollection tables;
  constrainFnOp.walk([this, &tables, &am](CallOp fnCall) mutable {
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
        auto prefix = ConstrainRef::get(fn.getArgument(i));
        auto replacements = ConstrainRef::get(fnCall.getOperand(i));
        if (mlir::failed(prefix)) {
          llvm::report_fatal_error("failed to look up prefix translation symbols");
        }
        if (prefix->size() != 1) {
          llvm::report_fatal_error("should only have one prefix symbol!");
        }
        if (mlir::failed(replacements)) {
          llvm::report_fatal_error("failed to look up replacement translation symbols");
        }

        for (auto &s : *replacements) {
          translations.push_back({prefix->at(0), s});
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
          // ensure there's nothing new being added unless it's a constant
          // the member iterator includes the leader, so the leader is checked as well.
          if (constraintSets.findValue(*mit) == constraintSets.end() && !mit->isConstant()) {
            llvm::report_fatal_error("translation returned an unknown value!");
          }
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
    // It may also be the case that the operand is just the argument value.
    auto signalVals = ConstrainRef::get(operand);
    if (mlir::succeeded(signalVals)) {
      usages.insert(usages.end(), signalVals->begin(), signalVals->end());
    } else {
      auto analysisRes = solver.lookupState<ConstrainRefLattice>(operand);
      if (!analysisRes) {
        llvm::report_fatal_error("untraversed value");
      }

      auto &signals = analysisRes->getValue().signals;
      usages.insert(usages.end(), signals.begin(), signals.end());
    }
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