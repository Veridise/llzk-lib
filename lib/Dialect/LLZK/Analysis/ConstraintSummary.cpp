#include "llzk/Dialect/LLZK/Analysis/ConstraintSummary.h"
#include "llzk/Dialect/LLZK/Util/Hash.h"
#include "llzk/Dialect/LLZK/Util/SymbolHelper.h"

#include <mlir/Analysis/DataFlow/DeadCodeAnalysis.h>
#include <mlir/Analysis/DataFlow/SparseAnalysis.h>
#include <mlir/Analysis/DataFlowFramework.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/Value.h>

#include <unordered_set>

namespace llzk {

/* SignalUsage */

mlir::FailureOr<std::vector<SignalUsage>> SignalUsage::get(mlir::Value val) {
  std::vector<SignalUsage> res;

  // If it's a field read, it reads a field def from a component.
  // If it's a felt, it doesn't need a field read

  // Due to the way constrain is defined, all signals are read from inputs.
  if (auto blockArg = mlir::dyn_cast_or_null<mlir::BlockArgument>(val)) {
    // to use this constructor, the block arg must be a felt
    res.push_back(SignalUsage(blockArg));
  } else if (auto fieldRead = mlir::dyn_cast_or_null<FieldReadOp>(val.getDefiningOp())) {
    std::deque<FieldDefOp> fields;
    mlir::SymbolTableCollection tables;
    mlir::BlockArgument arg;
    FieldReadOp currRead = fieldRead;
    while (currRead != nullptr) {
      auto component = currRead.getComponent();
      auto res = currRead.getFieldDefOp(tables);
      if (mlir::failed(res)) {
        fieldRead.emitError() << "could not find field read\n";
        return mlir::failure();
      }
      fields.push_front(res->get());
      arg = mlir::dyn_cast_or_null<mlir::BlockArgument>(component);
      currRead = mlir::dyn_cast_or_null<FieldReadOp>(component.getDefiningOp());
    }
    if (arg == nullptr) {
      fieldRead.emitError() << "could not follow a read chain!\n";
      return mlir::failure();
    }
    // We only want to generate this if the end value is a felt
    res.push_back(SignalUsage(arg, std::vector<FieldDefOp>(fields.begin(), fields.end())));
  } else if (val.getDefiningOp() != nullptr && mlir::isa<FeltConstantOp>(val.getDefiningOp())) {
    auto constFelt = mlir::dyn_cast<FeltConstantOp>(val.getDefiningOp());
    res.push_back(SignalUsage(constFelt));
  } else if (val.getDefiningOp() != nullptr) {
    // Fallback for every other type of operation
    // This also works for global func call ops, since we use an interprocedural dataflow solver
    for (auto operand : val.getDefiningOp()->getOperands()) {
      auto uses = SignalUsage::get(operand);
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

mlir::FailureOr<SignalUsage>
SignalUsage::translate(const SignalUsage &prefix, const SignalUsage &other) const {
  if (blockArg != prefix.blockArg || fieldRefs.size() < prefix.fieldRefs.size()) {
    return mlir::failure();
  }
  for (size_t i = 0; i < prefix.fieldRefs.size(); i++) {
    if (fieldRefs[i] != prefix.fieldRefs[i]) {
      return mlir::failure();
    }
  }
  auto newSignalUsage = other;
  for (size_t i = prefix.fieldRefs.size(); i < fieldRefs.size(); i++) {
    newSignalUsage.fieldRefs.push_back(fieldRefs[i]);
  }
  return newSignalUsage;
}

/* Utilities */

/**
 * Tracks the signals that operations use.
 * See value requirements:
 * https://github.com/llvm/llvm-project/blob/77c2b005539c4b0c0e2b7edeefd5f57b95019bc9/mlir/include/mlir/Analysis/DataFlow/SparseAnalysis.h#L83
 */
struct SignalUsageLatticeValue {
  SignalUsageLatticeValue() = default;

  // required
  static SignalUsageLatticeValue
  join(const SignalUsageLatticeValue &lhs, const SignalUsageLatticeValue &rhs) {
    SignalUsageLatticeValue combined;
    combined.signals.insert(lhs.signals.begin(), lhs.signals.end());
    combined.signals.insert(rhs.signals.begin(), rhs.signals.end());
    return combined;
  }

  // required
  bool operator==(const SignalUsageLatticeValue &rhs) const { return signals == rhs.signals; }

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

  std::unordered_set<SignalUsage, SignalUsage::Hash> signals;
};

using SignalUsageLattice = mlir::dataflow::Lattice<SignalUsageLatticeValue>;

class SignalUsageAnalysis
    : public mlir::dataflow::SparseForwardDataFlowAnalysis<SignalUsageLattice> {
public:
  using mlir::dataflow::SparseForwardDataFlowAnalysis<
      SignalUsageLattice>::SparseForwardDataFlowAnalysis;

  void visitOperation(
      mlir::Operation *op, mlir::ArrayRef<const SignalUsageLattice *> operands,
      mlir::ArrayRef<SignalUsageLattice *> results
  ) override {
    // First, see if any of this operations operands should be converted into a signal value.
    SignalUsageLatticeValue operandVals;
    for (auto &operand : op->getOpOperands()) {
      auto res = SignalUsage::get(operand.get());
      if (mlir::succeeded(res)) {
        operandVals.signals.insert(res->begin(), res->end());
      }
    }

    // This unfortunately does not visit the emit_eq operation, which we need
    // I may need to change emit_eq to be a memory effecting operation
    // Actually, what we do is we do this in two stages. We figure out what
    // signals are in what values, then just iterate over all the constrain operations.
    // and all call instructions.

    for (auto *res : results) {
      // add in the values from the operands
      mlir::ChangeResult changed = res->join(operandVals);

      // res->useDefSubscribe(this);
      // llvm::errs() << "Subscribing result at point " << res->getPoint() << "\n";

      for (const auto *operand : operands) {
        changed |= res->join(*operand);
      }

      propagateIfChanged(res, changed);
    }
  }

protected:
  void setToEntryState(SignalUsageLattice *lattice) override {
    // the entry state is empty, so do nothing.
  }
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
  for (auto leaderIt = constraintSets.begin(); leaderIt != constraintSets.end(); leaderIt++) {
    if (!leaderIt->isLeader()) {
      continue;
    }

    os << "    { ";
    for (auto mit = constraintSets.member_begin(leaderIt); mit != constraintSets.member_end();) {
      mit->print(os);
      mit++;
      if (mit != constraintSets.member_end()) {
        os << ", ";
      }
    }
    os << " }\n";
  }
  os << "}\n";
}

auto getSignalUsages(
    mlir::ModuleOp mod, StructDefOp s, mlir::BlockArgument blockArg,
    std::vector<FieldDefOp> fields = {}
) -> std::vector<SignalUsage> {
  std::vector<SignalUsage> res;
  for (auto f : s.getOps<FieldDefOp>()) {
    std::vector<FieldDefOp> subFields = fields;
    subFields.push_back(f);
    if (auto structTy = mlir::dyn_cast<llzk::StructType>(f.getType())) {
      mlir::SymbolTableCollection tables;
      auto sDef = structTy.getDefinition(tables, mod);
      if (mlir::failed(sDef)) {
        llvm::report_fatal_error("could not find struct definition from struct type");
      }
      auto subRes = getSignalUsages(mod, sDef->get(), blockArg, subFields);
      res.insert(res.end(), subRes.begin(), subRes.end());
    } else {
      res.push_back(SignalUsage(blockArg, subFields));
    }
  }
  return res;
}

auto getSignalUsages(mlir::ModuleOp mod, mlir::BlockArgument arg) -> std::vector<SignalUsage> {
  auto ty = arg.getType();
  std::vector<SignalUsage> res;
  if (auto structTy = mlir::dyn_cast<llzk::StructType>(ty)) {
    mlir::SymbolTableCollection tables;
    auto sDef = structTy.getDefinition(tables, mod);
    if (mlir::failed(sDef)) {
      llvm::report_fatal_error("could not find struct definition from struct type");
    }
    res = getSignalUsages(mod, sDef->get(), arg);
  } else if (mlir::isa<llzk::FeltType>(ty) || mlir::isa<mlir::IndexType>(ty)) {
    // Scalar type
    res.push_back(SignalUsage(arg));
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
  /**
   * The dataflow analysis I think would best be represented as:
   * - Compute the set of fields used for each operation
   * - Query all dataflow operations
   * - Query all constraint operations
   */

  std::vector<FieldDefOp> fields;
  for (auto field : structDef.getOps<FieldDefOp>()) {
    fields.emplace_back(std::move(field));
  }

  auto constrainFnOp = mlir::cast_or_null<FuncOp>(structDef.lookupSymbol("constrain"));

  // Run the analysis
  solver.load<SignalUsageAnalysis>();
  auto res = solver.initializeAndRun(constrainFnOp);
  if (res.failed()) {
    return mlir::failure();
  }

  /**
   * Now, given the analysis, construct the summary
   * 1. Add all signals
   * 2. Union all signals based on solver results
   * 3. Union all signals based on nested summaries
   */

  // Thanks to the self argument, we can just do everything through the
  // arguments to constrain
  for (auto a : constrainFnOp.getArguments()) {
    for (auto sigUsage : getSignalUsages(mod, a)) {
      constraintSets.insert(sigUsage);
    }
  }

  // Union all constraints from the analysis
  // This requires iterating over all of the emit operations
  constrainFnOp.walk([this, &solver](EmitEqualityOp emitOp) {
    std::vector<SignalUsage> usages;
    for (auto operand : emitOp.getOperands()) {
      // It may also be the case that the operand is just the argument value.
      auto signalVals = SignalUsage::get(operand);
      if (mlir::succeeded(signalVals)) {
        usages.insert(usages.end(), signalVals->begin(), signalVals->end());
      } else {
        auto analysisRes = solver.lookupState<SignalUsageLattice>(operand);
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
      /// TODO:
      // should do some scalar filtering in here, cuz currently some signal
      // usages are to structs due to the way we read fields
      // or maybe not, maybe we want to use higher-level objects for ease of use.
      constraintSets.unionSets(leader, *it);
    }
  });

  constrainFnOp.walk([this](EmitContainmentOp emitOp) {
    llvm::report_fatal_error("todo, containment not supported!");
  });

  /**
   * Step two of the analysis is to traverse all of the constrain calls.
   * This is the nested analysis, basically.
   * Constrain functions don't return, so I don't need to compute "values" from
   * the call. I just need to see what constrains are generated here, and
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
      llvm::errs() << "CALLING: " << calledStruct.getName() << " from " << this->structDef.getName()
                   << "\n";
      SignalUsageRemappings translations;
      // Map fn parameters to args in the call op
      for (unsigned i = 0; i < fn.getNumArguments(); i++) {
        llvm::errs() << "arg 0: mapping " << fn.getArgument(i) << " to " << fnCall.getOperand(i)
                     << "\n";
        auto prefix = SignalUsage::get(fn.getArgument(i));
        auto replacements = SignalUsage::get(fnCall.getOperand(i));
        if (mlir::failed(prefix)) {
          llvm::report_fatal_error("failed to look up prefix translation symbols");
        }
        if (prefix->size() != 1) {
          llvm::report_fatal_error("should only have one prefix symbol!");
        }
        if (mlir::failed(replacements)) {
          llvm::report_fatal_error("failed to look up replacement translation symbols");
        }
        // insert this value if not exists
        // constraintSets.insert(*replacement);
        // value should exist?
        // if (constraintSets.findValue(*replacement) == constraintSets.end()) {
        //   llvm::report_fatal_error("failed to add argument value from constrain call");
        // }
        for (auto &s : *replacements) {
          translations.push_back({prefix->at(0), s});
        }
      }
      auto summary = am.getChildAnalysis<ConstraintSummaryAnalysis>(calledStruct).getSummary();
      llvm::errs() << "pre translation summary: ";
      summary.dump();
      auto translatedSummary = summary.translate(translations);
      llvm::errs() << "post translation summary: ";
      translatedSummary.dump();

      // Now, union sets based on the translation
      // We should be able to just merge what is in the translatedSummary to the current summary
      auto &tSets = translatedSummary.constraintSets;
      for (auto lit = tSets.begin(); lit != tSets.end(); lit++) {
        if (!lit->isLeader()) {
          continue;
        }
        auto leader = lit->getData();
        for (auto mit = tSets.member_begin(lit); mit != tSets.member_end(); mit++) {
          // ensure there's nothing new being added
          // the member iterator includes the leader
          if (constraintSets.findValue(*mit) == constraintSets.end()) {
            llvm::report_fatal_error("translation returned an unknown value!");
          }
          constraintSets.unionSets(leader, *mit);
        }
      }
    }
  });

  dump();

  return mlir::success();
}

ConstraintSummary ConstraintSummary::translate(SignalUsageRemappings translation) {
  ConstraintSummary res(mod, structDef);
  auto translate = [&translation](const SignalUsage &elem
                   ) -> mlir::FailureOr<std::vector<SignalUsage>> {
    std::vector<SignalUsage> res;
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
    std::vector<SignalUsage> translated;
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

/* ConstraintSummaryAnalysis */

ConstraintSummaryAnalysis::ConstraintSummaryAnalysis(mlir::Operation *op) {
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

mlir::LogicalResult ConstraintSummaryAnalysis::constructSummary(
    mlir::DataFlowSolver &solver, mlir::AnalysisManager &moduleAnalysisManager
) {
  auto summaryRes = ConstraintSummary::compute(modOp, structDefOp, solver, moduleAnalysisManager);
  if (mlir::failed(summaryRes)) {
    return mlir::failure();
  }
  summary = std::make_shared<ConstraintSummary>(*summaryRes);
  return mlir::success();
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