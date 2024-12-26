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

/* Utilities */

/// @brief Defines a signal usage.
/// A signal usage is:
/// - The block argument index (all signals, even internal, are referenced via inputs arguments)
///   - "self", or internal signals, will always be block argument 0.
/// - The field definitions within the argument, if present. This would be if an input argument is
/// another struct.
///   - Like array references, this may be nested (e.g., signal foo of struct X within struct Y
///   would be Y[X[foo]]).
class SignalUsage {
public:
  /// Try to create a SignalUsage out of a given operation.
  static mlir::FailureOr<SignalUsage> get(mlir::Value val) {
    // If it's a field read, it reads a field def from a component.
    // If it's a felt, it doesn't need a field read

    // Due to the way constrain is defined, all signals are read from inputs.
    if (auto blockArg = mlir::dyn_cast_or_null<mlir::BlockArgument>(val)) {
      // to use this constructor, the block arg must be a felt
      return SignalUsage(blockArg.getArgNumber());
    }

    if (auto fieldRead = mlir::dyn_cast_or_null<FieldReadOp>(val.getDefiningOp())) {
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
      return SignalUsage(arg.getArgNumber(), std::vector<FieldDefOp>(fields.begin(), fields.end()));
    }

    return mlir::failure();
  }

  SignalUsage(unsigned b) : blockArgIdx(b), fieldRefs({}) {}
  SignalUsage(unsigned b, std::vector<FieldDefOp> f) : blockArgIdx(b), fieldRefs(f) {}

  unsigned getInputNum() const { return blockArgIdx; }

  void print(mlir::raw_ostream &os) const {
    os << "<input: " << getInputNum();
    for (auto f : fieldRefs) {
      os << ", field: " << f;
    }
    os << ">";
    // TODO: the rest
  }

  bool operator==(const SignalUsage &rhs) const {
    return blockArgIdx == rhs.blockArgIdx && fieldRefs == rhs.fieldRefs;
  }

  struct Hash {
    size_t operator()(const SignalUsage &val) const {
      size_t hash = std::hash<unsigned>{}(val.blockArgIdx);
      for (auto f : val.fieldRefs) {
        hash ^= OpHash<FieldDefOp>{}(f);
      }
      return hash;
    }
  };

private:
  /**
   * If the block arg is 0, then it refers to "self", meaning the signal is internal or an output
   * (public means an output) Otherwise, it is an input, either public or private.
   */
  unsigned blockArgIdx;
  std::vector<FieldDefOp> fieldRefs;
};

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
        operandVals.signals.insert(*res);
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

      res->useDefSubscribe(this);
      llvm::errs() << "Subscribing result at point " << res->getPoint() << "\n";

      for (const auto *operand : operands) {
        llvm::errs() << "\tjoining ";
        res->print(llvm::errs());
        llvm::errs() << " and ";
        operand->print(llvm::errs());
        llvm::errs() << '\n';

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

void reportUnconstrainedSignals(StructDefOp &component) {
  llvm::errs() << "Component is " << component << "\n";
  /**
   * Report two types of bugs:
   * 1. Signals that are read or written in the compute function but do not appear
   * in emitted constraints in the constraint function.
   * 2. Signals that are completely unused.
   *
   * To report both of these kinds of issues uniquely, we will apply the following
   * approach:
   * 1. Find all signals.
   * 2. Find all compute uses of signals.
   * 3. Find all constraint uses of signals.
   *
   * For any signal that has an empty constraint set, we report a bug. The text
   * of this bug will differ if the compute set is empty or non-empty; if empty,
   * this is an unused signal, and if non-empty, this signal is unconstrained.
   * References will be added in the latter case.
   */

  // This will be slightly more complicated based on the fact that we may have
  // function calls and such. So, we will need a form of dataflow analysis here.

  /**
   * The dataflow analysis I think would best be represented as:
   * - Compute the set of fields used for each operation
   * - Query all dataflow operations
   * - Query all constraint operations
   */

  std::vector<FieldDefOp> fields;
  for (auto field : component.getOps<FieldDefOp>()) {
    fields.emplace_back(std::move(field));
  }

  auto computeFnOp = mlir::cast_or_null<FuncOp>(component.lookupSymbol("compute"));
  auto constrainFnOp = mlir::cast_or_null<FuncOp>(component.lookupSymbol("constrain"));
  // ENSURE(computeFnOp != nullptr && constrainFnOp != nullptr, "component must have compute and
  // constrain function");

  // Run the analysis

  // The dataflow solver only runs on live operations, so we force everything to be live
  makeLive(solver, component.getOperation());
  auto analysis = solver.load<SignalUsageAnalysis>();
  // auto res = solver.initializeAndRun(computeFnOp);
  auto res = solver.initializeAndRun(constrainFnOp);
  if (res.failed()) {
    llvm::errs() << "Analysis failed!\n";
    return;
  }
  llvm::errs() << "We have run the analysis!\n";

  // query the analysis
  for (auto &op : constrainFnOp.getOps()) {
    for (auto r : op.getOpResults()) {
      auto resState = solver.lookupState<SignalUsageLattice>(r);
      if (resState) {
        llvm::errs() << "\t\tResult state for " << r << ": ";
        resState->print(llvm::errs());
        llvm::errs() << '\n';
      } else {
        llvm::errs() << "\t\tNo result state for " << r << '\n';
      }
    }
  }
  /**
   * Now, given the analysis, construct the summary
   * 1. Add all signals
   * 2. Union all signals based on solver results
   * 3. Union all signals based on nested summaries
   */

  llvm::EquivalenceClasses<SignalUsage> constraints;

  // Add all the constraints from the analysis

  /**
   * Step two of the analysis is to traverse all of the constrain calls.
   * This is the nested analysis, basically.
   * Constrain functions don't return, so I don't need to compute "values" from
   * the call. I just need to see what constrains are generated here, and
   * add them to the transitive closures.
   */
  mlir::SymbolTableCollection tables;
  constrainFnOp.walk([&tables](CallOp fnCall) mutable {
    auto res = resolveCallable<FuncOp>(tables, fnCall);
    if (mlir::failed(res)) {
      fnCall.emitError() << "Could not resolve callable!\n";
      return;
    }
    auto fn = res->get();
    if (fn.getName() == FUNC_NAME_CONSTRAIN) {
      // Nested
      llvm::errs() << "nested thing\n";
    }
  });
}

/* ConstraintSummary */

ConstraintSummary::ConstraintSummary(
    StructDefOp s, mlir::DataFlowSolver &solver, mlir::AnalysisManager &am
)
    : structDef(s) {
  reportUnconstrainedSignals(s);
}

void ConstraintSummary::dump() const { print(llvm::errs()); }

void ConstraintSummary::print(llvm::raw_ostream &os) const { os << "todo!\n"; }

/* ConstraintSummaryAnalysis */

ConstraintSummaryAnalysis::ConstraintSummaryAnalysis(
    mlir::Operation *op, mlir::AnalysisManager &am
) {
  if (auto modOp = mlir::dyn_cast<mlir::ModuleOp>(op)) {
    mlir::DataFlowSolver solver;
    makeLive(solver, modOp);

    modOp.walk([this, &solver, &am](StructDefOp structOp) mutable {
      ConstraintSummary summary(structOp, solver, am);
      summaries.emplace(std::make_pair(structOp, summary));
    });
  } else {
    auto error_message = "ConstraintSummaryAnalysis expects provided op to be a StructDefOp!";
    op->emitError(error_message);
    llvm::report_fatal_error(error_message);
  }
}

void ConstraintSummaryAnalysis::makeLive(mlir::DataFlowSolver &solver, mlir::Operation *top) {
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