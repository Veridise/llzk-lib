#pragma once

#include "llzk/Dialect/LLZK/Analysis/DenseAnalysis.h"
#include "llzk/Dialect/LLZK/Util/Compare.h"
#include "llzk/Dialect/LLZK/Util/ErrorHelper.h"
#include "llzk/Dialect/LLZK/Util/SymbolHelper.h"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/AnalysisManager.h>

#include <map>

namespace llzk {

template <typename Result> class StructAnalysis {
public:
  StructAnalysis(mlir::Operation *op) {
    structDefOp = mlir::dyn_cast<StructDefOp>(op);
    if (!structDefOp) {
      auto error_message = "StructAnalysis expects provided op to be a StructDefOp!";
      op->emitError(error_message);
      llvm::report_fatal_error(error_message);
    }
    auto maybeModOp = getRootModule(op);
    if (mlir::failed(maybeModOp)) {
      auto error_message = "StructAnalysis could not find root module from StructDefOp!";
      op->emitError(error_message);
      llvm::report_fatal_error(error_message);
    }
    modOp = *maybeModOp;
  }

  virtual mlir::LogicalResult
  runAnalysis(mlir::DataFlowSolver &solver, mlir::AnalysisManager &moduleAnalysisManager) = 0;

  bool constructed() const { return res != nullptr; }

  const Result &getResult() const { return *res; }

protected:
  mlir::ModuleOp getModule() const { return modOp; }

  StructDefOp getStruct() const { return structDefOp; }

  void setResult(Result &&r) { res = std::make_unique<Result>(r); }

private:
  mlir::ModuleOp modOp;
  StructDefOp structDefOp;
  std::unique_ptr<Result> res;
};

template <typename Analysis, typename Result>
concept StructAnalysisType =
    requires { requires std::is_base_of<StructAnalysis<Result>, Analysis>::value; };

template <
    typename Result, StructAnalysisType<Result> StructAnalysisType,
    typename... DataFlowSolverAnalyses>
class ModuleAnalysis {
  /// Using a map, not an unordered map, to control sorting order for iteration.
  using ResultMap =
      std::map<StructDefOp, std::reference_wrapper<const Result>, OpLocationLess<StructDefOp>>;

public:
  ModuleAnalysis(mlir::Operation *op, mlir::AnalysisManager &am) {
    if (modOp = mlir::dyn_cast<mlir::ModuleOp>(op)) {
      mlir::DataFlowConfig config;
      mlir::DataFlowSolver solver(config);
      dataflow::markAllOpsAsLive(solver, modOp);

      // The analysis is run at the module level so that lattices are computed
      // for global functions as well.
      ((solver.load<DataFlowSolverAnalyses>()), ...);
      auto res = solver.initializeAndRun(modOp);
      ensure(res.succeeded(), "solver failed to run on module!");

      modOp.walk([this, &solver, &am](StructDefOp s) mutable {
        auto &childAnalysis = am.getChildAnalysis<StructAnalysisType>(s);
        if (mlir::failed(childAnalysis.runAnalysis(solver, am))) {
          auto error_message = "StructAnalysis failed to run for " + mlir::Twine(s.getName());
          s->emitError(error_message);
          llvm::report_fatal_error(error_message);
        }
        // auto p = std::make_pair(s, std::reference_wrapper(childAnalysis.getResult()));
        // results.insert(std::move(p));
        results.insert(
            std::make_pair(StructDefOp(s), std::reference_wrapper(childAnalysis.getResult()))
        );
      });
    } else {
      auto error_message = "ModuleAnalysis expects provided op to be an mlir::ModuleOp!";
      op->emitError(error_message);
      llvm::report_fatal_error(error_message);
    }
  }

  bool hasResult(StructDefOp op) const { return results.find(op) != results.end(); }
  Result &getResult(StructDefOp op) {
    ensureResultCreated(op);
    return *results.at(op);
  }
  const Result &getResult(StructDefOp op) const {
    ensureResultCreated(op);
    return *results.at(op);
  }

  ResultMap::iterator begin() { return results.begin(); }
  ResultMap::iterator end() { return results.end(); }
  ResultMap::const_iterator cbegin() const { return results.cbegin(); }
  ResultMap::const_iterator cend() const { return results.cend(); }

private:
  mlir::ModuleOp modOp;
  ResultMap results;

  /// @brief Ensures that the given struct has a CDG.
  /// @param op The struct to ensure has a CDG.
  void ensureResultCreated(StructDefOp op) const {
    ensure(hasResult(op), "Result does not exist for StructDefOp " + mlir::Twine(op.getName()));
  }
};

} // namespace llzk
