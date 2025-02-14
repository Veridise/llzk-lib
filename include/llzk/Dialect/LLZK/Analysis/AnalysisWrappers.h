#pragma once

#include "llzk/Dialect/LLZK/Analysis/DenseAnalysis.h"
#include "llzk/Dialect/LLZK/Util/SymbolHelper.h"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/AnalysisManager.h>

namespace llzk {

class StructAnalysis {
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

protected:
  mlir::ModuleOp getModule() const { return modOp; }

  StructDefOp getStruct() const { return structDefOp; }

private:
  mlir::ModuleOp modOp;
  StructDefOp structDefOp;
};

template <typename A>
concept StructAnalysisType = requires {
  requires std::is_base_of<StructAnalysis, A>::value;
  // requires std::default_initializable<A>;
};

template <StructAnalysisType StructAnalysisType, typename... DataFlowSolverAnalyses>
class ModuleAnalysis {
public:
  ModuleAnalysis(mlir::Operation *op, mlir::AnalysisManager &am) {
    if (modOp = mlir::dyn_cast<mlir::ModuleOp>(op)) {
      mlir::DataFlowConfig config;
      mlir::DataFlowSolver solver(config);
      dataflow::markAllOpsAsLive(solver, modOp);

      // The analysis is run at the module level so that lattices are computed
      // for global functions as well.
      ((solver.load<DataFlowSolverAnalyses>()), ...);
      // solver.load<ConstrainRefAnalysis>();
      auto res = solver.initializeAndRun(modOp);
      debug::ensure(res.succeeded(), "solver failed to run on module!");

      modOp.walk([this, &solver, &am](StructDefOp s) {
        auto &childAnalysis = am.getChildAnalysis<StructAnalysisType>(s);
        if (mlir::failed(childAnalysis.runAnalysis(solver, am))) {
          auto error_message = "StructAnalysis failed to run for " + mlir::Twine(s.getName());
          s->emitError(error_message);
          llvm::report_fatal_error(error_message);
        }
      });
    } else {
      auto error_message = "ModuleAnalysis expects provided op to be an mlir::ModuleOp!";
      op->emitError(error_message);
      llvm::report_fatal_error(error_message);
    }
  }

protected:
  mlir::ModuleOp getModule() const { return modOp; }

private:
  mlir::ModuleOp modOp;
};

} // namespace llzk
