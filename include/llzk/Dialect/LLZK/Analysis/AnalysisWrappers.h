//===-- AnalysisWrappers.h --------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Convenience classes for a frequent pattern of dataflow analysis used in LLZK,
/// where an analysis is run across all `StructDefOp`s contained within a module,
/// where each of those analyses may need to reference the analysis results from
/// other `StructDefOp`s. This pattern reoccurs due to the instantiation of subcomponents
/// within components, which often requires the instantiating component to look up
/// the results of an analysis on the subcomponent. This kind of lookup is not
/// supported through mlir's AnalysisManager, as it only allows lookups on nested operations,
/// not sibling operations. This pattern allows subcomponents to instead use the ModuleOp's
/// analysis manager, allowing components to query analyses for any component in the module.
///
//===----------------------------------------------------------------------===//

#pragma once

#include "llzk/Dialect/LLZK/Analysis/DenseAnalysis.h"
#include "llzk/Dialect/LLZK/Util/Compare.h"
#include "llzk/Dialect/LLZK/Util/ErrorHelper.h"
#include "llzk/Dialect/LLZK/Util/SymbolHelper.h"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/AnalysisManager.h>

#include <map>

namespace llzk {

/// @brief This is the base class for a dataflow analysis designed to run on a single struct (i.e.,
/// a single component).
/// @tparam Result The output of the analysis.
/// @tparam Context Any module-level information or configuration needed to run this analysis.
template <typename Result, typename Context> class StructAnalysis {
public:
  /// @brief Assert that this analysis is being run on a StructDefOp and initializes the
  /// analysis with the current StructDefOp and its parent ModuleOp.
  /// @param op The presumed StructDefOp.
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

  /// @brief Perform the analysis and construct the `Result` output.
  /// @param solver The pre-configured dataflow solver. This solver should already have
  /// a liveness analysis run, otherwise this analysis may be a no-op.
  /// @param moduleAnalysisManager The analysis manager of the top-level module. By giving
  /// the struct analysis a reference to the module's analysis manager, we can query analyses of
  /// other structs by querying for a child analysis. Otherwise, a struct's analysis manager cannot
  /// query for the analyses of other operations unless they are nested within the struct.
  /// @param ctx The `Context` given to the analysis. This is presumed to have been created by the
  /// StructAnalysis's parent ModuleAnalysis.
  /// @return `mlir::success()` if the analysis ran without errors, and a `mlir::failure()`
  /// otherwise.
  virtual mlir::LogicalResult runAnalysis(
      mlir::DataFlowSolver &solver, mlir::AnalysisManager &moduleAnalysisManager, Context &ctx
  ) = 0;

  /// @brief Query if the analysis has constructed a `Result` object.
  bool constructed() const { return res != nullptr; }

  /// @brief Access the result iff it has been created.
  const Result &getResult() const {
    ensure(constructed(), mlir::Twine(__PRETTY_FUNCTION__) + ": result has not been constructed");
    return *res;
  }

protected:
  /// @brief Get the `ModuleOp` that is the parent of the `StructDefOp` that is under analysis.
  mlir::ModuleOp getModule() const { return modOp; }

  /// @brief Get the current `StructDefOp` that is under analysis.
  StructDefOp getStruct() const { return structDefOp; }

  /// @brief Initialize the final `Result` object.
  void setResult(Result &&r) { res = std::make_unique<Result>(r); }

private:
  mlir::ModuleOp modOp;
  StructDefOp structDefOp;
  std::unique_ptr<Result> res;
};

/// @brief An empty struct that is used for convenience for analyses that do not
/// require any context.
struct NoContext {};

/// @brief Any type that is a subclass of `StructAnalysis`.
template <typename Analysis, typename Result, typename Context>
concept StructAnalysisType =
    requires { requires std::is_base_of<StructAnalysis<Result, Context>, Analysis>::value; };

/// @brief An analysis wrapper that runs the given `StructAnalysisTy` struct analysis over
/// all of the struct contained within the module. Through the use of the `Context` object, this
/// analysis facilitates the sharing of common data and analyses across struct analyses.
/// @tparam Result The result of each `StructAnalysis`.
/// @tparam Context The context shared between `StructAnalysis` analyses.
/// @tparam StructAnalysisType The analysis run on all the contained module's structs.
template <typename Result, typename Context, StructAnalysisType<Result, Context> StructAnalysisTy>
class ModuleAnalysis {
  /// @brief A map of this module's structs to the result of the `StructAnalysis` on that struct.
  /// The `ResultMap` is implemented as an ordered map to control sorting order for iteration.
  using ResultMap =
      std::map<StructDefOp, std::reference_wrapper<const Result>, OpLocationLess<StructDefOp>>;

public:
  /// @brief Asserts that the analysis is being run on a `ModuleOp`.
  /// @note Derived classes may also use the `Analysis(mlir::Operation*, mlir::AnalysisManager&)`
  /// constructor that is allowed by classes that are constructed using the
  /// `AnalysisManager::getAnalysis<Analysis>()` method.
  ModuleAnalysis(mlir::Operation *op) {
    if (modOp = mlir::dyn_cast<mlir::ModuleOp>(op); !modOp) {
      auto error_message = "ModuleAnalysis expects provided op to be an mlir::ModuleOp!";
      op->emitError(error_message);
      llvm::report_fatal_error(error_message);
    }
  }

  /// @brief Run the `StructAnalysisTy` struct analysis on all child structs.
  /// @param am The module-level analysis manager that will be passed to
  /// `StructAnalysis::runAnalysis`. This analysis manager should be the same analysis manager used
  /// to construct this analysis.
  virtual void runAnalysis(mlir::AnalysisManager &am) { constructChildAnalyses(am); }

  /// @brief Checks if `op` has a result contained in the current result map.
  bool hasResult(StructDefOp op) const { return results.find(op) != results.end(); }

  /// @brief Asserts that `op` has a result and returns it.
  const Result &getResult(StructDefOp op) const {
    ensureResultCreated(op);
    return results.at(op).get();
  }

  ResultMap::iterator begin() { return results.begin(); }
  ResultMap::iterator end() { return results.end(); }
  ResultMap::const_iterator cbegin() const { return results.cbegin(); }
  ResultMap::const_iterator cend() const { return results.cend(); }

protected:
  /// @brief Initialize the shared dataflow solver with any common analyses required
  /// by the contained struct analyses.
  /// @param solver
  virtual void initializeSolver(mlir::DataFlowSolver &solver) = 0;

  /// @brief Create and return a valid `Context` object. This function is called
  /// once by `constructChildAnalyses` and the resulting `Context` is passed to
  /// each child struct analysis run by this module analysis.
  virtual Context getContext() = 0;

  /// @brief Construct and run the `StructAnalysisTy` analyses on each `StructDefOp` contained
  /// in the `ModuleOp` that is being subjected to this analysis.
  /// @param am The module's analysis manager.
  void constructChildAnalyses(mlir::AnalysisManager &am) {
    mlir::DataFlowConfig config;
    mlir::DataFlowSolver solver(config);
    dataflow::markAllOpsAsLive(solver, modOp);

    // The analysis is run at the module level so that lattices are computed
    // for global functions as well.
    initializeSolver(solver);
    auto res = solver.initializeAndRun(modOp);
    ensure(res.succeeded(), "solver failed to run on module!");

    auto ctx = getContext();
    modOp.walk([this, &solver, &am, &ctx](StructDefOp s) mutable {
      auto &childAnalysis = am.getChildAnalysis<StructAnalysisTy>(s);
      if (mlir::failed(childAnalysis.runAnalysis(solver, am, ctx))) {
        auto error_message = "StructAnalysis failed to run for " + mlir::Twine(s.getName());
        s->emitError(error_message);
        llvm::report_fatal_error(error_message);
      }
      results.insert(
          std::make_pair(StructDefOp(s), std::reference_wrapper(childAnalysis.getResult()))
      );
    });
  }

private:
  mlir::ModuleOp modOp;
  ResultMap results;

  /// @brief Ensures that the given struct has a result.
  /// @param op The struct to ensure has a result.
  void ensureResultCreated(StructDefOp op) const {
    ensure(hasResult(op), "Result does not exist for StructDefOp " + mlir::Twine(op.getName()));
  }
};

} // namespace llzk
