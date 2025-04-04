//===-- AnalysisPasses.td - LLZK Analysis Passes -----------*- tablegen -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_ANALYSIS_TD
#define LLZK_ANALYSIS_TD

include "mlir/Pass/PassBase.td"

def CallGraphPrinterPass : Pass<"llzk-print-call-graph", "::mlir::ModuleOp"> {
  let summary = "Print the LLZK module's call graph.";
  let constructor = "llzk::createCallGraphPrinterPass(llvm::errs())";
  let dependentDialects = ["llzk::LLZKDialect"];
}

def CallGraphSCCsPrinterPass
    : Pass<"llzk-print-call-graph-sccs", "::mlir::ModuleOp"> {
  let summary = "Print the SCCs from the LLZK module's call graph.";
  let constructor = "llzk::createCallGraphSCCsPrinterPass(llvm::errs())";
  let dependentDialects = ["llzk::LLZKDialect"];
}

def ConstraintDependencyGraphPrinterPass
    : Pass<"llzk-print-constraint-dependency-graphs", "::mlir::ModuleOp"> {
  let summary = "Print constraint dependency graph for all LLZK structs.";
  let constructor =
      "llzk::createConstraintDependencyGraphPrinterPass(llvm::errs())";
  let dependentDialects = ["llzk::LLZKDialect"];
}

def IntervalAnalysisPrinterPass
    : Pass<"llzk-print-interval-analysis", "::mlir::ModuleOp"> {
  let summary = "Print interval analysis results for all LLZK structs.";
  let constructor = "llzk::createIntervalAnalysisPrinterPass(llvm::errs())";
  let options =
      [Option<"fieldName", "field", "std::string", /* default */ "\"bn128\"",
              "The field to use for interval analysis. Supported fields: "
              "bn128/bn254, babybear, goldilocks, mersenne31">,
       Option<"printSolverConstraints", "print-solver-constraints", "bool",
              /* default */ "false",
              "Whether to output SMT solver constraints along with intervals.">,
  ];
  let dependentDialects = ["llzk::LLZKDialect"];
}

#endif // LLZK_ANALYSIS_TD
