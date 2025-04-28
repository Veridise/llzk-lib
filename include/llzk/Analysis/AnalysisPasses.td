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

include "llzk/Pass/PassBase.td"

def CallGraphPrinterPass : LLZKPass<"llzk-print-call-graph"> {
  let summary = "Print the LLZK module's call graph.";
  let constructor = "llzk::createCallGraphPrinterPass(llvm::errs())";
}

def CallGraphSCCsPrinterPass : LLZKPass<"llzk-print-call-graph-sccs"> {
  let summary = "Print the SCCs from the LLZK module's call graph.";
  let constructor = "llzk::createCallGraphSCCsPrinterPass(llvm::errs())";
}

def ConstraintDependencyGraphPrinterPass
    : LLZKPass<"llzk-print-constraint-dependency-graphs"> {
  let summary = "Print constraint dependency graph for all LLZK structs.";
  let constructor =
      "llzk::createConstraintDependencyGraphPrinterPass(llvm::errs())";
}

def IntervalAnalysisPrinterPass : LLZKPass<"llzk-print-interval-analysis"> {
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
}

#endif // LLZK_ANALYSIS_TD
