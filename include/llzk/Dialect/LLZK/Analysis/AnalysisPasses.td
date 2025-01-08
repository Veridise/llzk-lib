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

def ConstraintSummaryPrinterPass : Pass<"llzk-print-constraint-summaries", "::mlir::ModuleOp"> {
  let summary = "Print constraint summaries for all LLZK structs.";
  let constructor = "llzk::createConstraintSummaryPrinterPass(llvm::errs())";
  let dependentDialects = ["llzk::LLZKDialect"];
}

#endif // LLZK_ANALYSIS_TD
