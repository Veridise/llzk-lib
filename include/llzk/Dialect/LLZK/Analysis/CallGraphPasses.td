#ifndef LLZK_CALLGRAPH_TD
#define LLZK_CALLGRAPH_TD

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

#endif // LLZK_CALLGRAPH_TD
