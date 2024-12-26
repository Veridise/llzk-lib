/**
 * The contents of this file are adapted from llvm/include/llvm/Analysis/CallGraph.h.
 */

#pragma once

#include <mlir/Pass/Pass.h>

#include <llvm/ADT/SCCIterator.h>
#include <llvm/ADT/STLExtras.h>

#include <cassert>
#include <map>
#include <memory>
#include <unordered_set>
#include <utility>
#include <vector>

namespace llzk {

std::unique_ptr<mlir::Pass> createCallGraphPrinterPass(llvm::raw_ostream &os);

std::unique_ptr<mlir::Pass> createCallGraphSCCsPrinterPass(llvm::raw_ostream &os);

#define GEN_PASS_REGISTRATION
#include "llzk/Dialect/LLZK/Analysis/CallGraphPasses.h.inc"

} // namespace llzk
