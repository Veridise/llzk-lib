//===-- CallGraphAnalyses.h -------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llzk/Dialect/LLZK/Analysis/CallGraph.h"
#include "llzk/Dialect/LLZK/IR/Ops.h"

#include <mlir/Analysis/CallGraph.h>
#include <mlir/Pass/AnalysisManager.h>

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/SCCIterator.h>
#include <llvm/ADT/STLExtras.h>

#include <memory>
#include <utility>
#include <vector>

namespace llvm {

template <class GraphType> struct GraphTraits;
class raw_ostream;

} // namespace llvm

namespace mlir {

class Operation;
class ModuleOp;

} // namespace mlir

namespace llzk {

/// An analysis wrapper to compute the \c CallGraph for a \c Module.
///
/// This class implements the concept of an analysis pass used by the \c
/// ModuleAnalysisManager to run an analysis over a module and cache the
/// resulting data.
class CallGraphAnalysis {
  std::unique_ptr<llzk::CallGraph> cg;

public:
  CallGraphAnalysis(mlir::Operation *op);

  llzk::CallGraph &getCallGraph() { return *cg.get(); }
  const llzk::CallGraph &getCallGraph() const { return *cg.get(); }
};

/// Lazily-constructed reachability analysis.
class CallGraphReachabilityAnalysis {

  // Maps function -> callees
  using CalleeMapTy = mlir::DenseMap<FuncOp, mlir::DenseSet<FuncOp>>;

  mutable CalleeMapTy reachabilityMap;

  std::reference_wrapper<llzk::CallGraph> callGraph;

public:
  CallGraphReachabilityAnalysis(mlir::Operation *, mlir::AnalysisManager &am);

  bool isInvalidated(const mlir::AnalysisManager::PreservedAnalyses &pa) {
    return !pa.isPreserved<CallGraphReachabilityAnalysis>() || !pa.isPreserved<CallGraphAnalysis>();
  }

  /// Returns whether B is reachable from A.
  bool isReachable(FuncOp &A, FuncOp &B) const;

  const llzk::CallGraph &getCallGraph() const { return callGraph.get(); }

private:
  inline bool isReachableCached(FuncOp &A, FuncOp &B) const {
    auto it = reachabilityMap.find(A);
    return it != reachabilityMap.end() && it->second.find(B) != it->second.end();
  }
};

} // namespace llzk
