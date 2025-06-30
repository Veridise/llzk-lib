//===-- GraphUtil.h ---------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include <llvm/ADT/GraphTraits.h>
#include <llvm/ADT/SCCIterator.h>

namespace llzk {

template <typename GraphT> inline bool hasCycle(const GraphT &G) {
  for (llvm::scc_iterator<GraphT> I = llvm::scc_begin(G), E = llvm::scc_end(G); I != E; ++I) {
    const std::vector<typename llvm::GraphTraits<GraphT>::NodeRef> &SCC = *I;
    if (SCC.size() > 1) {
      return true;
    }
    // Detect self-loop
    auto *N = SCC.front();
    for (auto *child : llvm::children<GraphT>(N)) {
      if (child == N) {
        return true;
      }
    }
  }
  return false;
}

} // namespace llzk
