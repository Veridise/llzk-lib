//===-- AnalysisPassEnums.h -----------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include <mlir/Pass/Pass.h>
#include <mlir/Support/LogicalResult.h>

#include <llvm/ADT/StringRef.h>
#include <llvm/Support/raw_ostream.h>

#include <optional>

// Include TableGen'd declarations
#include "llzk/Analysis/AnalysisPassEnums.h.inc"

namespace llzk {

llvm::raw_ostream &toStream(OutputStream val);
inline llvm::raw_ostream &toStream(mlir::Pass::Option<OutputStream> &val) {
  return toStream(val.getValue());
}

} // namespace llzk
