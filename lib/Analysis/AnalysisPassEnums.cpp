//===-- AnalysisPassEnums.cpp ---------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Analysis/AnalysisPassEnums.h"

#include <llvm/ADT/StringSwitch.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/ErrorHandling.h>

// TableGen'd implementation files
#include "llzk/Analysis/AnalysisPassEnums.cpp.inc"

namespace llzk {

llvm::raw_ostream &toStream(OutputStream val) {
  switch (val) {
  case OutputStream::Outs:
    return llvm::outs();
  case OutputStream::Errs:
    return llvm::errs();
  case OutputStream::Dbgs:
    return llvm::dbgs();
  }
  llvm_unreachable("Unhandled OutputStream value");
}

} // namespace llzk
