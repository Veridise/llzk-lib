//===-- Export.h - LLZK to Picus translation --------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines llzk::translateModuleToPicus.
///
//===----------------------------------------------------------------------===//

#pragma once

#include <memory>

namespace mlir {
class Operation;
}

namespace picus {
class Circuit;
}

namespace llzk {

/// Converts a LLZK program into a Picus circuit that can be used for verification.
/// Accepts a ModuleOp with the LLZK language attribute and the prime number that
/// defines the finite field used by the circuit.
/// Returns a valid Circuit pointer if the translation was successfull and a nullptr
/// if it failed.
std::unique_ptr<picus::Circuit> translateModuleToPicus(mlir::Operation *, llvm::APInt);

} // namespace llzk
