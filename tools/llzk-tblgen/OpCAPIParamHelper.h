//===- OpCAPIParamHelper.h ------------------------------------------------===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// Helpers for generating C API and C API link tests for Operations.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <mlir/TableGen/Argument.h>
#include <mlir/TableGen/Operator.h>

#include <string>

/// Helper struct to generate a string from operation operand, attribute, and result pieces.
struct GenStringFromOpPieces {
  virtual ~GenStringFromOpPieces() = default;

  /// Generate string from the operation pieces.
  std::string gen(const mlir::tblgen::Operator &op);

protected:
  /// Generate header code to `os`. Default does nothing.
  virtual void genHeader(llvm::raw_ostream &os) {}

  /// Generate code for `result` to `os`.
  virtual void genResult(
      llvm::raw_ostream &os, const mlir::tblgen::NamedTypeConstraint &result,
      const std::string &resultName
  ) = 0;

  /// Generate code to `os` when result type is inferred. Default does nothing.
  virtual void genResultInferred(llvm::raw_ostream &os) {}

  /// Generate code for `operand` to `os`.
  virtual void
  genOperand(llvm::raw_ostream &os, const mlir::tblgen::NamedTypeConstraint &operand) = 0;

  /// Generate attribute section prefix code to `os`. Default does nothing.
  virtual void genAttributesPrefix(llvm::raw_ostream &os, const mlir::tblgen::Operator &op) {}

  /// Generate code for `attr` to `os`.
  virtual void genAttribute(llvm::raw_ostream &os, const mlir::tblgen::NamedAttribute &attr) = 0;

  /// Generate attribute section suffix code to `os`. Default does nothing.
  virtual void genAttributesSuffix(llvm::raw_ostream &os, const mlir::tblgen::Operator &op) {}

  /// Generate region section prefix code to `os`. Default does nothing.
  virtual void genRegionsPrefix(llvm::raw_ostream &os, const mlir::tblgen::Operator &op) {}

  /// Generate code for `region` to `os`.
  virtual void genRegion(llvm::raw_ostream &os, const mlir::tblgen::NamedRegion &region) = 0;

  /// Generate region section suffix code to `os`. Default does nothing.
  virtual void genRegionsSuffix(llvm::raw_ostream &os, const mlir::tblgen::Operator &op) {}
};
