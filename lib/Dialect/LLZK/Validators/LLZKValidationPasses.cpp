//===-- LLZKValidationPasses.cpp - LLZK validation passes -------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the implementation for the `-llzk-validate-field-writes`
/// pass.
///
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/LLZK/IR/Ops.h"
#include "llzk/Dialect/LLZK/Validators/LLZKValidationPasses.h"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>

// Include the generated base pass class definitions.
namespace llzk {
#define GEN_PASS_DEF_FIELDWRITEVALIDATORPASS
#include "llzk/Dialect/LLZK/Validators/LLZKValidationPasses.h.inc"
} // namespace llzk

using namespace mlir;
using namespace llzk;

namespace {
class FieldWriteValidatorPass
    : public llzk::impl::FieldWriteValidatorPassBase<FieldWriteValidatorPass> {
  void runOnOperation() override {
    StructDefOp structDef = getOperation();
    FuncOp computeFunc = structDef.getComputeFuncOp();

    // Initialize map with all field names mapped to nullptr (i.e. no write found).
    llvm::StringMap<FieldWriteOp> fieldNameToWriteOp;
    for (FieldDefOp x : structDef.getFieldDefs()) {
      fieldNameToWriteOp[x.getSymName()] = nullptr;
    }
    // Search the function body for writes, store them in the map and emit warning if multiple
    // writes to the same field are found.
    for (Block &block : computeFunc.getBody()) {
      for (Operation &op : block) {
        if (FieldWriteOp write = dyn_cast<FieldWriteOp>(op)) {
          // FieldWriteOp::verifySymbolUses() ensures FieldWriteOp only target the containing "self"
          // struct. That means the target of the FieldWriteOp must be in `fieldNameToWriteOp` so
          // using 'at()' will not abort.
          assert(structDef.getType() == write.getComponent().getType());
          StringRef writeToFieldName = write.getFieldName();
          if (FieldWriteOp earlierWrite = fieldNameToWriteOp.at(writeToFieldName)) {
            write.emitWarning()
                .append(
                    "found multiple writes to '", FieldDefOp::getOperationName(), "' named \"@",
                    writeToFieldName, "\""
                )
                .attachNote(earlierWrite.getLoc())
                .append("earlier write here");
          }
          fieldNameToWriteOp[writeToFieldName] = write;
        }
      }
    }
    // Finally, report a warning if any field was not written at all.
    for (auto &[a, b] : fieldNameToWriteOp) {
      if (!b) {
        computeFunc.emitWarning().append(
            "'", FuncOp::getOperationName(), "' op \"@", FUNC_NAME_COMPUTE, "\" missing write to '",
            FieldDefOp::getOperationName(), "' named \"@", a, "\""
        );
      }
    }

    markAllAnalysesPreserved();
  }
};
} // namespace

std::unique_ptr<mlir::Pass> llzk::createFieldWriteValidatorPass() {
  return std::make_unique<FieldWriteValidatorPass>();
};
