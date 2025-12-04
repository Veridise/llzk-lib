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

#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Validators/LLZKValidationPasses.h"

#include <mlir/IR/BuiltinOps.h>

// Include the generated base pass class definitions.
namespace llzk {
#define GEN_PASS_DEF_FIELDWRITEVALIDATORPASS
#include "llzk/Validators/LLZKValidationPasses.h.inc"
} // namespace llzk

using namespace mlir;
using namespace llzk;
using namespace llzk::component;
using namespace llzk::function;

namespace {
class FieldWriteValidatorPass
    : public llzk::impl::FieldWriteValidatorPassBase<FieldWriteValidatorPass> {
  void runOnOperation() override {
    StructDefOp structDef = getOperation();
    FuncDefOp computeFunc = structDef.getComputeFuncOp();

    // Initialize map with all field names mapped to nullptr (i.e., no write found).
    llvm::StringMap<FieldWriteOp> fieldNameToWriteOp;
    for (MemberDefOp x : structDef.getMemberDefs()) {
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
            auto diag = write.emitWarning().append(
                "found multiple writes to '", MemberDefOp::getOperationName(), "' named \"@",
                writeToFieldName, '"'
            );
            diag.attachNote(earlierWrite.getLoc()).append("earlier write here");
            diag.report();
          }
          fieldNameToWriteOp[writeToFieldName] = write;
        }
      }
    }
    // Finally, report a warning if any field was not written at all.
    for (auto &[a, b] : fieldNameToWriteOp) {
      if (!b) {
        computeFunc.emitWarning()
            .append(
                '\'', FuncDefOp::getOperationName(), "' op \"@", FUNC_NAME_COMPUTE,
                "\" missing write to '", MemberDefOp::getOperationName(), "' named \"@", a, '"'
            )
            .report();
      }
    }

    markAllAnalysesPreserved();
  }
};
} // namespace

std::unique_ptr<mlir::Pass> llzk::createFieldWriteValidatorPass() {
  return std::make_unique<FieldWriteValidatorPass>();
};
