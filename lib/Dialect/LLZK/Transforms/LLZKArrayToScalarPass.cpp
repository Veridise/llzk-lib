//===-- LLZKArrayToScalarPass.cpp -------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the `-llzk-array-to-scalar` pass.
///
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/LLZK/IR/Ops.h"
#include "llzk/Dialect/LLZK/Transforms/LLZKTransformationPasses.h"
#include "llzk/Dialect/LLZK/Util/ArrayTypeHelper.h"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/Passes.h>

#include <llvm/Support/Debug.h>

// Include the generated base pass class definitions.
namespace llzk {
#define GEN_PASS_DEF_ARRAYTOSCALARPASS
#include "llzk/Dialect/LLZK/Transforms/LLZKTransformationPasses.h.inc"
} // namespace llzk

using namespace mlir;
using namespace llzk;

#define DEBUG_TYPE "llzk-array-to-scalar"

namespace {

class SplitInitFromCreate : public OpConversionPattern<CreateArrayOp> {
public:
  SplitInitFromCreate(MLIRContext *ctx) : OpConversionPattern<CreateArrayOp>(ctx) {}

  LogicalResult match(CreateArrayOp op) const override { return failure(op.getElements().empty()); }

  void
  rewrite(CreateArrayOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    // Remove elements from `op`
    rewriter.modifyOpInPlace(op, [&op]() { op.getElementsMutable().clear(); });
    // Generate an individual write for each initialization element
    rewriter.setInsertionPointAfter(op);
    Location loc = op.getLoc();
    ArrayIndexGen idxGen = ArrayIndexGen::from(op.getType());
    for (auto [i, init] : llvm::enumerate(adaptor.getElements())) {
      // Convert the linear index 'i' into a multi-dim index
      assert(std::cmp_less_equal(i, std::numeric_limits<int64_t>::max()));
      std::optional<SmallVector<Value>> multiDimIdxVals =
          idxGen.delinearize(static_cast<int64_t>(i), loc, rewriter);
      // ASSERT: CreateArrayOp verifier ensures the number of elements provided matches the full
      // linear array size so delinearization of `i` will not fail.
      assert(multiDimIdxVals.has_value());
      // Create the write
      rewriter.create<WriteArrayOp>(loc, op.getResult(), ValueRange(*multiDimIdxVals), init);
    }
  }
};

LogicalResult splitArrayCreateInit(ModuleOp modOp) {
  MLIRContext *ctx = modOp.getContext();
  RewritePatternSet patterns(ctx);
  patterns.add<SplitInitFromCreate>(ctx);
  ConversionTarget target(*ctx);
  target.addLegalDialect<LLZKDialect, arith::ArithDialect, scf::SCFDialect>();
  target.addLegalOp<ModuleOp>();
  target.addDynamicallyLegalOp<CreateArrayOp>([&](CreateArrayOp op) {
    return op.getElements().empty();
  });
  return applyFullConversion(modOp, target, std::move(patterns));
}

class ArrayToScalarPass : public llzk::impl::ArrayToScalarPassBase<ArrayToScalarPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    // Separate array initialization from creation by removing the initalization list from
    // CreateArrayOp and inserting the corresponding WriteArrayOp following it.
    if (failed(splitArrayCreateInit(module))) {
      signalPassFailure();
      return;
    }
    OpPassManager nestedPM(ModuleOp::getOperationName());
    // Use SROA (Destructurable* interfaces) to split each array with linear size N into N arrays of
    // size 1. This is necessary because the mem2reg pass cannot deal with indexing and splitting up
    // memory, i.e. it can only convert scalar memory access into SSA values.
    nestedPM.addPass(createSROA());
    // The mem2reg pass converts all of the size 1 array allocation and access into SSA values.
    nestedPM.addPass(createMem2Reg());
    if (failed(runPipeline(nestedPM, module))) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace

std::unique_ptr<Pass> llzk::createArrayToScalarPass() {
  return std::make_unique<ArrayToScalarPass>();
};
