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

  static bool legal(CreateArrayOp op) { return op.getElements().empty(); }

  LogicalResult match(CreateArrayOp op) const override { return failure(legal(op)); }

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

inline bool containsArrayType(Type t) {
  return t
      .walk([](ArrayType a) {
    return a.hasStaticShape() ? WalkResult::interrupt() : WalkResult::skip();
  }).wasInterrupted();
}

bool containsArrayType(OperandRange::type_range types) {
  for (Type t : types) {
    if (containsArrayType(t)) {
      return true;
    }
  }
  return false;
}

void splitArrayType(Type t, SmallVector<Type> &collect) {
  if (ArrayType at = dyn_cast<ArrayType>(t)) {
    int64_t n = at.getNumElements();
    assert(std::cmp_less_equal(n, std::numeric_limits<SmallVector<Type>::size_type>::max()));
    collect.append(n, at.getElementType());
  } else {
    collect.push_back(t);
  }
}

inline void splitArrayType(ArrayRef<Type> types, SmallVector<Type> &collect) {
  for (Type t : types) {
    splitArrayType(t, collect);
  }
}

inline SmallVector<Type> splitArrayType(ArrayRef<Type> types) {
  SmallVector<Type> collect;
  splitArrayType(types, collect);
  return collect;
}

class SplitArrayInFuncDefOp : public OpConversionPattern<FuncOp> {
public:
  SplitArrayInFuncDefOp(MLIRContext *ctx) : OpConversionPattern<FuncOp>(ctx) {}

  inline static bool legal(FuncOp op) { return !containsArrayType(op.getFunctionType()); }

  LogicalResult match(FuncOp op) const override { return failure(legal(op)); }

  void rewrite(FuncOp op, OpAdaptor, ConversionPatternRewriter &rewriter) const override {
    // Just need to update in/out types of the function to replace arrays with scalars
    rewriter.modifyOpInPlace(op, [&op]() {
      FunctionType funcTy = op.getFunctionType();
      SmallVector<Type> newInputs = splitArrayType(funcTy.getInputs());
      SmallVector<Type> newOutputs = splitArrayType(funcTy.getResults());
      FunctionType newTy =
          FunctionType::get(funcTy.getContext(), TypeRange(newInputs), TypeRange(newOutputs));
      op.setFunctionType(newTy);
    });
  }
};

ReadArrayOp
createRead(ArrayAttr index, Value baseArrayOp, Location loc, ConversionPatternRewriter &rewriter) {
  SmallVector<Value> readOperands;
  for (Attribute a : index) {
    // ASSERT: Attributes are index constants, created by ArrayType::getSubelementIndices().
    IntegerAttr ia = llvm::dyn_cast<IntegerAttr>(a);
    assert(ia && ia.getType().isIndex());
    readOperands.push_back(rewriter.create<arith::ConstantOp>(loc, ia));
  }
  return rewriter.create<ReadArrayOp>(loc, baseArrayOp, ValueRange(readOperands));
}

void processInputOperand(
    Value operand, Location loc, SmallVector<Value> &newOperands,
    ConversionPatternRewriter &rewriter
) {
  if (ArrayType at = dyn_cast<ArrayType>(operand.getType())) {
    std::optional<SmallVector<ArrayAttr>> indices = at.getSubelementIndices();
    assert(indices.has_value() && "passed earlier hasStaticShape() check");
    for (ArrayAttr index : indices.value()) {
      newOperands.push_back(createRead(index, operand, loc, rewriter));
    }
  } else {
    newOperands.push_back(operand);
  }
}

// For each operand with ArrayType, add N reads from the array and use those N values instead.
void processInputOperands(
    MutableOperandRange operands, Operation *op, ConversionPatternRewriter &rewriter
) {
  SmallVector<Value> newOperands;
  for (OpOperand &operand : operands) {
    processInputOperand(operand.get(), op->getLoc(), newOperands, rewriter);
  }
  rewriter.modifyOpInPlace(op, [&operands, &newOperands]() {
    operands.assign(ValueRange(newOperands));
  });
}

class SplitArrayInReturnOp : public OpConversionPattern<ReturnOp> {
public:
  SplitArrayInReturnOp(MLIRContext *ctx) : OpConversionPattern<ReturnOp>(ctx) {}

  inline static bool legal(ReturnOp op) { return !containsArrayType(op.getOperands().getTypes()); }

  LogicalResult match(ReturnOp op) const override { return failure(legal(op)); }

  void rewrite(ReturnOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    // ASSERT: No Pattern in this file replaces ops, only modifies or adds. Hence, the OpAdaptor can
    // be ignored which simplifies the implementation of processInputOperands().
    assert(llvm::equal(op.getOperands(), adaptor.getOperands()));
    processInputOperands(op.getOperandsMutable(), op, rewriter);
  }
};

class SplitArrayInCallOp : public OpConversionPattern<CallOp> {
public:
  SplitArrayInCallOp(MLIRContext *ctx) : OpConversionPattern<CallOp>(ctx) {}

  inline static bool legal(CallOp op) { return !containsArrayType(op.getArgOperands().getTypes()); }

  LogicalResult match(CallOp op) const override { return failure(legal(op)); }

  void rewrite(CallOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    // ASSERT: No Pattern in this file replaces ops, only modifies or adds. Hence, the OpAdaptor can
    // be ignored which simplifies the implementation of processInputOperands().
    assert(llvm::equal(op.getOperands(), adaptor.getOperands()));
    processInputOperands(op.getArgOperandsMutable(), op, rewriter);

    // TODO: what about the results? Have to expect multiple results instead and add the needed
    // writes to the orginal array instance.
    assert(false && "TODO");
  }
};

LogicalResult splitArrayCreateInit(ModuleOp modOp) {
  MLIRContext *ctx = modOp.getContext();

  RewritePatternSet patterns(ctx);
  patterns.add<
      // clang-format off
      SplitInitFromCreate,
      SplitArrayInFuncDefOp,
      SplitArrayInReturnOp,
      SplitArrayInCallOp
      // clang-format off
      >(ctx);

  ConversionTarget target(*ctx);
  target.addLegalDialect<LLZKDialect, arith::ArithDialect, scf::SCFDialect>();
  target.addLegalOp<ModuleOp>();
  target.addDynamicallyLegalOp<CreateArrayOp>(SplitInitFromCreate::legal);
  target.addDynamicallyLegalOp<FuncOp>(SplitArrayInFuncDefOp::legal);
  target.addDynamicallyLegalOp<ReturnOp>(SplitArrayInReturnOp::legal);
  target.addDynamicallyLegalOp<CallOp>(SplitArrayInCallOp::legal);

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
