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

inline ArrayType splittableArray(ArrayType at) { return at.hasStaticShape() ? at : nullptr; }

inline ArrayType splittableArray(Type t) {
  if (ArrayType at = dyn_cast<ArrayType>(t)) {
    return splittableArray(at);
  } else {
    return nullptr;
  }
}

inline bool containsArrayType(Type t) {
  return t
      .walk([](ArrayType a) {
    return splittableArray(a) ? WalkResult::interrupt() : WalkResult::skip();
  }).wasInterrupted();
}

template <typename T> bool containsArrayType(ValueTypeRange<T> types) {
  for (Type t : types) {
    if (containsArrayType(t)) {
      return true;
    }
  }
  return false;
}

void splitArrayType(Type t, SmallVector<Type> &collect) {
  if (ArrayType at = splittableArray(t)) {
    int64_t n = at.getNumElements();
    assert(std::cmp_less_equal(n, std::numeric_limits<SmallVector<Type>::size_type>::max()));
    collect.append(n, at.getElementType());
  } else {
    collect.push_back(t);
  }
}

template <typename TypeCollection>
inline void splitArrayType(TypeCollection types, SmallVector<Type> &collect) {
  for (Type t : types) {
    splitArrayType(t, collect);
  }
}

template <typename TypeCollection> inline SmallVector<Type> splitArrayType(TypeCollection types) {
  SmallVector<Type> collect;
  splitArrayType(types, collect);
  return collect;
}

SmallVector<Value>
genIndexConstants(ArrayAttr index, Location loc, ConversionPatternRewriter &rewriter) {
  SmallVector<Value> operands;
  for (Attribute a : index) {
    // ASSERT: Attributes are index constants, created by ArrayType::getSubelementIndices().
    IntegerAttr ia = llvm::dyn_cast<IntegerAttr>(a);
    assert(ia && ia.getType().isIndex());
    operands.push_back(rewriter.create<arith::ConstantOp>(loc, ia));
  }
  return operands;
}

inline WriteArrayOp createWrite(
    Location loc, Value baseArrayOp, ArrayAttr index, Value init,
    ConversionPatternRewriter &rewriter
) {
  SmallVector<Value> readOperands = genIndexConstants(index, loc, rewriter);
  return rewriter.create<WriteArrayOp>(loc, baseArrayOp, ValueRange(readOperands), init);
}

CallOp newCallOpWithSplitResults(
    CallOp oldCall, CallOp::Adaptor adaptor, ConversionPatternRewriter &rewriter
) {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointAfter(oldCall);

  Operation::result_range oldResults = oldCall.getResults();
  CallOp newCall = rewriter.create<CallOp>(
      oldCall.getLoc(), splitArrayType(oldResults.getTypes()), oldCall.getCallee(),
      adaptor.getArgOperands()
  );

  auto newResults = newCall.getResults().begin();
  for (Value oldVal : oldResults) {
    if (ArrayType at = splittableArray(oldVal.getType())) {
      Location loc = oldVal.getLoc();
      // Generate `CreateArrayOp` and replace uses of the result with it.
      auto newArray = rewriter.create<CreateArrayOp>(loc, at);
      rewriter.replaceAllUsesWith(oldVal, newArray);

      // For all indices in the ArrayType (i.e. the element count), write the next
      // result from the new CallOp to the new array.
      std::optional<SmallVector<ArrayAttr>> allIndices = at.getSubelementIndices();
      assert(allIndices); // follows from legal() check
      assert(std::cmp_equal(allIndices->size(), at.getNumElements()));
      for (ArrayAttr subIdx : allIndices.value()) {
        createWrite(loc, newArray, subIdx, *newResults, rewriter);
        newResults++;
      }
    } else {
      newResults++;
    }
  }
  // erase the original CallOp
  rewriter.eraseOp(oldCall);

  return newCall;
}

void processBlockArgs(Block &entryBlock, ConversionPatternRewriter &rewriter) {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(&entryBlock);

  for (unsigned i = 0; i < entryBlock.getNumArguments();) {
    Value oldV = entryBlock.getArgument(i);
    if (ArrayType at = splittableArray(oldV.getType())) {
      Location loc = oldV.getLoc();
      // Generate `CreateArrayOp` and replace uses of the argument with it.
      auto newArray = rewriter.create<CreateArrayOp>(loc, at);
      rewriter.replaceAllUsesWith(oldV, newArray);
      // Remove the argument from the block
      entryBlock.eraseArgument(i);
      // For all indices in the ArrayType (i.e. the element count), generate a new block
      // argument and a write of that argument to the new array.
      std::optional<SmallVector<ArrayAttr>> allIndices = at.getSubelementIndices();
      assert(allIndices); // follows from legal() check
      assert(std::cmp_equal(allIndices->size(), at.getNumElements()));
      for (ArrayAttr subIdx : allIndices.value()) {
        BlockArgument newArg = entryBlock.insertArgument(i, at.getElementType(), loc);
        createWrite(loc, newArray, subIdx, newArg, rewriter);
        ++i;
      }
    } else {
      ++i;
    }
  }
}

inline ReadArrayOp
createRead(Location loc, Value baseArrayOp, ArrayAttr index, ConversionPatternRewriter &rewriter) {
  SmallVector<Value> readOperands = genIndexConstants(index, loc, rewriter);
  return rewriter.create<ReadArrayOp>(loc, baseArrayOp, ValueRange(readOperands));
}

void processInputOperand(
    Location loc, Value operand, SmallVector<Value> &newOperands,
    ConversionPatternRewriter &rewriter
) {
  if (ArrayType at = splittableArray(operand.getType())) {
    std::optional<SmallVector<ArrayAttr>> indices = at.getSubelementIndices();
    assert(indices.has_value() && "passed earlier hasStaticShape() check");
    for (ArrayAttr index : indices.value()) {
      newOperands.push_back(createRead(loc, operand, index, rewriter));
    }
  } else {
    newOperands.push_back(operand);
  }
}

// For each operand with ArrayType, add N reads from the array and use those N values instead.
void processInputOperands(
    ValueRange operands, MutableOperandRange outputOpRef, Operation *op,
    ConversionPatternRewriter &rewriter
) {
  SmallVector<Value> newOperands;
  for (Value v : operands) {
    processInputOperand(op->getLoc(), v, newOperands, rewriter);
  }
  rewriter.modifyOpInPlace(op, [&outputOpRef, &newOperands]() {
    outputOpRef.assign(ValueRange(newOperands));
  });
}

class SplitInitFromCreate : public OpConversionPattern<CreateArrayOp> {
public:
  using OpConversionPattern<CreateArrayOp>::OpConversionPattern;

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

class SplitArrayInFuncDefOp : public OpConversionPattern<FuncOp> {
public:
  using OpConversionPattern<FuncOp>::OpConversionPattern;

  inline static bool legal(FuncOp op) { return !containsArrayType(op.getFunctionType()); }

  LogicalResult match(FuncOp op) const override { return failure(legal(op)); }

  void rewrite(FuncOp op, OpAdaptor, ConversionPatternRewriter &rewriter) const override {
    // Update in/out types of the function to replace arrays with scalars
    FunctionType oldTy = op.getFunctionType();
    SmallVector<Type> newInputs = splitArrayType(oldTy.getInputs());
    SmallVector<Type> newOutputs = splitArrayType(oldTy.getResults());
    FunctionType newTy =
        FunctionType::get(oldTy.getContext(), TypeRange(newInputs), TypeRange(newOutputs));
    if (newTy == oldTy) {
      return; // nothing to change
    }
    rewriter.modifyOpInPlace(op, [&op, &newTy]() { op.setFunctionType(newTy); });

    // If the function has a body, ensure the entry block arguments match the function inputs.
    if (Region *body = op.getCallableRegion()) {
      Block &entryBlock = body->front();
      if (std::cmp_equal(entryBlock.getNumArguments(), newInputs.size())) {
        return; // nothing to change
      }
      processBlockArgs(entryBlock, rewriter);
    }
  }
};

class SplitArrayInReturnOp : public OpConversionPattern<ReturnOp> {
public:
  using OpConversionPattern<ReturnOp>::OpConversionPattern;

  inline static bool legal(ReturnOp op) { return !containsArrayType(op.getOperands().getTypes()); }

  LogicalResult match(ReturnOp op) const override { return failure(legal(op)); }

  void rewrite(ReturnOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    processInputOperands(adaptor.getOperands(), op.getOperandsMutable(), op, rewriter);
  }
};

class SplitArrayInCallOp : public OpConversionPattern<CallOp> {
public:
  using OpConversionPattern<CallOp>::OpConversionPattern;

  inline static bool legal(CallOp op) {
    return !containsArrayType(op.getArgOperands().getTypes()) &&
           !containsArrayType(op.getResultTypes());
  }

  LogicalResult match(CallOp op) const override { return failure(legal(op)); }

  void rewrite(CallOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    assert(isNullOrEmpty(op.getMapOpGroupSizesAttr()) && "structs must be previously flattened");

    // Create new CallOp with split results first so, then process its inputs to split types
    CallOp newCall = newCallOpWithSplitResults(op, adaptor, rewriter);
    processInputOperands(
        newCall.getArgOperands(), newCall.getArgOperandsMutable(), newCall, rewriter
    );
  }
};

class ReplaceKnownArrayLengthOp : public OpConversionPattern<ArrayLengthOp> {
public:
  using OpConversionPattern<ArrayLengthOp>::OpConversionPattern;

  /// If 'dimIdx' is constant and that dimension of the ArrayType has static size, return it.
  static std::optional<llvm::APInt> getDimSizeIfKnown(Value dimIdx, ArrayType baseArrType) {
    if (splittableArray(baseArrType)) {
      llvm::APInt idxAP;
      if (mlir::matchPattern(dimIdx, mlir::m_ConstantInt(&idxAP))) {
        uint64_t idx64 = idxAP.getZExtValue();
        assert(std::cmp_less_equal(idx64, std::numeric_limits<size_t>::max()));
        Attribute dimSizeAttr = baseArrType.getDimensionSizes()[static_cast<size_t>(idx64)];
        if (mlir::matchPattern(dimSizeAttr, mlir::m_ConstantInt(&idxAP))) {
          return idxAP;
        }
      }
    }
    return std::nullopt;
  }

  inline static bool legal(ArrayLengthOp op) {
    // rewrite() can only work with constant dim size, i.e. must consider it legal otherwise
    return !getDimSizeIfKnown(op.getDim(), op.getArrRefType()).has_value();
  }

  LogicalResult match(ArrayLengthOp op) const override { return failure(legal(op)); }

  void
  rewrite(ArrayLengthOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    ArrayType arrTy = dyn_cast<ArrayType>(adaptor.getArrRef().getType());
    assert(arrTy); // must have array type per ODS spec of ArrayLengthOp
    std::optional<llvm::APInt> len = getDimSizeIfKnown(adaptor.getDim(), arrTy);
    assert(len.has_value()); // follows from legal() check
    rewriter.replaceOpWithNewOp<arith::ConstantIndexOp>(op, llzk::fromAPInt(len.value()));
  }
};

/// field name and type
using FieldInfo = std::pair<StringAttr, Type>;
/// ArrayAttr index -> scalar field info
using LocalFieldReplacementMap = DenseMap<ArrayAttr, FieldInfo>;
/// struct -> array-type field name -> LocalFieldReplacementMap
using FieldReplacementMap = DenseMap<StructDefOp, DenseMap<StringAttr, LocalFieldReplacementMap>>;

class SplitArrayInFieldDefOp : public OpConversionPattern<FieldDefOp> {
  SymbolTableCollection &tables;
  FieldReplacementMap &repMapRef;

public:
  SplitArrayInFieldDefOp(
      MLIRContext *ctx, SymbolTableCollection &symTables, FieldReplacementMap &fieldRepMap
  )
      : OpConversionPattern<FieldDefOp>(ctx), tables(symTables), repMapRef(fieldRepMap) {}

  inline static bool legal(FieldDefOp op) { return !containsArrayType(op.getType()); }

  LogicalResult match(FieldDefOp op) const override { return failure(legal(op)); }

  void rewrite(FieldDefOp op, OpAdaptor, ConversionPatternRewriter &rewriter) const override {
    StructDefOp inStruct = op->getParentOfType<StructDefOp>();
    assert(inStruct);
    LocalFieldReplacementMap &localRepMapRef = repMapRef[inStruct][op.getSymNameAttr()];

    ArrayType arrTy = dyn_cast<ArrayType>(op.getType());
    assert(arrTy); // follows from legal() check
    auto subIdxs = arrTy.getSubelementIndices();
    assert(subIdxs.has_value());
    Type elemTy = arrTy.getElementType();

    SymbolTable &structSymbolTable = tables.getSymbolTable(inStruct);
    for (ArrayAttr idx : subIdxs.value()) {
      // Create scalar version of the field
      FieldDefOp newField =
          rewriter.create<FieldDefOp>(op.getLoc(), op.getSymNameAttr(), elemTy, op.getColumn());
      // Use SymbolTable to give it a unique name and store to the replacement map
      localRepMapRef[idx] = std::make_pair(structSymbolTable.insert(newField), elemTy);
    }
    rewriter.eraseOp(op);
  }
};

template <typename ImplClass, typename FieldRefOpType, typename GenPrefixType>
class SplitArrayInFieldRefOp : public OpConversionPattern<FieldRefOpType> {
  SymbolTableCollection &tables;
  const FieldReplacementMap &repMapRef;

  inline static void ensureImplementedAtCompile() {
    static_assert(
        sizeof(FieldRefOpType) == 0, "SplitArrayInFieldRefOp not implemented for requested type."
    );
  }

protected:
  using OpAdaptor = typename FieldRefOpType::Adaptor;

  static GenPrefixType genPrefix(FieldRefOpType, ConversionPatternRewriter &) {
    ensureImplementedAtCompile();
  }

  static void
  forIndex(Location, GenPrefixType, ArrayAttr, FieldInfo, OpAdaptor, ConversionPatternRewriter &) {
    ensureImplementedAtCompile();
  }

public:
  SplitArrayInFieldRefOp(
      MLIRContext *ctx, SymbolTableCollection &symTables, const FieldReplacementMap &fieldRepMap
  )
      : OpConversionPattern<FieldRefOpType>(ctx), tables(symTables), repMapRef(fieldRepMap) {}

  static bool legal(FieldRefOpType) { ensureImplementedAtCompile(); }

  LogicalResult match(FieldRefOpType op) const override { return failure(ImplClass::legal(op)); }

  void rewrite(FieldRefOpType op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter)
      const override {
    StructType tgtStructTy = llvm::cast<FieldRefOpInterface>(op.getOperation()).getStructType();
    assert(tgtStructTy);
    auto tgtStructDef = tgtStructTy.getDefinition(tables, op);
    assert(succeeded(tgtStructDef));

    GenPrefixType prefixResult = ImplClass::genPrefix(op, rewriter);

    const LocalFieldReplacementMap &idxToName =
        repMapRef.at(tgtStructDef->get()).at(op.getFieldNameAttr().getAttr());
    // Split the array field write into a series of read array + write scalar field
    for (auto [idx, newField] : idxToName) {
      ImplClass::forIndex(op.getLoc(), prefixResult, idx, newField, adaptor, rewriter);
    }
    rewriter.eraseOp(op);
  }
};

class SplitArrayInFieldWriteOp
    : public SplitArrayInFieldRefOp<SplitArrayInFieldWriteOp, FieldWriteOp, void *> {
public:
  using SplitArrayInFieldRefOp<
      SplitArrayInFieldWriteOp, FieldWriteOp, void *>::SplitArrayInFieldRefOp;

  static bool legal(FieldWriteOp op) { return !containsArrayType(op.getVal().getType()); }

  static void *genPrefix(FieldWriteOp, ConversionPatternRewriter &) { return nullptr; }

  static void forIndex(
      Location loc, void *, ArrayAttr idx, FieldInfo newField, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter
  ) {
    ReadArrayOp scalarRead = createRead(loc, adaptor.getVal(), idx, rewriter);
    rewriter.create<FieldWriteOp>(
        loc, adaptor.getComponent(), FlatSymbolRefAttr::get(newField.first), scalarRead
    );
  }
};

class SplitArrayInFieldReadOp
    : public SplitArrayInFieldRefOp<SplitArrayInFieldReadOp, FieldReadOp, Value> {
public:
  using SplitArrayInFieldRefOp<SplitArrayInFieldReadOp, FieldReadOp, Value>::SplitArrayInFieldRefOp;

  static bool legal(FieldReadOp op) { return !containsArrayType(op.getResult().getType()); }

  static Value genPrefix(FieldReadOp op, ConversionPatternRewriter &rewriter) {
    CreateArrayOp newArray =
        rewriter.create<CreateArrayOp>(op.getLoc(), llvm::cast<ArrayType>(op.getType()));
    rewriter.replaceAllUsesWith(op, newArray);
    return newArray;
  }

  static void forIndex(
      Location loc, Value newArray, ArrayAttr idx, FieldInfo newField, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter
  ) {
    FieldReadOp scalarRead =
        rewriter.create<FieldReadOp>(loc, newField.second, adaptor.getComponent(), newField.first);
    createWrite(loc, newArray, idx, scalarRead, rewriter);
  }
};

LogicalResult
step1(ModuleOp modOp, SymbolTableCollection &symTables, FieldReplacementMap &fieldRepMap) {
  MLIRContext *ctx = modOp.getContext();

  RewritePatternSet patterns(ctx);

  patterns.add<SplitArrayInFieldDefOp>(ctx, symTables, fieldRepMap);

  ConversionTarget target(*ctx);
  target.addLegalDialect<LLZKDialect, arith::ArithDialect, scf::SCFDialect>();
  target.addLegalOp<ModuleOp>();
  target.addDynamicallyLegalOp<FieldDefOp>(SplitArrayInFieldDefOp::legal);

  return applyFullConversion(modOp, target, std::move(patterns));
}

LogicalResult
step2(ModuleOp modOp, SymbolTableCollection &symTables, const FieldReplacementMap &fieldRepMap) {
  MLIRContext *ctx = modOp.getContext();

  RewritePatternSet patterns(ctx);
  patterns.add<
      // clang-format off
      SplitInitFromCreate,
      SplitArrayInFuncDefOp,
      SplitArrayInReturnOp,
      SplitArrayInCallOp,
      ReplaceKnownArrayLengthOp
      // clang-format on
      >(ctx);

  patterns.add<
      // clang-format off
      SplitArrayInFieldWriteOp,
      SplitArrayInFieldReadOp
      // clang-format on
      >(ctx, symTables, fieldRepMap);

  ConversionTarget target(*ctx);
  target.addLegalDialect<LLZKDialect, arith::ArithDialect, scf::SCFDialect>();
  target.addLegalOp<ModuleOp>();
  target.addDynamicallyLegalOp<CreateArrayOp>(SplitInitFromCreate::legal);
  target.addDynamicallyLegalOp<FuncOp>(SplitArrayInFuncDefOp::legal);
  target.addDynamicallyLegalOp<ReturnOp>(SplitArrayInReturnOp::legal);
  target.addDynamicallyLegalOp<CallOp>(SplitArrayInCallOp::legal);
  target.addDynamicallyLegalOp<ArrayLengthOp>(ReplaceKnownArrayLengthOp::legal);
  target.addDynamicallyLegalOp<FieldWriteOp>(SplitArrayInFieldWriteOp::legal);
  target.addDynamicallyLegalOp<FieldReadOp>(SplitArrayInFieldReadOp::legal);

  return applyFullConversion(modOp, target, std::move(patterns));
}

LogicalResult splitArrayCreateInit(ModuleOp modOp) {
  SymbolTableCollection symTables;
  FieldReplacementMap fieldRepMap;

  // This is divided into 2 steps to simplify the implementation for field-related ops. The issue is
  // that the conversions for field read/write expect the mapping of array index to field name+type
  // to already be populated for the referenced field (although this could be computed on demand if
  // desired but it complicates the implementation a bit).
  if (failed(step1(modOp, symTables, fieldRepMap))) {
    return failure();
  }
  return step2(modOp, symTables, fieldRepMap);
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
