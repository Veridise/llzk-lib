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
/// The steps of this transformation are as follows:
///
/// 1. Run a dialect conversion that replaces `ArrayType` fields with `N` scalar fields.
///
/// 2. Run a dialect conversion that does the following:
///
///    - Replace `FieldReadOp` and `FieldWriteOp` targeting the fields that were split in step 1 so
///      they instead perform scalar reads and writes from the new fields. The transformation is
///      local to the current op. Therefore, when replacing the `FieldReadOp` a new array is created
///      locally and all uses of the `FieldReadOp` are replaced with the new array Value, then each
///      scalar field read is followed by scalar write into the new array. Similarly, when replacing
///      a `FieldWriteOp`, each element in the array operand needs a scalar read from the array
///      followed by a scalar write to the new field. Making only local changes keeps this step
///      simple and later steps will optimize.
///
///    - Replace `ArrayLengthOp` with the constant size of the selected dimension.
///
///    - Remove element initialization from `CreateArrayOp` and instead insert a list of
///      `WriteArrayOp` immediately following.
///
///    - Desugar `InsertArrayOp` and `ExtractArrayOp` into their element-wise scalar reads/writes.
///
///    - Split arrays to scalars in `FuncOp`, `CallOp`, and `ReturnOp` and insert the necessary
///      create/read/write ops so the changes are as local as possible (just as described for
///      `FieldReadOp` and `FieldWriteOp`)
///
/// 3. Run MLIR "sroa" pass to split each array with linear size `N` into `N` arrays of size 1 (to
///    prepare for "mem2reg" pass because it's API does not allow for indexing to split aggregates).
///
/// 4. Run MLIR "mem2reg" pass to convert all of the size 1 array allocation and access into SSA
///    values. This pass also runs several standard optimizations so the final result is condensed.
///
/// Note: This transformation imposes a "last write wins" semantics on array elements. If
/// different/configurable semantics are added in the future, some additional transformation would
/// be necessary before/during this pass so that multiple writes to the same index can be handled
/// properly while they still exist.
///
/// Note: This transformation will introduce an undef op when there exists a read from an array
/// index that was not earlier written to.
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

/// If the given ArrayType can be split into scalars, return it, otherwise nullptr.
inline ArrayType splittableArray(ArrayType at) { return at.hasStaticShape() ? at : nullptr; }

/// If the given Type is an ArrayType that can be split into scalars, return it, otherwise nullptr.
inline ArrayType splittableArray(Type t) {
  if (ArrayType at = dyn_cast<ArrayType>(t)) {
    return splittableArray(at);
  } else {
    return nullptr;
  }
}

/// Return `true` iff the given type is or contains an ArrayType that can be split into scalars.
inline bool containsSplittableArrayType(Type t) {
  return t
      .walk([](ArrayType a) {
    return splittableArray(a) ? WalkResult::interrupt() : WalkResult::skip();
  }).wasInterrupted();
}

/// Return `true` iff the given range contains any ArrayType that can be split into scalars.
template <typename T> bool containsSplittableArrayType(ValueTypeRange<T> types) {
  for (Type t : types) {
    if (containsSplittableArrayType(t)) {
      return true;
    }
  }
  return false;
}

/// If the given Type is an ArrayType that can be split into scalars, append `collect` with all of
/// the scalar types that result from splitting the ArrayType. Otherwise, just push the `Type`.
void splitArrayTypeTo(Type t, SmallVector<Type> &collect) {
  if (ArrayType at = splittableArray(t)) {
    int64_t n = at.getNumElements();
    assert(std::cmp_less_equal(n, std::numeric_limits<SmallVector<Type>::size_type>::max()));
    collect.append(n, at.getElementType());
  } else {
    collect.push_back(t);
  }
}

/// For each Type in the given input collection, call `splitArrayTypeTo(Type,...)`.
template <typename TypeCollection>
inline void splitArrayTypeTo(TypeCollection types, SmallVector<Type> &collect) {
  for (Type t : types) {
    splitArrayTypeTo(t, collect);
  }
}

/// Return a list such that each scalar Type is directly added to the list but for each splittable
/// ArrayType, the proper number of scalar element types are added instead.
template <typename TypeCollection> inline SmallVector<Type> splitArrayType(TypeCollection types) {
  SmallVector<Type> collect;
  splitArrayTypeTo(types, collect);
  return collect;
}

/// Generate `arith::ConstantOp` at the current position of the `rewriter` for each int attribute in
/// the ArrayAttr.
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

inline WriteArrayOp genWrite(
    Location loc, Value baseArrayOp, ArrayAttr index, Value init,
    ConversionPatternRewriter &rewriter
) {
  SmallVector<Value> readOperands = genIndexConstants(index, loc, rewriter);
  return rewriter.create<WriteArrayOp>(loc, baseArrayOp, ValueRange(readOperands), init);
}

/// Replace the given CallOp with a new one where any ArrayType in the results are split into their
/// scalar elements. Also, after the CallOp, generate a CreateArrayOp for each ArrayType result and
/// generate writes from the corresponding scalar result values to the new array.
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
        genWrite(loc, newArray, subIdx, *newResults, rewriter);
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

/// For each argument to the Block that has a splittable ArrayType, replace it with the necessary
/// number of scalar arguments, generate a CreateArrayOp, and generate writes from the new block
/// scalar arguments to the new array. All users of the original block argument are updated to
/// target the result of the CreateArrayOp.
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
        genWrite(loc, newArray, subIdx, newArg, rewriter);
        ++i;
      }
    } else {
      ++i;
    }
  }
}

inline ReadArrayOp
genRead(Location loc, Value baseArrayOp, ArrayAttr index, ConversionPatternRewriter &rewriter) {
  SmallVector<Value> readOperands = genIndexConstants(index, loc, rewriter);
  return rewriter.create<ReadArrayOp>(loc, baseArrayOp, ValueRange(readOperands));
}

// If the operand has ArrayType, add N reads from the array to the `newOperands` list otherwise add
// the original operand to the list.
void processInputOperand(
    Location loc, Value operand, SmallVector<Value> &newOperands,
    ConversionPatternRewriter &rewriter
) {
  if (ArrayType at = splittableArray(operand.getType())) {
    std::optional<SmallVector<ArrayAttr>> indices = at.getSubelementIndices();
    assert(indices.has_value() && "passed earlier hasStaticShape() check");
    for (ArrayAttr index : indices.value()) {
      newOperands.push_back(genRead(loc, operand, index, rewriter));
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

namespace {

enum Direction {
  /// Copying a smaller array into a larger one, i.e. `InsertArrayOp`
  SMALL_TO_LARGE,
  /// Copying a larger array into a smaller one, i.e. `ExtractArrayOp`
  LARGE_TO_SMALL,
};

/// Common implementation for handling `InsertArrayOp` and `ExtractArrayOp`. For all indices in the
/// given ArrayType, perform writes from one array to the other, in the specified Direction.
template <Direction dir>
inline void rewriteImpl(
    ArrayAccessOpInterface op, ArrayType smallType, Value smallArr, Value largeArr,
    ConversionPatternRewriter &rewriter
) {
  assert(smallType); // follows from legal() check
  Location loc = op.getLoc();
  MLIRContext *ctx = op.getContext();

  ArrayAttr indexAsAttr = op.indexOperandsToAttributeArray();
  assert(indexAsAttr); // follows from legal() check

  // For all indices in the ArrayType (i.e. the element count), read from one array into the other
  // (depending on direction flag).
  std::optional<SmallVector<ArrayAttr>> subIndices = smallType.getSubelementIndices();
  assert(subIndices); // follows from legal() check
  assert(std::cmp_equal(subIndices->size(), smallType.getNumElements()));
  for (ArrayAttr indexingTail : subIndices.value()) {
    SmallVector<Attribute> joined;
    joined.append(indexAsAttr.begin(), indexAsAttr.end());
    joined.append(indexingTail.begin(), indexingTail.end());
    ArrayAttr fullIndex = ArrayAttr::get(ctx, joined);

    if constexpr (dir == Direction::SMALL_TO_LARGE) {
      auto init = genRead(loc, smallArr, indexingTail, rewriter);
      genWrite(loc, largeArr, fullIndex, init, rewriter);
    } else if constexpr (dir == Direction::LARGE_TO_SMALL) {
      auto init = genRead(loc, largeArr, fullIndex, rewriter);
      genWrite(loc, smallArr, indexingTail, init, rewriter);
    }
  }
}

} // namespace

class SplitInsertArrayOp : public OpConversionPattern<InsertArrayOp> {
public:
  using OpConversionPattern<InsertArrayOp>::OpConversionPattern;

  static bool legal(InsertArrayOp op) {
    return !containsSplittableArrayType(op.getRvalue().getType());
  }

  LogicalResult match(InsertArrayOp op) const override { return failure(legal(op)); }

  void
  rewrite(InsertArrayOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    ArrayType at = splittableArray(op.getRvalue().getType());
    rewriteImpl<SMALL_TO_LARGE>(
        llvm::cast<ArrayAccessOpInterface>(op.getOperation()), at, adaptor.getRvalue(),
        adaptor.getArrRef(), rewriter
    );
    rewriter.eraseOp(op);
  }
};

class SplitExtractArrayOp : public OpConversionPattern<ExtractArrayOp> {
public:
  using OpConversionPattern<ExtractArrayOp>::OpConversionPattern;

  static bool legal(ExtractArrayOp op) {
    return !containsSplittableArrayType(op.getResult().getType());
  }

  LogicalResult match(ExtractArrayOp op) const override { return failure(legal(op)); }

  void rewrite(ExtractArrayOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter)
      const override {
    ArrayType at = splittableArray(op.getResult().getType());
    // Generate `CreateArrayOp` in place of the current op.
    auto newArray = rewriter.replaceOpWithNewOp<CreateArrayOp>(op, at);
    rewriteImpl<LARGE_TO_SMALL>(
        llvm::cast<ArrayAccessOpInterface>(op.getOperation()), at, newArray, adaptor.getArrRef(),
        rewriter
    );
  }
};

class SplitInitFromCreateArrayOp : public OpConversionPattern<CreateArrayOp> {
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

  inline static bool legal(FuncOp op) { return !containsSplittableArrayType(op.getFunctionType()); }

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

  inline static bool legal(ReturnOp op) {
    return !containsSplittableArrayType(op.getOperands().getTypes());
  }

  LogicalResult match(ReturnOp op) const override { return failure(legal(op)); }

  void rewrite(ReturnOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    processInputOperands(adaptor.getOperands(), op.getOperandsMutable(), op, rewriter);
  }
};

class SplitArrayInCallOp : public OpConversionPattern<CallOp> {
public:
  using OpConversionPattern<CallOp>::OpConversionPattern;

  inline static bool legal(CallOp op) {
    return !containsSplittableArrayType(op.getArgOperands().getTypes()) &&
           !containsSplittableArrayType(op.getResultTypes());
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

  inline static bool legal(FieldDefOp op) { return !containsSplittableArrayType(op.getType()); }

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

/// Common implementation for handling `FieldWriteOp` and `FieldReadOp`. The `forIndex()` function
/// should be implemented to generate the necessary code for each index of the referenced array. The
/// `genHeader()` function is called before
///
/// @tparam ImplClass       the concrete subclass
/// @tparam FieldRefOpType  the concrete op class
/// @tparam GenHeaderType   return type of `genHeader()`, used to pass data to `forIndex()`
template <typename ImplClass, typename FieldRefOpType, typename GenHeaderType>
class SplitArrayInFieldRefOp : public OpConversionPattern<FieldRefOpType> {
  SymbolTableCollection &tables;
  const FieldReplacementMap &repMapRef;

  // static check to ensure the functions are implemented in all subclasses
  inline static void ensureImplementedAtCompile() {
    static_assert(
        sizeof(FieldRefOpType) == 0, "SplitArrayInFieldRefOp not implemented for requested type."
    );
  }

protected:
  using OpAdaptor = typename FieldRefOpType::Adaptor;

  /// Executed at the start of `rewrite()` to (optionally) generate anything that should be before
  /// the element-wise operations that will be added by `forIndex()`.
  static GenHeaderType genHeader(FieldRefOpType, ConversionPatternRewriter &) {
    ensureImplementedAtCompile();
    assert(false && "unreachable");
  }

  /// Executed for each multi-dimensional array index in the ArrayType of the original field to
  /// generate the element-wise scalar operations on the new scalar fields.
  static void
  forIndex(Location, GenHeaderType, ArrayAttr, FieldInfo, OpAdaptor, ConversionPatternRewriter &) {
    ensureImplementedAtCompile();
    assert(false && "unreachable");
  }

public:
  SplitArrayInFieldRefOp(
      MLIRContext *ctx, SymbolTableCollection &symTables, const FieldReplacementMap &fieldRepMap
  )
      : OpConversionPattern<FieldRefOpType>(ctx), tables(symTables), repMapRef(fieldRepMap) {}

  static bool legal(FieldRefOpType) {
    ensureImplementedAtCompile();
    assert(false && "unreachable");
  }

  LogicalResult match(FieldRefOpType op) const override { return failure(ImplClass::legal(op)); }

  void rewrite(FieldRefOpType op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter)
      const override {
    StructType tgtStructTy = llvm::cast<FieldRefOpInterface>(op.getOperation()).getStructType();
    assert(tgtStructTy);
    auto tgtStructDef = tgtStructTy.getDefinition(tables, op);
    assert(succeeded(tgtStructDef));

    GenHeaderType prefixResult = ImplClass::genHeader(op, rewriter);

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

  static bool legal(FieldWriteOp op) { return !containsSplittableArrayType(op.getVal().getType()); }

  static void *genHeader(FieldWriteOp, ConversionPatternRewriter &) { return nullptr; }

  static void forIndex(
      Location loc, void *, ArrayAttr idx, FieldInfo newField, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter
  ) {
    ReadArrayOp scalarRead = genRead(loc, adaptor.getVal(), idx, rewriter);
    rewriter.create<FieldWriteOp>(
        loc, adaptor.getComponent(), FlatSymbolRefAttr::get(newField.first), scalarRead
    );
  }
};

class SplitArrayInFieldReadOp
    : public SplitArrayInFieldRefOp<SplitArrayInFieldReadOp, FieldReadOp, CreateArrayOp> {
public:
  using SplitArrayInFieldRefOp<
      SplitArrayInFieldReadOp, FieldReadOp, CreateArrayOp>::SplitArrayInFieldRefOp;

  static bool legal(FieldReadOp op) {
    return !containsSplittableArrayType(op.getResult().getType());
  }

  static CreateArrayOp genHeader(FieldReadOp op, ConversionPatternRewriter &rewriter) {
    CreateArrayOp newArray =
        rewriter.create<CreateArrayOp>(op.getLoc(), llvm::cast<ArrayType>(op.getType()));
    rewriter.replaceAllUsesWith(op, newArray);
    return newArray;
  }

  static void forIndex(
      Location loc, CreateArrayOp newArray, ArrayAttr idx, FieldInfo newField, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter
  ) {
    FieldReadOp scalarRead =
        rewriter.create<FieldReadOp>(loc, newField.second, adaptor.getComponent(), newField.first);
    genWrite(loc, newArray, idx, scalarRead, rewriter);
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

  LLVM_DEBUG(llvm::dbgs() << "Begin step 1: split array fields\n";);
  return applyFullConversion(modOp, target, std::move(patterns));
}

LogicalResult
step2(ModuleOp modOp, SymbolTableCollection &symTables, const FieldReplacementMap &fieldRepMap) {
  MLIRContext *ctx = modOp.getContext();

  RewritePatternSet patterns(ctx);
  patterns.add<
      // clang-format off
      SplitInitFromCreateArrayOp,
      SplitInsertArrayOp,
      SplitExtractArrayOp,
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
  target.addDynamicallyLegalOp<CreateArrayOp>(SplitInitFromCreateArrayOp::legal);
  target.addDynamicallyLegalOp<InsertArrayOp>(SplitInsertArrayOp::legal);
  target.addDynamicallyLegalOp<ExtractArrayOp>(SplitExtractArrayOp::legal);
  target.addDynamicallyLegalOp<FuncOp>(SplitArrayInFuncDefOp::legal);
  target.addDynamicallyLegalOp<ReturnOp>(SplitArrayInReturnOp::legal);
  target.addDynamicallyLegalOp<CallOp>(SplitArrayInCallOp::legal);
  target.addDynamicallyLegalOp<ArrayLengthOp>(ReplaceKnownArrayLengthOp::legal);
  target.addDynamicallyLegalOp<FieldWriteOp>(SplitArrayInFieldWriteOp::legal);
  target.addDynamicallyLegalOp<FieldReadOp>(SplitArrayInFieldReadOp::legal);

  LLVM_DEBUG(llvm::dbgs() << "Begin step 2: update/split other array ops\n";);
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
