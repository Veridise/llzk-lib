//===-- Ops.cpp - Array operation implementations ---------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/Array/IR/Ops.h"
#include "llzk/Util/BuilderHelper.h"

#include <mlir/Dialect/Utils/IndexingUtils.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LogicalResult.h>

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/StringSet.h>
#include <llvm/ADT/Twine.h>

// TableGen'd implementation files
#define GET_OP_CLASSES
#include "llzk/Dialect/Array/IR/OpInterfaces.cpp.inc"

// TableGen'd implementation files
#define GET_OP_CLASSES
#include "llzk/Dialect/Array/IR/Ops.cpp.inc"

namespace llzk::array {

using namespace mlir;

//===------------------------------------------------------------------===//
// CreateArrayOp
//===------------------------------------------------------------------===//

void CreateArrayOp::build(
    OpBuilder &odsBuilder, OperationState &odsState, ArrayType result, ValueRange elements
) {
  odsState.addTypes(result);
  odsState.addOperands(elements);
  // This builds CreateArrayOp from a list of elements. In that case, the dimensions of the array
  // type cannot be defined via an affine map which means there are no affine map operands.
  affineMapHelpers::buildInstantiationAttrsEmpty<CreateArrayOp>(
      odsBuilder, odsState, static_cast<int32_t>(elements.size())
  );
}

void CreateArrayOp::build(
    OpBuilder &odsBuilder, OperationState &odsState, ArrayType result,
    ArrayRef<ValueRange> mapOperands, DenseI32ArrayAttr numDimsPerMap
) {
  odsState.addTypes(result);
  affineMapHelpers::buildInstantiationAttrs<CreateArrayOp>(
      odsBuilder, odsState, mapOperands, numDimsPerMap
  );
}

LogicalResult CreateArrayOp::verifySymbolUses(SymbolTableCollection &tables) {
  // Ensure any SymbolRef used in the type are valid
  return verifyTypeResolution(tables, *this, llvm::cast<Type>(getType()));
}

void CreateArrayOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  setNameFn(getResult(), "array");
}

llvm::SmallVector<Type> CreateArrayOp::resultTypeToElementsTypes(Type resultType) {
  // The ODS restricts $result with LLZK_ArrayType so this cast is safe.
  ArrayType a = llvm::cast<ArrayType>(resultType);
  return llvm::SmallVector<Type>(a.getNumElements(), a.getElementType());
}

ParseResult CreateArrayOp::parseInferredArrayType(
    OpAsmParser &parser, llvm::SmallVector<Type, 1> &elementsTypes,
    ArrayRef<OpAsmParser::UnresolvedOperand> elements, Type resultType
) {
  assert(elementsTypes.size() == 0); // it was not yet initialized
  // If the '$elements' operand is not empty, then the expected type for the operand
  //  is computed to match the type of the '$result'. Otherwise, it remains empty.
  if (elements.size() > 0) {
    elementsTypes.append(resultTypeToElementsTypes(resultType));
  }
  return success();
}

void CreateArrayOp::printInferredArrayType(
    OpAsmPrinter &printer, CreateArrayOp, TypeRange, OperandRange, Type
) {
  // nothing to print, it's derived and therefore not represented in the output
}

LogicalResult CreateArrayOp::verify() {
  Type retTy = getResult().getType();
  assert(llvm::isa<ArrayType>(retTy)); // per ODS spec of CreateArrayOp

  // Collect the array dimensions that are defined via AffineMapAttr
  SmallVector<AffineMapAttr> mapAttrs;
  for (Attribute a : llvm::cast<ArrayType>(retTy).getDimensionSizes()) {
    if (AffineMapAttr m = dyn_cast<AffineMapAttr>(a)) {
      mapAttrs.push_back(m);
    }
  }
  return affineMapHelpers::verifyAffineMapInstantiations(
      getMapOperands(), getNumDimsPerMap(), mapAttrs, *this
  );
}

/// Required by DestructurableAllocationOpInterface / SROA pass
SmallVector<DestructurableMemorySlot> CreateArrayOp::getDestructurableSlots() {
  assert(getElements().empty() && "must run after initialization is split from allocation");
  ArrayType arrType = getType();
  if (!arrType.hasStaticShape() || arrType.getNumElements() == 1) {
    return {};
  }
  if (auto destructured = arrType.getSubelementIndexMap()) {
    return {DestructurableMemorySlot {{getResult(), arrType}, std::move(*destructured)}};
  }
  return {};
}

/// Required by DestructurableAllocationOpInterface / SROA pass
DenseMap<Attribute, MemorySlot> CreateArrayOp::destructure(
    const DestructurableMemorySlot &slot, const SmallPtrSetImpl<Attribute> &usedIndices,
    RewriterBase &rewriter
) {
  assert(slot.ptr == getResult());
  assert(slot.elemType == getType());

  rewriter.setInsertionPointAfter(*this);

  DenseMap<Attribute, MemorySlot> slotMap; // result
  for (Attribute index : usedIndices) {
    // This is an ArrayAttr since indexing is multi-dimensional
    ArrayAttr indexAsArray = llvm::dyn_cast<ArrayAttr>(index);
    assert(indexAsArray && "expected ArrayAttr");

    Type destructAs = getType().getTypeAtIndex(indexAsArray);
    assert(destructAs == slot.elementPtrs.lookup(indexAsArray));

    ArrayType destructAsArrayTy = llvm::dyn_cast<ArrayType>(destructAs);
    assert(destructAsArrayTy && "expected ArrayType");

    auto subCreate = rewriter.create<CreateArrayOp>(getLoc(), destructAsArrayTy);
    slotMap.try_emplace<MemorySlot>(index, {subCreate.getResult(), destructAs});
  }

  return slotMap;
}

/// Required by DestructurableAllocationOpInterface / SROA pass
void CreateArrayOp::handleDestructuringComplete(
    const DestructurableMemorySlot &slot, RewriterBase &rewriter
) {
  assert(slot.ptr == getResult());
  rewriter.eraseOp(*this);
}

/// Required by PromotableAllocationOpInterface / mem2reg pass
SmallVector<MemorySlot> CreateArrayOp::getPromotableSlots() {
  ArrayType arrType = getType();
  if (!arrType.hasStaticShape()) {
    return {};
  }
  // Can only support arrays containing a single element (the SROA pass can be run first to
  // destructure all arrays into size-1 arrays).
  if (arrType.getNumElements() != 1) {
    return {};
  }
  return {MemorySlot {getResult(), arrType.getElementType()}};
}

/// Required by PromotableAllocationOpInterface / mem2reg pass
Value CreateArrayOp::getDefaultValue(const MemorySlot &slot, RewriterBase &rewriter) {
  return rewriter.create<UndefOp>(getLoc(), slot.elemType);
}

/// Required by PromotableAllocationOpInterface / mem2reg pass
void CreateArrayOp::handleBlockArgument(const MemorySlot &, BlockArgument, RewriterBase &) {}

/// Required by PromotableAllocationOpInterface / mem2reg pass
void CreateArrayOp::handlePromotionComplete(
    const MemorySlot &slot, Value defaultValue, RewriterBase &rewriter
) {
  if (defaultValue.use_empty()) {
    rewriter.eraseOp(defaultValue.getDefiningOp());
  } else {
    rewriter.eraseOp(*this);
  }
}

//===------------------------------------------------------------------===//
// ArrayAccessOpInterface
//===------------------------------------------------------------------===//

/// Returns the multi-dimensional indices of the array access as an Attribute
/// array or a null pointer if a valid index cannot be computed for any dimension.
ArrayAttr ArrayAccessOpInterface::indexOperandsToAttributeArray() {
  ArrayType arrTy = getArrRefType();
  if (arrTy.hasStaticShape()) {
    if (auto converted = ArrayIndexGen::from(arrTy).checkAndConvert(getIndices())) {
      return ArrayAttr::get(getContext(), *converted);
    }
  }
  return nullptr;
}

/// Required by DestructurableAllocationOpInterface / SROA pass
bool ArrayAccessOpInterface::canRewire(
    const DestructurableMemorySlot &slot, SmallPtrSetImpl<Attribute> &usedIndices,
    SmallVectorImpl<MemorySlot> &mustBeSafelyUsed
) {
  if (slot.ptr != getArrRef()) {
    return false;
  }

  ArrayAttr indexAsAttr = indexOperandsToAttributeArray();
  if (!indexAsAttr) {
    return false;
  }

  // Scalar read/write case has 0 dimensions in the read/write value.
  if (!getValueOperandDims().empty()) {
    return false;
  }

  // Just insert the index.
  usedIndices.insert(indexAsAttr);
  return true;
}

/// Required by DestructurableAllocationOpInterface / SROA pass
DeletionKind ArrayAccessOpInterface::rewire(
    const DestructurableMemorySlot &slot, DenseMap<Attribute, MemorySlot> &subslots,
    RewriterBase &rewriter
) {
  assert(slot.ptr == getArrRef());
  assert(slot.elemType == getArrRefType());
  // ASSERT: non-scalar read/write should have been desugared earlier
  assert(getValueOperandDims().empty() && "only scalar read/write supported");

  ArrayAttr indexAsAttr = indexOperandsToAttributeArray();
  assert(indexAsAttr && "canRewire() should have returned false");
  const MemorySlot &memorySlot = subslots.at(indexAsAttr);

  // Write to the sub-slot created for the index of `this`, using index 0
  auto idx0 = rewriter.create<arith::ConstantIndexOp>(getLoc(), 0);
  rewriter.modifyOpInPlace(*this, [&]() {
    getArrRefMutable().set(memorySlot.ptr);
    getIndicesMutable().clear();
    getIndicesMutable().assign(idx0);
  });
  return DeletionKind::Keep;
}

//===------------------------------------------------------------------===//
// ReadArrayOp
//===------------------------------------------------------------------===//

namespace {

LogicalResult
ensureNumIndicesMatchDims(ArrayType ty, size_t numIndices, const OwningEmitErrorFn &errFn) {
  ArrayRef<Attribute> dims = ty.getDimensionSizes();
  // Ensure the number of provided indices matches the array dimensions
  auto compare = numIndices <=> dims.size();
  if (compare != 0) {
    return errFn().append(
        "has ", (compare < 0 ? "insufficient" : "too many"), " indexed dimensions: expected ",
        dims.size(), " but found ", numIndices
    );
  }
  return success();
}

} // namespace

LogicalResult ReadArrayOp::verifySymbolUses(SymbolTableCollection &tables) {
  // Ensure any SymbolRef used in the type are valid
  return verifyTypeResolution(tables, *this, ArrayRef<Type> {getArrRef().getType(), getType()});
}

LogicalResult ReadArrayOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ReadArrayOpAdaptor adaptor,
    llvm::SmallVectorImpl<Type> &inferredReturnTypes
) {
  inferredReturnTypes.resize(1);
  Type lvalType = adaptor.getArrRef().getType();
  assert(llvm::isa<ArrayType>(lvalType)); // per ODS spec of ReadArrayOp
  inferredReturnTypes[0] = llvm::cast<ArrayType>(lvalType).getElementType();
  return success();
}

bool ReadArrayOp::isCompatibleReturnTypes(TypeRange l, TypeRange r) {
  return singletonTypeListsUnify(l, r);
}

LogicalResult ReadArrayOp::verify() {
  // Ensure the number of indices used match the shape of the array exactly.
  return ensureNumIndicesMatchDims(getArrRefType(), getIndices().size(), getEmitOpErrFn(this));
}

/// Required by PromotableMemOpInterface / mem2reg pass
bool ReadArrayOp::canUsesBeRemoved(
    const MemorySlot &slot, const SmallPtrSetImpl<OpOperand *> &blockingUses,
    SmallVectorImpl<OpOperand *> &newBlockingUses
) {
  if (blockingUses.size() != 1) {
    return false;
  }
  Value blockingUse = (*blockingUses.begin())->get();
  return blockingUse == slot.ptr && getArrRef() == slot.ptr &&
         getResult().getType() == slot.elemType;
}

/// Required by PromotableMemOpInterface / mem2reg pass
DeletionKind ReadArrayOp::removeBlockingUses(
    const MemorySlot &slot, const SmallPtrSetImpl<OpOperand *> &blockingUses,
    RewriterBase &rewriter, Value reachingDefinition
) {
  // `canUsesBeRemoved` checked this blocking use must be the loaded `slot.ptr`
  rewriter.replaceAllUsesWith(getResult(), reachingDefinition);
  return DeletionKind::Delete;
}

//===------------------------------------------------------------------===//
// WriteArrayOp
//===------------------------------------------------------------------===//

LogicalResult WriteArrayOp::verifySymbolUses(SymbolTableCollection &tables) {
  // Ensure any SymbolRef used in the type are valid
  return verifyTypeResolution(
      tables, *this, ArrayRef<Type> {getArrRefType(), getRvalue().getType()}
  );
}

LogicalResult WriteArrayOp::verify() {
  // Ensure the number of indices used match the shape of the array exactly.
  return ensureNumIndicesMatchDims(getArrRefType(), getIndices().size(), getEmitOpErrFn(this));
}

/// Required by PromotableMemOpInterface / mem2reg pass
bool WriteArrayOp::canUsesBeRemoved(
    const MemorySlot &slot, const SmallPtrSetImpl<OpOperand *> &blockingUses,
    SmallVectorImpl<OpOperand *> &newBlockingUses
) {
  if (blockingUses.size() != 1) {
    return false;
  }
  Value blockingUse = (*blockingUses.begin())->get();
  return blockingUse == slot.ptr && getArrRef() == slot.ptr && getRvalue() != slot.ptr &&
         getRvalue().getType() == slot.elemType;
}

/// Required by PromotableMemOpInterface / mem2reg pass
DeletionKind WriteArrayOp::removeBlockingUses(
    const MemorySlot &slot, const SmallPtrSetImpl<OpOperand *> &blockingUses,
    RewriterBase &rewriter, Value reachingDefinition
) {
  return DeletionKind::Delete;
}

//===------------------------------------------------------------------===//
// ExtractArrayOp
//===------------------------------------------------------------------===//

LogicalResult ExtractArrayOp::verifySymbolUses(SymbolTableCollection &tables) {
  // Ensure any SymbolRef used in the type are valid
  return verifyTypeResolution(tables, *this, getArrRefType());
}

LogicalResult ExtractArrayOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ExtractArrayOpAdaptor adaptor,
    llvm::SmallVectorImpl<Type> &inferredReturnTypes
) {
  size_t numToSkip = adaptor.getIndices().size();
  Type arrRefType = adaptor.getArrRef().getType();
  assert(llvm::isa<ArrayType>(arrRefType)); // per ODS spec of ExtractArrayOp
  ArrayType arrRefArrType = llvm::cast<ArrayType>(arrRefType);
  ArrayRef<Attribute> arrRefDimSizes = arrRefArrType.getDimensionSizes();

  // Check for invalid cases
  auto compare = numToSkip <=> arrRefDimSizes.size();
  if (compare == 0) {
    return mlir::emitOptionalError(
        location, "'", ExtractArrayOp::getOperationName(),
        "' op cannot select all dimensions of an array. Use '", ReadArrayOp::getOperationName(),
        "' instead."
    );
  } else if (compare > 0) {
    return mlir::emitOptionalError(
        location, "'", ExtractArrayOp::getOperationName(),
        "' op cannot select more dimensions than exist in the source array"
    );
  }

  // Generate and store reduced array type
  inferredReturnTypes.resize(1);
  inferredReturnTypes[0] =
      ArrayType::get(arrRefArrType.getElementType(), arrRefDimSizes.drop_front(numToSkip));
  return success();
}

bool ExtractArrayOp::isCompatibleReturnTypes(TypeRange l, TypeRange r) {
  return singletonTypeListsUnify(l, r);
}

//===------------------------------------------------------------------===//
// InsertArrayOp
//===------------------------------------------------------------------===//

LogicalResult InsertArrayOp::verifySymbolUses(SymbolTableCollection &tables) {
  // Ensure any SymbolRef used in the types are valid
  return verifyTypeResolution(
      tables, *this, ArrayRef<Type> {getArrRefType(), getRvalue().getType()}
  );
}

LogicalResult InsertArrayOp::verify() {
  size_t numIndices = getIndices().size();

  ArrayType baseArrRefArrType = getArrRefType();

  Type rValueType = getRvalue().getType();
  assert(llvm::isa<ArrayType>(rValueType)); // per ODS spec of InsertArrayOp
  ArrayType rValueArrType = llvm::cast<ArrayType>(rValueType);

  ArrayRef<Attribute> dimsFromBase = baseArrRefArrType.getDimensionSizes();
  // Ensure the number of indices specified does not exceed base dimension count.
  if (numIndices > dimsFromBase.size()) {
    return emitOpError("cannot select more dimensions than exist in the source array");
  }

  ArrayRef<Attribute> dimsFromRValue = rValueArrType.getDimensionSizes();
  ArrayRef<Attribute> dimsFromBaseReduced = dimsFromBase.drop_front(numIndices);
  // Ensure the rValue dimension count equals the base reduced dimension count
  auto compare = dimsFromRValue.size() <=> dimsFromBaseReduced.size();
  if (compare != 0) {
    return emitOpError().append(
        "has ", (compare < 0 ? "insufficient" : "too many"), " indexed dimensions: expected ",
        (dimsFromBase.size() - dimsFromRValue.size()), " but found ", numIndices
    );
  }

  // Ensure dimension sizes are compatible (ignoring the indexed dimensions)
  if (!typeParamsUnify(dimsFromBaseReduced, dimsFromRValue)) {
    std::string message;
    llvm::raw_string_ostream ss(message);
    auto appendOne = [&ss](Attribute a) { appendWithoutType(ss, a); };
    ss << "cannot unify array dimensions [";
    llvm::interleaveComma(dimsFromBaseReduced, ss, appendOne);
    ss << "] with [";
    llvm::interleaveComma(dimsFromRValue, ss, appendOne);
    ss << "]";
    return emitOpError().append(message);
  }

  // Ensure element types of the arrays are compatible
  if (!typesUnify(baseArrRefArrType.getElementType(), rValueArrType.getElementType())) {
    return emitOpError().append(
        "incorrect array element type; expected: ", baseArrRefArrType.getElementType(),
        ", found: ", rValueArrType.getElementType()
    );
  }

  return success();
}

//===------------------------------------------------------------------===//
// ArrayLengthOp
//===------------------------------------------------------------------===//

LogicalResult ArrayLengthOp::verifySymbolUses(SymbolTableCollection &tables) {
  // Ensure any SymbolRef used in the type are valid
  return verifyTypeResolution(tables, *this, getArrRefType());
}

} // namespace llzk::array
