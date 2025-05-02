//===-- Types.cpp - Array type implementations ------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/Array/IR/Ops.h"
#include "llzk/Dialect/Array/IR/Types.h"
#include "llzk/Dialect/Array/Util/ArrayTypeHelper.h"
#include "llzk/Util/TypeHelper.h"

using namespace mlir;

namespace llzk::array {

LogicalResult computeDimsFromShape(
    MLIRContext *ctx, ArrayRef<int64_t> shape, SmallVector<Attribute> &dimensionSizes
) {
  Builder builder(ctx);
  dimensionSizes = llvm::map_to_vector(shape, [&builder](int64_t v) -> Attribute {
    return builder.getIndexAttr(v);
  });
  assert(dimensionSizes.size() == shape.size()); // fully computed by this function
  return success();
}

LogicalResult computeShapeFromDims(
    EmitErrorFn emitError, MLIRContext *ctx, ArrayRef<Attribute> dimensionSizes,
    SmallVector<int64_t> &shape
) {
  assert(shape.empty()); // fully computed by this function

  // Ensure all Attributes are valid Attribute classes for ArrayType.
  // In the case where `emitError==null`, we mirror how the verification failure is handled by
  // `*Type::get()` via `StorageUserBase` (i.e. use DefaultDiagnosticEmitFn and assert). See:
  //  https://github.com/llvm/llvm-project/blob/0897373f1a329a7a02f8ce3c501a05d2f9c89390/mlir/include/mlir/IR/StorageUniquerSupport.h#L179-L180
  auto errFunc = emitError ? llvm::unique_function<InFlightDiagnostic()>(emitError)
                           : mlir::detail::getDefaultDiagnosticEmitFn(ctx);
  if (verifyArrayDimSizes(errFunc, dimensionSizes).failed()) {
    assert(emitError);
    return failure();
  }

  // Convert the Attributes to int64_t
  for (Attribute a : dimensionSizes) {
    if (auto p = a.dyn_cast<IntegerAttr>()) {
      shape.push_back(fromAPInt(p.getValue()));
    } else if (a.isa<SymbolRefAttr, AffineMapAttr>()) {
      // The ShapedTypeInterface uses 'kDynamic' for dimensions with non-static size.
      shape.push_back(ShapedType::kDynamic);
    } else {
      // For every Attribute class in ArrayDimensionTypes, there should be a case here.
      llvm::report_fatal_error("computeShapeFromDims() is out of sync with ArrayDimensionTypes");
      return failure();
    }
  }
  assert(shape.size() == dimensionSizes.size()); // fully computed by this function
  return success();
}

ParseResult parseDerivedShape(
    AsmParser &parser, SmallVector<int64_t> &shape, SmallVector<Attribute> dimensionSizes
) {
  // This is not actually parsing. It's computing the derived
  //  `shape` from the `dimensionSizes` attributes.
  auto emitError = [&parser] { return parser.emitError(parser.getCurrentLocation()); };
  return computeShapeFromDims(emitError, parser.getContext(), dimensionSizes, shape);
}
void printDerivedShape(AsmPrinter &, ArrayRef<int64_t>, ArrayRef<Attribute>) {
  // nothing to print, it's derived and therefore not represented in the output
}

LogicalResult ArrayType::verify(
    EmitErrorFn emitError, Type elementType, ArrayRef<Attribute> dimensionSizes,
    ArrayRef<int64_t> shape
) {
  return verifyArrayType(emitError, elementType, dimensionSizes);
}

ArrayType ArrayType::cloneWith(std::optional<ArrayRef<int64_t>> shape, Type elementType) const {
  return ArrayType::get(elementType, shape.has_value() ? shape.value() : getShape());
}

ArrayType
ArrayType::cloneWith(Type elementType, std::optional<ArrayRef<Attribute>> dimensions) const {
  return ArrayType::get(
      elementType, dimensions.has_value() ? dimensions.value() : getDimensionSizes()
  );
}

namespace {

inline ArrayType createArrayOfSizeOne(Type elemType) { return ArrayType::get(elemType, {1}); }

} // namespace

bool ArrayType::collectIndices(llvm::function_ref<void(ArrayAttr)> inserter) const {
  if (!hasStaticShape()) {
    return false;
  }
  MLIRContext *ctx = getContext();
  ArrayIndexGen idxGen = ArrayIndexGen::from(*this);
  for (int64_t e = getNumElements(), i = 0; i < e; ++i) {
    auto delinearized = idxGen.delinearize(i, ctx);
    assert(delinearized.has_value()); // cannot fail since loop is over array size
    inserter(ArrayAttr::get(ctx, delinearized.value()));
  }
  return true;
}

std::optional<SmallVector<ArrayAttr>> ArrayType::getSubelementIndices() const {
  SmallVector<ArrayAttr> ret;
  bool success = collectIndices([&ret](ArrayAttr v) { ret.push_back(v); });
  return success ? std::make_optional(ret) : std::nullopt;
}

/// Required by DestructurableTypeInterface / SROA pass
std::optional<DenseMap<Attribute, Type>> ArrayType::getSubelementIndexMap() const {
  DenseMap<Attribute, Type> ret;
  Type destructAs = createArrayOfSizeOne(getElementType());
  bool success = collectIndices([&](ArrayAttr v) { ret[v] = destructAs; });
  return success ? std::make_optional(ret) : std::nullopt;
}

/// Required by DestructurableTypeInterface / SROA pass
Type ArrayType::getTypeAtIndex(Attribute index) const {
  if (!hasStaticShape()) {
    return nullptr;
  }
  // Since indexing is multi-dimensional, `index` should be ArrayAttr
  ArrayAttr indexAttr = llvm::dyn_cast<ArrayAttr>(index);
  if (!indexAttr) {
    return nullptr;
  }
  // Ensure the shape is valid and dimensions are valid for the shape by computing linear index.
  if (!ArrayIndexGen::from(*this).linearize(indexAttr.getValue())) {
    return nullptr;
  }
  // If that's successful, the destructured type is the size-1 array of the element type.
  return createArrayOfSizeOne(getElementType());
}

ParseResult parseAttrVec(AsmParser &parser, SmallVector<Attribute> &value) {
  auto parseResult = FieldParser<SmallVector<Attribute>>::parse(parser);
  if (failed(parseResult)) {
    return parser.emitError(parser.getCurrentLocation(), "failed to parse array dimensions");
  }
  value = forceIntAttrTypes(*parseResult);
  return success();
}

namespace {

// Adapted from AsmPrinter::printStrippedAttrOrType(), but without printing type.
void printAttrs(AsmPrinter &printer, ArrayRef<Attribute> attrs, const StringRef &separator) {
  llvm::interleave(attrs, printer.getStream(), [&printer](Attribute a) {
    if (succeeded(printer.printAlias(a))) {
      return;
    }
    raw_ostream &os = printer.getStream();
    uint64_t posPrior = os.tell();
    printer.printAttributeWithoutType(a);
    // Fallback to printing with prefix if the above failed to write anything to the output stream.
    if (posPrior == os.tell()) {
      printer << a;
    }
  }, separator);
}

} // namespace

void printAttrVec(AsmPrinter &printer, ArrayRef<Attribute> value) {
  printAttrs(printer, value, ",");
}

} // namespace llzk::array
