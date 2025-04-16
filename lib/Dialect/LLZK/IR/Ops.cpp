//===-- Ops.cpp - Operation implementations ---------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/LLZK/IR/Ops.h"
#include "llzk/Dialect/LLZK/IR/Types.h"
#include "llzk/Dialect/LLZK/Util/AttributeHelper.h"
#include "llzk/Dialect/LLZK/Util/IncludeHelper.h"
#include "llzk/Dialect/LLZK/Util/SymbolHelper.h"

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

#include <cstdint>
#include <numeric>

// TableGen'd implementation files
#define GET_OP_CLASSES
#include "llzk/Dialect/LLZK/IR/OpInterfaces.cpp.inc"

// TableGen'd implementation files
#define GET_OP_CLASSES
#include "llzk/Dialect/LLZK/IR/Ops.cpp.inc"

namespace llzk {

using namespace mlir;

namespace affineMapHelpers {

namespace {
template <unsigned N>
ParseResult parseDimAndSymbolListImpl(
    OpAsmParser &parser, SmallVector<OpAsmParser::UnresolvedOperand, N> &mapOperands,
    int32_t &numDims
) {
  // Parse the required dimension operands.
  if (parser.parseOperandList(mapOperands, OpAsmParser::Delimiter::Paren)) {
    return failure();
  }
  // Store number of dimensions for validation by caller.
  numDims = mapOperands.size();

  // Parse the optional symbol operands.
  return parser.parseOperandList(mapOperands, OpAsmParser::Delimiter::OptionalSquare);
}

void printDimAndSymbolListImpl(
    OpAsmPrinter &printer, Operation *op, OperandRange mapOperands, size_t numDims
) {
  printer << '(' << mapOperands.take_front(numDims) << ')';
  if (mapOperands.size() > numDims) {
    printer << '[' << mapOperands.drop_front(numDims) << ']';
  }
}
} // namespace

template <unsigned N>
ParseResult parseDimAndSymbolList(
    OpAsmParser &parser, SmallVector<OpAsmParser::UnresolvedOperand, N> &mapOperands,
    IntegerAttr &numDims
) {
  int32_t numDimsRes = -1;
  ParseResult res = parseDimAndSymbolListImpl(parser, mapOperands, numDimsRes);
  numDims = parser.getBuilder().getIndexAttr(numDimsRes);
  return res;
}

void printDimAndSymbolList(
    OpAsmPrinter &printer, Operation *op, OperandRange mapOperands, IntegerAttr numDims
) {
  printDimAndSymbolListImpl(printer, op, mapOperands, numDims.getInt());
}

ParseResult parseMultiDimAndSymbolList(
    OpAsmParser &parser, SmallVector<SmallVector<OpAsmParser::UnresolvedOperand>> &multiMapOperands,
    DenseI32ArrayAttr &numDimsPerMap
) {
  SmallVector<int32_t> numDimsPerMapRes;
  auto parseEach = [&]() -> ParseResult {
    SmallVector<OpAsmParser::UnresolvedOperand> nextMapOps;
    int32_t nextMapDims = -1;
    ParseResult res = parseDimAndSymbolListImpl(parser, nextMapOps, nextMapDims);
    numDimsPerMapRes.push_back(nextMapDims);
    multiMapOperands.push_back(nextMapOps);
    return res;
  };
  ParseResult res = parser.parseCommaSeparatedList(AsmParser::Delimiter::None, parseEach);

  numDimsPerMap = parser.getBuilder().getDenseI32ArrayAttr(numDimsPerMapRes);
  return res;
}

void printMultiDimAndSymbolList(
    OpAsmPrinter &printer, Operation *op, OperandRangeRange multiMapOperands,
    DenseI32ArrayAttr numDimsPerMap
) {
  size_t count = numDimsPerMap.size();
  assert(multiMapOperands.size() == count);
  llvm::interleaveComma(llvm::seq<size_t>(0, count), printer.getStream(), [&](size_t i) {
    printDimAndSymbolListImpl(printer, op, multiMapOperands[i], numDimsPerMap[i]);
  });
}

ParseResult
parseAttrDictWithWarnings(OpAsmParser &parser, NamedAttrList &extraAttrs, OperationState &state) {
  // Replicate what ODS generates w/o the custom<AttrDictWithWarnings> directive
  llvm::SMLoc loc = parser.getCurrentLocation();
  if (parser.parseOptionalAttrDict(extraAttrs)) {
    return failure();
  }
  if (failed(state.name.verifyInherentAttrs(extraAttrs, [&]() {
    return parser.emitError(loc) << "'" << state.name.getStringRef() << "' op ";
  }))) {
    return failure();
  }
  // Ignore, with warnings, any attributes that are specified and shouldn't be
  for (StringAttr skipName : state.name.getAttributeNames()) {
    if (extraAttrs.erase(skipName)) {
      auto msg =
          "Ignoring attribute '" + Twine(skipName) + "' because it must be computed automatically.";
      mlir::emitWarning(parser.getEncodedSourceLoc(loc), msg).report();
    }
  }
  // There is no failure from this last check, only warnings
  return success();
}

template <typename ConcreteOp>
void printAttrDictWithWarnings(
    OpAsmPrinter &printer, ConcreteOp op, DictionaryAttr extraAttrs,
    typename ConcreteOp::Properties state
) {
  printer.printOptionalAttrDict(extraAttrs.getValue(), ConcreteOp::getAttributeNames());
}

namespace {
inline InFlightDiagnostic msgInstantiationGroupAttrMismatch(
    Operation *op, size_t mapOpGroupSizesCount, size_t mapOperandsSize
) {
  return op->emitOpError().append(
      "map instantiation group count (", mapOperandsSize,
      ") does not match with length of 'mapOpGroupSizes' attribute (", mapOpGroupSizesCount, ")"
  );
}
} // namespace

LogicalResult verifySizesForMultiAffineOps(
    Operation *op, int32_t segmentSize, ArrayRef<int32_t> mapOpGroupSizes,
    OperandRangeRange mapOperands, ArrayRef<int32_t> numDimsPerMap
) {
  // Ensure the `mapOpGroupSizes` and `operandSegmentSizes` attributes agree.
  // NOTE: the ODS generates verifyValueSizeAttr() which ensures 'mapOpGroupSizes' has no negative
  // elements and its sum is equal to the operand group size (which is similar to this check).
  // If segmentSize < 0 the check is validated regardless of the difference.
  int32_t totalMapOpGroupSizes = std::reduce(mapOpGroupSizes.begin(), mapOpGroupSizes.end());
  if (totalMapOpGroupSizes != segmentSize && segmentSize >= 0) {
    // Since `mapOpGroupSizes` and `segmentSize` are computed this should never happen.
    return op->emitOpError().append(
        "number of operands for affine map instantiation (", totalMapOpGroupSizes,
        ") does not match with the total size (", segmentSize,
        ") specified in attribute 'operandSegmentSizes'"
    );
  }

  // Ensure the size of `mapOperands` and its two list attributes are the same.
  // This will be true if the op was constructed via parseMultiDimAndSymbolList()
  //  but when constructed via the build() API, it can be inconsistent.
  size_t count = mapOpGroupSizes.size();
  if (mapOperands.size() != count) {
    return msgInstantiationGroupAttrMismatch(op, count, mapOperands.size());
  }
  if (numDimsPerMap.size() != count) {
    // Tested in CallOpTests.cpp
    return op->emitOpError().append(
        "length of 'numDimsPerMap' attribute (", numDimsPerMap.size(),
        ") does not match with length of 'mapOpGroupSizes' attribute (", count, ")"
    );
  }

  // Verify the following:
  //   1. 'mapOperands' element sizes match 'mapOpGroupSizes' values
  //   2. each 'numDimsPerMap' is <= corresponding 'mapOpGroupSizes'
  LogicalResult aggregateResult = success();
  for (size_t i = 0; i < count; ++i) {
    auto currMapOpGroupSize = mapOpGroupSizes[i];
    if (std::cmp_not_equal(mapOperands[i].size(), currMapOpGroupSize)) {
      // Since `mapOpGroupSizes` is computed this should never happen.
      aggregateResult = op->emitOpError().append(
          "map instantiation group ", i, " operand count (", mapOperands[i].size(),
          ") does not match group ", i, " size in 'mapOpGroupSizes' attribute (",
          currMapOpGroupSize, ")"
      );
    } else if (std::cmp_greater(numDimsPerMap[i], currMapOpGroupSize)) {
      // Tested in CallOpTests.cpp
      aggregateResult = op->emitOpError().append(
          "map instantiation group ", i, " dimension count (", numDimsPerMap[i], ") exceeds group ",
          i, " size in 'mapOpGroupSizes' attribute (", currMapOpGroupSize, ")"
      );
    }
  }
  return aggregateResult;
}

LogicalResult verifyAffineMapInstantiations(
    OperandRangeRange mapOps, ArrayRef<int32_t> numDimsPerMap, ArrayRef<AffineMapAttr> mapAttrs,
    Operation *origin
) {
  size_t count = numDimsPerMap.size();
  if (mapOps.size() != count) {
    return msgInstantiationGroupAttrMismatch(origin, count, mapOps.size());
  }

  // Ensure there is one OperandRange for each AffineMapAttr
  if (mapAttrs.size() != count) {
    // Tested in array_build_fail.llzk, call_with_affinemap_fail.llzk, CallOpTests.cpp, and
    // CreateArrayOpTests.cpp
    return origin->emitOpError().append(
        "map instantiation group count (", count,
        ") does not match the number of affine map instantiations (", mapAttrs.size(),
        ") required by the type"
    );
  }

  // Ensure the affine map identifier counts match the instantiation.
  // Rather than immediately returning on failure, we check all dimensions and aggregate to provide
  // as many errors are possible in a single verifier run.
  LogicalResult aggregateResult = success();
  for (size_t i = 0; i < count; ++i) {
    AffineMap map = mapAttrs[i].getAffineMap();
    if (std::cmp_not_equal(map.getNumDims(), numDimsPerMap[i])) {
      // Tested in array_build_fail.llzk and call_with_affinemap_fail.llzk
      aggregateResult = origin->emitOpError().append(
          "instantiation of map ", i, " expected ", map.getNumDims(), " but found ",
          numDimsPerMap[i], " dimension values in ()"
      );
    } else if (std::cmp_not_equal(map.getNumInputs(), mapOps[i].size())) {
      // Tested in array_build_fail.llzk and call_with_affinemap_fail.llzk
      aggregateResult = origin->emitOpError().append(
          "instantiation of map ", i, " expected ", map.getNumSymbols(), " but found ",
          (mapOps[i].size() - numDimsPerMap[i]), " symbol values in []"
      );
    }
  }
  return aggregateResult;
}
} // namespace affineMapHelpers

bool isInStruct(Operation *op) { return succeeded(getParentOfType<StructDefOp>(op)); }

FailureOr<StructDefOp> verifyInStruct(Operation *op) {
  FailureOr<StructDefOp> res = getParentOfType<StructDefOp>(op);
  if (failed(res)) {
    return op->emitOpError() << "only valid within a '" << getOperationName<StructDefOp>()
                             << "' ancestor";
  }
  return res;
}

bool isInStructFunctionNamed(Operation *op, char const *funcName) {
  FailureOr<FuncOp> parentFuncOpt = getParentOfType<FuncOp>(op);
  if (succeeded(parentFuncOpt)) {
    FuncOp parentFunc = parentFuncOpt.value();
    if (isInStruct(parentFunc.getOperation())) {
      if (parentFunc.getSymName().compare(funcName) == 0) {
        return true;
      }
    }
  }
  return false;
}

template <typename TypeClass> LogicalResult InStruct<TypeClass>::verifyTrait(Operation *op) {
  return verifyInStruct(op);
}

//===------------------------------------------------------------------===//
// IncludeOp (see IncludeHelper.cpp for other functions)
//===------------------------------------------------------------------===//

IncludeOp IncludeOp::create(Location loc, llvm::StringRef name, llvm::StringRef path) {
  return delegate_to_build<IncludeOp>(loc, name, path);
}

IncludeOp IncludeOp::create(Location loc, StringAttr name, StringAttr path) {
  return delegate_to_build<IncludeOp>(loc, name, path);
}

InFlightDiagnostic genCompareErr(StructDefOp &expected, Operation *origin, const char *aspect) {
  std::string prefix = std::string();
  if (SymbolOpInterface symbol = llvm::dyn_cast<SymbolOpInterface>(origin)) {
    prefix += "\"@";
    prefix += symbol.getName();
    prefix += "\" ";
  }
  return origin->emitOpError().append(
      prefix, "must use type of its ancestor '", StructDefOp::getOperationName(), "' \"",
      expected.getHeaderString(), "\" as ", aspect, " type"
  );
}

/// Verifies that the given `actualType` matches the `StructDefOp` given (i.e. for the "self" type
/// parameter and return of the struct functions).
LogicalResult checkSelfType(
    SymbolTableCollection &tables, StructDefOp &expectedStruct, Type actualType, Operation *origin,
    const char *aspect
) {
  if (StructType actualStructType = llvm::dyn_cast<StructType>(actualType)) {
    auto actualStructOpt =
        lookupTopLevelSymbol<StructDefOp>(tables, actualStructType.getNameRef(), origin);
    if (failed(actualStructOpt)) {
      return origin->emitError().append(
          "could not find '", StructDefOp::getOperationName(), "' named \"",
          actualStructType.getNameRef(), "\""
      );
    }
    StructDefOp actualStruct = actualStructOpt.value().get();
    if (actualStruct != expectedStruct) {
      return genCompareErr(expectedStruct, origin, aspect)
          .attachNote(actualStruct.getLoc())
          .append("uses this type instead");
    }
    // Check for an EXACT match in the parameter list since it must reference the "self" type.
    if (expectedStruct.getConstParamsAttr() != actualStructType.getParams()) {
      // To make error messages more consistent and meaningful, if the parameters don't match
      // because the actual type uses symbols that are not defined, generate an error about the
      // undefined symbol(s).
      if (ArrayAttr tyParams = actualStructType.getParams()) {
        if (failed(verifyParamsOfType(tables, tyParams.getValue(), actualStructType, origin))) {
          return failure();
        }
      }
      // Otherwise, generate an error stating the parent struct type must be used.
      return genCompareErr(expectedStruct, origin, aspect)
          .attachNote(actualStruct.getLoc())
          .append("should be type of this '", StructDefOp::getOperationName(), "'");
    }
  } else {
    return genCompareErr(expectedStruct, origin, aspect);
  }
  return success();
}

//===------------------------------------------------------------------===//
// AssertOp
//===------------------------------------------------------------------===//

// This side effect models "program termination".
// Based on
// https://github.com/llvm/llvm-project/blob/f325e4b2d836d6e65a4d0cf3efc6b0996ccf3765/mlir/lib/Dialect/ControlFlow/IR/ControlFlowOps.cpp#L92-L97
void AssertOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects
) {
  effects.emplace_back(MemoryEffects::Write::get());
}

//===------------------------------------------------------------------===//
// GlobalDefOp
//===------------------------------------------------------------------===//

ParseResult GlobalDefOp::parseGlobalInitialValue(
    OpAsmParser &parser, Attribute &initialValue, TypeAttr typeAttr
) {
  if (parser.parseOptionalEqual()) {
    // When there's no equal sign, there's no initial value to parse.
    return success();
  }
  Type specifiedType = typeAttr.getValue();

  // Special case for parsing LLZK FeltType to match format of FeltConstantOp.
  // Not actually necessary but the default format is verbose. ex: "#llzk<constfelt 35>"
  if (isa<FeltType>(specifiedType)) {
    FeltConstAttr feltConstAttr;
    if (parser.parseCustomAttributeWithFallback<FeltConstAttr>(feltConstAttr)) {
      return failure();
    }
    initialValue = feltConstAttr;
    return success();
  }
  // Fallback to default parser for all other types.
  if (failed(parser.parseAttribute(initialValue, specifiedType))) {
    return failure();
  }
  return success();
}

void GlobalDefOp::printGlobalInitialValue(
    OpAsmPrinter &p, GlobalDefOp op, Attribute initialValue, TypeAttr typeAttr
) {
  if (initialValue) {
    p << " = ";
    // Special case for LLZK FeltType to match format of FeltConstantOp.
    // Not actually necessary but the default format is verbose. ex: "#llzk<constfelt 35>"
    if (FeltConstAttr feltConstAttr = llvm::dyn_cast<FeltConstAttr>(initialValue)) {
      p.printStrippedAttrOrType<FeltConstAttr>(feltConstAttr);
    } else {
      p.printAttributeWithoutType(initialValue);
    }
  }
}

LogicalResult GlobalDefOp::verifySymbolUses(SymbolTableCollection &tables) {
  // Ensure any SymbolRef used in the type are valid
  return verifyTypeResolution(tables, *this, getType());
}

namespace {

inline InFlightDiagnostic reportMismatch(
    EmitErrorFn errFn, Type rootType, const Twine &aspect, const Twine &expected, const Twine &found
) {
  return errFn().append(
      "with type ", rootType, " expected ", expected, " ", aspect, " but found ", found
  );
}

inline InFlightDiagnostic reportMismatch(
    EmitErrorFn errFn, Type rootType, const Twine &aspect, const Twine &expected, Attribute found
) {
  return reportMismatch(errFn, rootType, aspect, expected, found.getAbstractAttribute().getName());
}

LogicalResult ensureAttrTypeMatch(
    Type type, Attribute valAttr, const OwningEmitErrorFn &errFn, Type rootType, const Twine &aspect
) {
  if (!isValidGlobalType(type)) {
    // Same error message ODS-generated code would produce
    return errFn().append("attribute 'type' failed to satisfy constraint: type attribute of "
                          "any LLZK type except non-constant types");
  }
  if (type.isSignlessInteger(1)) {
    if (IntegerAttr ia = llvm::dyn_cast<IntegerAttr>(valAttr)) {
      APInt val = ia.getValue();
      if (!val.isZero() && !val.isOne()) {
        return errFn().append("integer constant out of range for attribute");
      }
    } else if (!llvm::isa<BoolAttr>(valAttr)) {
      return reportMismatch(errFn, rootType, aspect, "builtin.bool or builtin.integer", valAttr);
    }
  } else if (llvm::isa<IndexType>(type)) {
    // The explicit check for BoolAttr is needed because the LLVM isa/cast functions treat
    // BoolAttr as a subtype of IntegerAttr but this scenario should not allow BoolAttr.
    bool isBool = llvm::isa<BoolAttr>(valAttr);
    if (isBool || !llvm::isa<IntegerAttr>(valAttr)) {
      return reportMismatch(
          errFn, rootType, aspect, "builtin.index",
          isBool ? "builtin.bool" : valAttr.getAbstractAttribute().getName()
      );
    }
  } else if (llvm::isa<FeltType>(type)) {
    if (!llvm::isa<FeltConstAttr, IntegerAttr>(valAttr)) {
      return reportMismatch(errFn, rootType, aspect, "llzk.felt", valAttr);
    }
  } else if (llvm::isa<StringType>(type)) {
    if (!llvm::isa<StringAttr>(valAttr)) {
      return reportMismatch(errFn, rootType, aspect, "builtin.string", valAttr);
    }
  } else if (ArrayType arrTy = llvm::dyn_cast<ArrayType>(type)) {
    if (ArrayAttr arrVal = llvm::dyn_cast<ArrayAttr>(valAttr)) {
      // Ensure the number of elements is correct for the ArrayType
      assert(arrTy.hasStaticShape() && "implied by earlier isValidGlobalType() check");
      int64_t expectedCount = arrTy.getNumElements();
      size_t actualCount = arrVal.size();
      if (std::cmp_not_equal(actualCount, expectedCount)) {
        return reportMismatch(
            errFn, rootType, Twine(aspect) + " to contain " + Twine(expectedCount) + " elements",
            "builtin.array", Twine(actualCount)
        );
      }
      // Ensure the type of each element is correct for the ArrayType.
      // Rather than immediately returning on failure, check all elements and aggregate to provide
      // as many errors are possible in a single verifier run.
      bool hasFailure = false;
      Type expectedElemTy = arrTy.getElementType();
      for (Attribute e : arrVal.getValue()) {
        hasFailure |=
            failed(ensureAttrTypeMatch(expectedElemTy, e, errFn, rootType, "array element"));
      }
      if (hasFailure) {
        return failure();
      }
    } else {
      return reportMismatch(errFn, rootType, aspect, "builtin.array", valAttr);
    }
  } else {
    return errFn().append("expected a valid LLZK type but found ", type);
  }
  return success();
}

} // namespace

LogicalResult GlobalDefOp::verify() {
  if (Attribute initValAttr = getInitialValueAttr()) {
    Type ty = getType();
    OwningEmitErrorFn errFn = getEmitOpErrFn(this);
    return ensureAttrTypeMatch(ty, initValAttr, errFn, ty, "attribute value");
  }
  // If there is no initial value, it cannot have "const".
  if (isConstant()) {
    return emitOpError("marked as 'const' must be assigned a value");
  }
  return success();
}

//===------------------------------------------------------------------===//
// GlobalReadOp / GlobalWriteOp
//===------------------------------------------------------------------===//

FailureOr<SymbolLookupResult<GlobalDefOp>>
GlobalRefOpInterface::getGlobalDefOp(SymbolTableCollection &tables) {
  return lookupTopLevelSymbol<GlobalDefOp>(tables, getNameRef(), getOperation());
}

namespace {

FailureOr<SymbolLookupResult<GlobalDefOp>>
verifySymbolUsesImpl(GlobalRefOpInterface refOp, SymbolTableCollection &tables) {
  // Ensure this op references a valid GlobalDefOp name
  auto tgt = refOp.getGlobalDefOp(tables);
  if (failed(tgt)) {
    return failure();
  }
  // Ensure the SSA Value type matches the GlobalDefOp type
  Type globalType = tgt->get().getType();
  if (!typesUnify(refOp.getVal().getType(), globalType, tgt->getIncludeSymNames())) {
    return refOp->emitOpError() << "has wrong type; expected " << globalType << ", got "
                                << refOp.getVal().getType();
  }
  return tgt;
}

} // namespace

LogicalResult GlobalReadOp::verifySymbolUses(SymbolTableCollection &tables) {
  if (failed(verifySymbolUsesImpl(*this, tables))) {
    return failure();
  }
  // Ensure any SymbolRef used in the type are valid
  return verifyTypeResolution(tables, *this, getType());
}

LogicalResult GlobalWriteOp::verifySymbolUses(SymbolTableCollection &tables) {
  auto tgt = verifySymbolUsesImpl(*this, tables);
  if (failed(tgt)) {
    return failure();
  }
  if (tgt->get().isConstant()) {
    return emitOpError().append(
        "cannot target '", GlobalDefOp::getOperationName(), "' marked as 'const'"
    );
  }
  return success();
}

//===------------------------------------------------------------------===//
// StructDefOp
//===------------------------------------------------------------------===//
namespace {

inline LogicalResult msgOneFunction(EmitErrorFn emitError, const Twine &name) {
  return emitError() << "must define exactly one '" << name << "' function";
}

} // namespace

StructType StructDefOp::getType(std::optional<ArrayAttr> constParams) {
  auto pathRes = getPathFromRoot(*this);
  assert(succeeded(pathRes)); // consistent with StructType::get() with invalid args
  return StructType::get(pathRes.value(), constParams.value_or(getConstParamsAttr()));
}

std::string StructDefOp::getHeaderString() {
  std::string output;
  llvm::raw_string_ostream oss(output);
  FailureOr<SymbolRefAttr> pathToExpected = getPathFromRoot(*this);
  if (succeeded(pathToExpected)) {
    oss << pathToExpected.value();
  } else {
    // When there is a failure trying to get the resolved name of the struct,
    //  just print its symbol name directly.
    oss << "@" << this->getSymName();
  }
  if (auto attr = this->getConstParamsAttr()) {
    oss << "<" << attr << ">";
  }
  return output;
}

bool StructDefOp::hasParamNamed(StringAttr find) {
  if (ArrayAttr params = this->getConstParamsAttr()) {
    for (Attribute attr : params) {
      assert(llvm::isa<FlatSymbolRefAttr>(attr)); // per ODS
      if (llvm::cast<FlatSymbolRefAttr>(attr).getRootReference() == find) {
        return true;
      }
    }
  }
  return false;
}

SymbolRefAttr StructDefOp::getFullyQualifiedName() {
  auto res = getPathFromRoot(*this);
  assert(succeeded(res));
  return res.value();
}

LogicalResult StructDefOp::verifySymbolUses(SymbolTableCollection &tables) {
  if (ArrayAttr params = this->getConstParamsAttr()) {
    // Ensure struct parameter names are unique
    llvm::StringSet<> uniqNames;
    for (Attribute attr : params) {
      assert(llvm::isa<FlatSymbolRefAttr>(attr)); // per ODS
      StringRef name = llvm::cast<FlatSymbolRefAttr>(attr).getValue();
      if (!uniqNames.insert(name).second) {
        return this->emitOpError().append("has more than one parameter named \"@", name, "\"");
      }
    }
    // Ensure they do not conflict with existing symbols
    for (Attribute attr : params) {
      auto res = lookupTopLevelSymbol(tables, llvm::cast<FlatSymbolRefAttr>(attr), *this, false);
      if (succeeded(res)) {
        return this->emitOpError()
            .append("parameter name \"@")
            .append(llvm::cast<FlatSymbolRefAttr>(attr).getValue())
            .append("\" conflicts with an existing symbol")
            .attachNote(res->get()->getLoc())
            .append("symbol already defined here");
      }
    }
  }
  return success();
}

namespace {

inline LogicalResult checkMainFuncParamType(Type pType, FuncOp inFunc, bool appendSelf) {
  if (isSignalType(pType)) {
    return success();
  } else if (auto arrayParamTy = mlir::dyn_cast<ArrayType>(pType)) {
    if (isSignalType(arrayParamTy.getElementType())) {
      return success();
    }
  }

  std::string message;
  llvm::raw_string_ostream ss(message);
  ss << "\"@" << COMPONENT_NAME_MAIN << "\" component \"@" << inFunc.getSymName()
     << "\" function parameters must be one of: {";
  if (appendSelf) {
    ss << "!" << StructType::name << "<@" << COMPONENT_NAME_MAIN << ">, ";
  }
  ss << "!" << StructType::name << "<@" << COMPONENT_NAME_SIGNAL << ">, ";
  ss << "!" << ArrayType::name << "<.. x !" << StructType::name << "<@" << COMPONENT_NAME_SIGNAL
     << ">>}";
  return inFunc.emitError(message);
}

} // namespace

LogicalResult StructDefOp::verifyRegions() {
  assert(getBody().hasOneBlock()); // per ODS, SizedRegion<1>
  std::optional<FuncOp> foundCompute = std::nullopt;
  std::optional<FuncOp> foundConstrain = std::nullopt;
  {
    // Verify the following:
    // 1. The only ops within the body are field and function definitions
    // 2. The only functions defined in the struct are `compute()` and `constrain()`
    OwningEmitErrorFn emitError = getEmitOpErrFn(this);
    for (Operation &op : getBody().front()) {
      if (!llvm::isa<FieldDefOp>(op)) {
        if (FuncOp funcDef = llvm::dyn_cast<FuncOp>(op)) {
          if (funcDef.nameIsCompute()) {
            if (foundCompute) {
              return msgOneFunction(emitError, FUNC_NAME_COMPUTE);
            }
            foundCompute = std::make_optional(funcDef);
          } else if (funcDef.nameIsConstrain()) {
            if (foundConstrain) {
              return msgOneFunction(emitError, FUNC_NAME_CONSTRAIN);
            }
            foundConstrain = std::make_optional(funcDef);
          } else {
            // Must do a little more than a simple call to '?.emitOpError()' to
            // tag the error with correct location and correct op name.
            return op.emitError() << "'" << getOperationName() << "' op " << "must define only \"@"
                                  << FUNC_NAME_COMPUTE << "\" and \"@" << FUNC_NAME_CONSTRAIN
                                  << "\" functions;" << " found \"@" << funcDef.getSymName()
                                  << "\"";
          }
        } else {
          return op.emitOpError() << "invalid operation in '" << StructDefOp::getOperationName()
                                  << "'; only '" << FieldDefOp::getOperationName() << "'"
                                  << " and '" << FuncOp::getOperationName()
                                  << "' operations are permitted";
        }
      }
    }
    if (!foundCompute.has_value()) {
      return msgOneFunction(emitError, FUNC_NAME_COMPUTE);
    }
    if (!foundConstrain.has_value()) {
      return msgOneFunction(emitError, FUNC_NAME_CONSTRAIN);
    }
  }

  // Verify parameter types are valid. Skip the first parameter of the "constrain" function; it is
  // already checked via verifyFuncTypeConstrain() in FuncOps.cpp.
  ArrayRef<Type> computeParams = foundCompute->getFunctionType().getInputs();
  ArrayRef<Type> constrainParams = foundConstrain->getFunctionType().getInputs().drop_front();
  if (COMPONENT_NAME_MAIN == this->getSymName()) {
    // Verify that the Struct has no parameters.
    if (!isNullOrEmpty(this->getConstParamsAttr())) {
      return this->emitError().append(
          "The \"@", COMPONENT_NAME_MAIN, "\" component must have no parameters"
      );
    }
    // Verify the input parameter types are legal. The error message is explicit about what types
    // are allowed so there is no benefit to report multiple errors if more than one parameter in
    // the referenced function has an illegal type.
    for (Type t : computeParams) {
      if (failed(checkMainFuncParamType(t, *foundCompute, false))) {
        return failure(); // checkMainFuncParamType() already emits a sufficient error message
      }
    }
    for (Type t : constrainParams) {
      if (failed(checkMainFuncParamType(t, *foundConstrain, true))) {
        return failure(); // checkMainFuncParamType() already emits a sufficient error message
      }
    }
  }
  // Verify that function input types from `compute()` and `constrain()` match, sans the first
  // parameter of `constrain()` which is the instance of the parent struct.
  if (!typeListsUnify(computeParams, constrainParams)) {
    return foundConstrain->emitError()
        .append(
            "expected \"@", FUNC_NAME_CONSTRAIN,
            "\" function argument types (sans the first one) to match \"@", FUNC_NAME_COMPUTE,
            "\" function argument types"
        )
        .attachNote(foundCompute->getLoc())
        .append("\"@", FUNC_NAME_COMPUTE, "\" function defined here");
  }

  return success();
}

FieldDefOp StructDefOp::getFieldDef(StringAttr fieldName) {
  assert(getBody().hasOneBlock()); // per ODS, SizedRegion<1>
  // Just search front() since there's only one Block.
  for (Operation &op : getBody().front()) {
    if (FieldDefOp fieldDef = llvm::dyn_cast_if_present<FieldDefOp>(op)) {
      if (fieldName.compare(fieldDef.getSymNameAttr()) == 0) {
        return fieldDef;
      }
    }
  }
  return nullptr;
}

std::vector<FieldDefOp> StructDefOp::getFieldDefs() {
  assert(getBody().hasOneBlock()); // per ODS, SizedRegion<1>
  // Just search front() since there's only one Block.
  std::vector<FieldDefOp> res;
  for (Operation &op : getBody().front()) {
    if (FieldDefOp fieldDef = llvm::dyn_cast_if_present<FieldDefOp>(op)) {
      res.push_back(fieldDef);
    }
  }
  return res;
}

FuncOp StructDefOp::getComputeFuncOp() {
  return llvm::dyn_cast_if_present<FuncOp>(lookupSymbol(FUNC_NAME_COMPUTE));
}

FuncOp StructDefOp::getConstrainFuncOp() {
  return llvm::dyn_cast_if_present<FuncOp>(lookupSymbol(FUNC_NAME_CONSTRAIN));
}

bool StructDefOp::isMainComponent() { return COMPONENT_NAME_MAIN == this->getSymName(); }

//===------------------------------------------------------------------===//
// ConstReadOp
//===------------------------------------------------------------------===//

LogicalResult ConstReadOp::verifySymbolUses(SymbolTableCollection &tables) {
  FailureOr<StructDefOp> getParentRes = verifyInStruct(*this);
  if (failed(getParentRes)) {
    return failure(); // verifyInStruct() already emits a sufficient error message
  }
  // Ensure the named constant is a a parameter of the parent struct
  if (!getParentRes->hasParamNamed(this->getConstNameAttr())) {
    return this->emitOpError()
        .append("references unknown symbol \"", this->getConstNameAttr(), "\"")
        .attachNote(getParentRes->getLoc())
        .append("must reference a parameter of this struct");
  }
  // Ensure any SymbolRef used in the type are valid
  return verifyTypeResolution(tables, *this, getType());
}

//===------------------------------------------------------------------===//
// FieldDefOp
//===------------------------------------------------------------------===//

void FieldDefOp::build(
    OpBuilder &odsBuilder, OperationState &odsState, StringAttr sym_name, TypeAttr type,
    bool isColumn
) {
  Properties &props = odsState.getOrAddProperties<Properties>();
  props.setSymName(sym_name);
  props.setType(type);
  if (isColumn) {
    props.column = odsBuilder.getUnitAttr();
  }
}

void FieldDefOp::build(
    OpBuilder &odsBuilder, OperationState &odsState, StringRef sym_name, Type type, bool isColumn
) {
  build(odsBuilder, odsState, odsBuilder.getStringAttr(sym_name), TypeAttr::get(type), isColumn);
}

void FieldDefOp::build(
    OpBuilder &odsBuilder, OperationState &odsState, TypeRange resultTypes, ValueRange operands,
    ArrayRef<NamedAttribute> attributes, bool isColumn
) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
  if (isColumn) {
    odsState.getOrAddProperties<Properties>().column = odsBuilder.getUnitAttr();
  }
}
void FieldDefOp::setPublicAttr(bool newValue) {
  if (newValue) {
    getOperation()->setAttr(PublicAttr::name, PublicAttr::get(getContext()));
  } else {
    getOperation()->removeAttr(PublicAttr::name);
  }
}

static LogicalResult
verifyFieldDefTypeImpl(Type fieldType, SymbolTableCollection &tables, Operation *origin) {
  if (StructType fieldStructType = llvm::dyn_cast<StructType>(fieldType)) {
    // Special case for StructType verifies that the field type can resolve and that it is NOT the
    // parent struct (i.e. struct fields cannot create circular references).
    auto fieldTypeRes = verifyStructTypeResolution(tables, fieldStructType, origin);
    if (failed(fieldTypeRes)) {
      return failure(); // above already emits a sufficient error message
    }
    FailureOr<StructDefOp> parentRes = getParentOfType<StructDefOp>(origin);
    assert(succeeded(parentRes) && "FieldDefOp parent is always StructDefOp"); // per ODS def
    if (fieldTypeRes.value() == parentRes.value()) {
      return origin->emitOpError()
          .append("type is circular")
          .attachNote(parentRes.value().getLoc())
          .append("references parent component defined here");
    }
    return success();
  } else {
    return verifyTypeResolution(tables, origin, fieldType);
  }
}

LogicalResult FieldDefOp::verifySymbolUses(SymbolTableCollection &tables) {
  Type fieldType = this->getType();
  if (failed(verifyFieldDefTypeImpl(fieldType, tables, *this))) {
    return failure();
  }

  if (!getColumn()) {
    return success();
  }
  // If the field is marked as a column only a small subset of types are allowed.
  if (!isValidColumnType(getType(), tables, *this)) {
    return emitOpError() << "marked as column can only contain felts, arrays of column types, or "
                            "structs with columns, but field has type "
                         << getType();
  }
  return success();
}

//===------------------------------------------------------------------===//
// FieldRefOp implementations
//===------------------------------------------------------------------===//
namespace {

FailureOr<SymbolLookupResult<FieldDefOp>>
getFieldDefOpImpl(FieldRefOpInterface refOp, SymbolTableCollection &tables, StructType tyStruct) {
  Operation *op = refOp.getOperation();
  auto structDefRes = tyStruct.getDefinition(tables, op);
  if (failed(structDefRes)) {
    return failure(); // getDefinition() already emits a sufficient error message
  }
  auto res = llzk::lookupSymbolIn<FieldDefOp>(
      tables, SymbolRefAttr::get(refOp->getContext(), refOp.getFieldName()),
      std::move(*structDefRes), op
  );
  if (failed(res)) {
    return refOp->emitError() << "could not find '" << FieldDefOp::getOperationName()
                              << "' named \"@" << refOp.getFieldName() << "\" in \""
                              << tyStruct.getNameRef() << "\"";
  }
  return std::move(res.value());
}

static FailureOr<SymbolLookupResult<FieldDefOp>>
findField(FieldRefOpInterface refOp, SymbolTableCollection &tables) {
  // Ensure the base component/struct type reference can be resolved.
  StructType tyStruct = refOp.getStructType();
  if (failed(tyStruct.verifySymbolRef(tables, refOp.getOperation()))) {
    return failure();
  }
  // Ensure the field name can be resolved in that struct.
  return getFieldDefOpImpl(refOp, tables, tyStruct);
}

static LogicalResult verifySymbolUsesImpl(
    FieldRefOpInterface refOp, SymbolTableCollection &tables, SymbolLookupResult<FieldDefOp> &field
) {
  // Ensure the type of the referenced field declaration matches the type used in this op.
  Type actualType = refOp.getVal().getType();
  Type fieldType = field.get().getType();
  if (!typesUnify(actualType, fieldType, field.getIncludeSymNames())) {
    return refOp->emitOpError() << "has wrong type; expected " << fieldType << ", got "
                                << actualType;
  }
  // Ensure any SymbolRef used in the type are valid
  return verifyTypeResolution(tables, refOp.getOperation(), actualType);
}

LogicalResult verifySymbolUsesImpl(FieldRefOpInterface refOp, SymbolTableCollection &tables) {
  // Ensure the field name can be resolved in that struct.
  auto field = findField(refOp, tables);
  if (failed(field)) {
    return field; // getFieldDefOp() already emits a sufficient error message
  }
  return verifySymbolUsesImpl(refOp, tables, *field);
}

} // namespace

FailureOr<SymbolLookupResult<FieldDefOp>>
FieldRefOpInterface::getFieldDefOp(SymbolTableCollection &tables) {
  return getFieldDefOpImpl(*this, tables, getStructType());
}

LogicalResult FieldReadOp::verifySymbolUses(SymbolTableCollection &tables) {
  auto field = findField(*this, tables);
  if (failed(field)) {
    return failure();
  }
  if (failed(verifySymbolUsesImpl(*this, tables, *field))) {
    return failure();
  }
  // If the field is not a column and an offset was specified then fail to validate
  if (!field->get().getColumn() && getTableOffset().has_value()) {
    return emitOpError("cannot read with table offset from a field that is not a column")
        .attachNote(field->get().getLoc())
        .append("field defined here");
  }

  return success();
}

LogicalResult FieldWriteOp::verifySymbolUses(SymbolTableCollection &tables) {
  // Ensure the write op only targets fields in the current struct.
  FailureOr<StructDefOp> getParentRes = verifyInStruct(*this);
  if (failed(getParentRes)) {
    return failure(); // verifyInStruct() already emits a sufficient error message
  }
  if (failed(checkSelfType(tables, *getParentRes, getComponent().getType(), *this, "base value"))) {
    return failure(); // checkSelfType() already emits a sufficient error message
  }
  // Perform the standard field ref checks.
  return verifySymbolUsesImpl(*this, tables);
}

//===------------------------------------------------------------------===//
// FieldReadOp
//===------------------------------------------------------------------===//

void FieldReadOp::build(
    OpBuilder &builder, OperationState &state, Type resultType, Value component, StringAttr field
) {
  Properties &props = state.getOrAddProperties<Properties>();
  props.setFieldName(FlatSymbolRefAttr::get(field));
  state.addTypes(resultType);
  state.addOperands(component);
  affineMapHelpers::buildInstantiationAttrsEmptyNoSegments<FieldReadOp>(builder, state);
}

void FieldReadOp::build(
    OpBuilder &builder, OperationState &state, Type resultType, Value component, StringAttr field,
    Attribute dist, ValueRange mapOperands, std::optional<int32_t> numDims
) {
  assert(numDims.has_value() != mapOperands.empty());
  state.addOperands(component);
  state.addTypes(resultType);
  if (numDims.has_value()) {
    affineMapHelpers::buildInstantiationAttrsNoSegments<FieldReadOp>(
        builder, state, ArrayRef({mapOperands}), builder.getDenseI32ArrayAttr({*numDims})
    );
  } else {
    affineMapHelpers::buildInstantiationAttrsEmptyNoSegments<FieldReadOp>(builder, state);
  }
  Properties &props = state.getOrAddProperties<Properties>();
  props.setFieldName(FlatSymbolRefAttr::get(field));
  props.setTableOffset(dist);
}

void FieldReadOp::build(
    OpBuilder &builder, OperationState &state, TypeRange resultTypes, ValueRange operands,
    ArrayRef<NamedAttribute> attrs
) {
  state.addTypes(resultTypes);
  state.addOperands(operands);
  state.addAttributes(attrs);
}

LogicalResult FieldReadOp::verify() {
  mlir::SmallVector<AffineMapAttr, 1> mapAttrs;
  if (AffineMapAttr map =
          mlir::dyn_cast_if_present<AffineMapAttr>(getTableOffset().value_or(nullptr))) {
    mapAttrs.push_back(map);
  }
  return affineMapHelpers::verifyAffineMapInstantiations(
      getMapOperands(), getNumDimsPerMap(), mapAttrs, *this
  );
}

//===------------------------------------------------------------------===//
// FeltConstantOp
//===------------------------------------------------------------------===//

void FeltConstantOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  llvm::SmallString<32> buf;
  llvm::raw_svector_ostream os(buf);
  os << "felt_const_";
  getValue().getValue().toStringUnsigned(buf);
  setNameFn(getResult(), buf);
}

OpFoldResult FeltConstantOp::fold(FeltConstantOp::FoldAdaptor) { return getValue(); }

//===------------------------------------------------------------------===//
// FeltNonDetOp
//===------------------------------------------------------------------===//

void FeltNonDetOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  setNameFn(getResult(), "felt_nondet");
}

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

//===------------------------------------------------------------------===//
// ReadArrayOp
//===------------------------------------------------------------------===//

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

//===------------------------------------------------------------------===//
// WriteArrayOp
//===------------------------------------------------------------------===//

LogicalResult WriteArrayOp::verifySymbolUses(SymbolTableCollection &tables) {
  // Ensure any SymbolRef used in the type are valid
  return verifyTypeResolution(
      tables, *this, ArrayRef<Type> {getArrRef().getType(), getRvalue().getType()}
  );
}

//===------------------------------------------------------------------===//
// ExtractArrayOp
//===------------------------------------------------------------------===//

LogicalResult ExtractArrayOp::verifySymbolUses(SymbolTableCollection &tables) {
  // Ensure any SymbolRef used in the type are valid
  return verifyTypeResolution(tables, *this, getArrRef().getType());
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
      tables, *this, ArrayRef<Type> {getArrRef().getType(), getRvalue().getType()}
  );
}

LogicalResult InsertArrayOp::verify() {
  size_t numIndices = getIndices().size();

  Type baseArrRefType = getArrRef().getType();
  assert(llvm::isa<ArrayType>(baseArrRefType)); // per ODS spec of InsertArrayOp
  ArrayType baseArrRefArrType = llvm::cast<ArrayType>(baseArrRefType);

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
  return verifyTypeResolution(tables, *this, getArrRef().getType());
}

//===------------------------------------------------------------------===//
// EmitEqualityOp
//===------------------------------------------------------------------===//

LogicalResult EmitEqualityOp::verifySymbolUses(SymbolTableCollection &tables) {
  // Ensure any SymbolRef used in the type are valid
  return verifyTypeResolution(
      tables, *this, ArrayRef<Type> {getLhs().getType(), getRhs().getType()}
  );
}

Type EmitEqualityOp::inferRHS(Type lhsType) { return lhsType; }

//===------------------------------------------------------------------===//
// EmitContainmentOp
//===------------------------------------------------------------------===//

LogicalResult EmitContainmentOp::verifySymbolUses(SymbolTableCollection &tables) {
  // Ensure any SymbolRef used in the type are valid
  return verifyTypeResolution(
      tables, *this, ArrayRef<Type> {getLhs().getType(), getRhs().getType()}
  );
}

Type EmitContainmentOp::inferRHS(Type lhsType) {
  assert(llvm::isa<ArrayType>(lhsType)); // per ODS spec of EmitContainmentOp
  return llvm::cast<ArrayType>(lhsType).getElementType();
}

//===------------------------------------------------------------------===//
// CreateStructOp
//===------------------------------------------------------------------===//

void CreateStructOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  setNameFn(getResult(), "self");
}

LogicalResult CreateStructOp::verifySymbolUses(SymbolTableCollection &tables) {
  FailureOr<StructDefOp> getParentRes = verifyInStruct(*this);
  if (failed(getParentRes)) {
    return failure(); // verifyInStruct() already emits a sufficient error message
  }
  if (failed(checkSelfType(tables, *getParentRes, this->getType(), *this, "result"))) {
    return failure();
  }
  return success();
}

//===------------------------------------------------------------------===//
// ApplyMapOp
//===------------------------------------------------------------------===//

LogicalResult ApplyMapOp::verify() {
  // Check input and output dimensions match.
  AffineMap map = getMap();

  // Verify that the map only produces one result.
  if (map.getNumResults() != 1) {
    return emitOpError("must produce exactly one value");
  }

  // Verify that operand count matches affine map dimension and symbol count.
  unsigned mapDims = map.getNumDims();
  if (getNumOperands() != mapDims + map.getNumSymbols()) {
    return emitOpError("operand count must equal affine map dimension+symbol count");
  } else if (mapDims != getNumDimsAttr().getInt()) {
    return emitOpError("dimension operand count must equal affine map dimension count");
  }

  return success();
}

//===------------------------------------------------------------------===//
// LitStringOp
//===------------------------------------------------------------------===//

OpFoldResult LitStringOp::fold(LitStringOp::FoldAdaptor) { return getValueAttr(); }

//===------------------------------------------------------------------===//
// FeltToIndexOp
//===------------------------------------------------------------------===//

LogicalResult FeltToIndexOp::verify() {
  if (auto parentOr = getParentOfType<FuncOp>(*this);
      succeeded(parentOr) && parentOr->isStructConstrain()) {
    // Traverse the def-use chain to see if this operand, which is a felt, ever
    // derives from a Signal struct.
    SmallVector<Value, 2> frontier {getValue()};
    DenseSet<Value> visited;

    while (!frontier.empty()) {
      Value v = frontier.pop_back_val();
      if (visited.contains(v)) {
        continue;
      }
      visited.insert(v);

      if (Operation *op = v.getDefiningOp()) {
        if (FieldReadOp readf = mlir::dyn_cast<FieldReadOp>(op);
            readf && isSignalType(readf.getComponent().getType())) {
          return emitOpError()
              .append("input is derived from a Signal struct, which is illegal in struct constrain "
                      "function")
              .attachNote(readf.getLoc())
              .append("Signal struct value is read here");
        }
        frontier.insert(frontier.end(), op->operand_begin(), op->operand_end());
      }
    }
  }

  return success();
}

//===------------------------------------------------------------------===//
// UnifiableCastOp
//===------------------------------------------------------------------===//

LogicalResult UnifiableCastOp::verify() {
  if (!typesUnify(getInput().getType(), getResult().getType())) {
    return emitOpError() << "input type " << getInput().getType() << " and output type "
                         << getResult().getType() << " are not unifiable";
  }

  return success();
}

} // namespace llzk
