//===-- Ops.cpp - Operation implementations ---------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/Array/Util/ArrayTypeHelper.h"
#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Dialect/Include/Util/IncludeHelper.h"
#include "llzk/Dialect/LLZK/IR/Ops.h"
#include "llzk/Dialect/LLZK/IR/Types.h"
#include "llzk/Util/AffineHelper.h"
#include "llzk/Util/AttributeHelper.h"
#include "llzk/Util/SymbolHelper.h"

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

#include <cstdint>
#include <numeric>

// TableGen'd implementation files
#include "llzk/Dialect/LLZK/IR/OpInterfaces.cpp.inc"

// TableGen'd implementation files
#define GET_OP_CLASSES
#include "llzk/Dialect/LLZK/IR/Ops.cpp.inc"

namespace llzk {

using namespace mlir;
using namespace function;
using namespace array;

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

inline LogicalResult checkMainFuncParamType(Type pType, FuncDefOp inFunc, bool appendSelf) {
  if (isSignalType(pType)) {
    return success();
  } else if (auto arrayParamTy = llvm::dyn_cast<ArrayType>(pType)) {
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
  std::optional<FuncDefOp> foundCompute = std::nullopt;
  std::optional<FuncDefOp> foundConstrain = std::nullopt;
  {
    // Verify the following:
    // 1. The only ops within the body are field and function definitions
    // 2. The only functions defined in the struct are `compute()` and `constrain()`
    OwningEmitErrorFn emitError = getEmitOpErrFn(this);
    for (Operation &op : getBody().front()) {
      if (!llvm::isa<FieldDefOp>(op)) {
        if (FuncDefOp funcDef = llvm::dyn_cast<FuncDefOp>(op)) {
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
                                  << " and '" << FuncDefOp::getOperationName()
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

  // ASSERT: The `InitAllowConstraintsAttrs` trait on StructDefOp set the attributes correctly.
  assert(foundConstrain->hasAllowConstraintsAttr());
  assert(!foundCompute->hasAllowConstraintsAttr());

  // Verify parameter types are valid. Skip the first parameter of the "constrain" function; it is
  // already checked via verifyFuncTypeConstrain() in Function/IR/Ops.cpp.
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

FuncDefOp StructDefOp::getComputeFuncOp() {
  return llvm::dyn_cast_if_present<FuncDefOp>(lookupSymbol(FUNC_NAME_COMPUTE));
}

FuncDefOp StructDefOp::getConstrainFuncOp() {
  return llvm::dyn_cast_if_present<FuncDefOp>(lookupSymbol(FUNC_NAME_CONSTRAIN));
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
  SmallVector<AffineMapAttr, 1> mapAttrs;
  if (AffineMapAttr map =
          llvm::dyn_cast_if_present<AffineMapAttr>(getTableOffset().value_or(nullptr))) {
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
// UnifiableCastOp
//===------------------------------------------------------------------===//

LogicalResult UnifiableCastOp::verify() {
  if (!typesUnify(getInput().getType(), getResult().getType())) {
    return emitOpError() << "input type " << getInput().getType() << " and output type "
                         << getResult().getType() << " are not unifiable";
  }

  return success();
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

} // namespace llzk
