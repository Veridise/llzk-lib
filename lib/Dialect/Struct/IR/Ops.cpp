//===-- Ops.cpp - Struct op implementations ---------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/Array/IR/Types.h"
#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Dialect/LLZK/IR/AttributeHelper.h"
#include "llzk/Dialect/Struct/IR/Ops.h"
#include "llzk/Util/AffineHelper.h"
#include "llzk/Util/Constants.h"
#include "llzk/Util/StreamHelper.h"
#include "llzk/Util/SymbolHelper.h"

#include <mlir/IR/IRMapping.h>
#include <mlir/IR/OpImplementation.h>

#include <llvm/ADT/MapVector.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/StringSet.h>

#include <optional>

// TableGen'd implementation files
#include "llzk/Dialect/Struct/IR/OpInterfaces.cpp.inc"

// TableGen'd implementation files
#define GET_OP_CLASSES
#include "llzk/Dialect/Struct/IR/Ops.cpp.inc"

using namespace mlir;
using namespace llzk::array;
using namespace llzk::function;

namespace llzk::component {

bool isInStruct(Operation *op) { return succeeded(getParentOfType<StructDefOp>(op)); }

FailureOr<StructDefOp> verifyInStruct(Operation *op) {
  FailureOr<StructDefOp> res = getParentOfType<StructDefOp>(op);
  if (failed(res)) {
    return op->emitOpError() << "only valid within a '" << StructDefOp::getOperationName()
                             << "' ancestor";
  }
  return res;
}

bool isInStructFunctionNamed(Operation *op, char const *funcName) {
  FailureOr<FuncDefOp> parentFuncOpt = getParentOfType<FuncDefOp>(op);
  if (succeeded(parentFuncOpt)) {
    FuncDefOp parentFunc = parentFuncOpt.value();
    if (isInStruct(parentFunc.getOperation())) {
      if (parentFunc.getSymName().compare(funcName) == 0) {
        return true;
      }
    }
  }
  return false;
}

// Again, only valid/implemented for StructDefOp
template <> LogicalResult SetFuncAllowAttrs<StructDefOp>::verifyTrait(Operation *structOp) {
  assert(llvm::isa<StructDefOp>(structOp));
  llvm::cast<StructDefOp>(structOp).getBody().walk([](FuncDefOp funcDef) {
    if (funcDef.nameIsConstrain()) {
      funcDef.setAllowConstraintAttr();
      funcDef.setAllowWitnessAttr(false);
    } else if (funcDef.nameIsCompute()) {
      funcDef.setAllowConstraintAttr(false);
      funcDef.setAllowWitnessAttr();
    } else if (funcDef.nameIsProduct()) {
      funcDef.setAllowConstraintAttr();
      funcDef.setAllowWitnessAttr();
    }
  });
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

static inline InFlightDiagnostic structFunDefError(Operation *origin) {
  return origin->emitError() << '\'' << StructDefOp::getOperationName() << "' op "
                             << "must define only \"@" << FUNC_NAME_COMPUTE << "\" and \"@"
                             << FUNC_NAME_CONSTRAIN << "\" functions, or a \"@" << FUNC_NAME_PRODUCT
                             << "\" function; ";
}

/// Verifies that the given `actualType` matches the `StructDefOp` given (i.e., for the "self" type
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
          actualStructType.getNameRef(), '"'
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
          .append("should be type of this '", StructDefOp::getOperationName(), '\'');
    }
  } else {
    return genCompareErr(expectedStruct, origin, aspect);
  }
  return success();
}

//===------------------------------------------------------------------===//
// StructDefOp
//===------------------------------------------------------------------===//

StructType StructDefOp::getType(std::optional<ArrayAttr> constParams) {
  auto pathRes = getPathFromRoot(*this);
  assert(succeeded(pathRes)); // consistent with StructType::get() with invalid args
  return StructType::get(pathRes.value(), constParams.value_or(getConstParamsAttr()));
}

std::string StructDefOp::getHeaderString() {
  return buildStringViaCallback([this](llvm::raw_ostream &ss) {
    FailureOr<SymbolRefAttr> pathToExpected = getPathFromRoot(*this);
    if (succeeded(pathToExpected)) {
      ss << pathToExpected.value();
    } else {
      // When there is a failure trying to get the resolved name of the struct,
      //  just print its symbol name directly.
      ss << '@' << this->getSymName();
    }
    if (auto attr = this->getConstParamsAttr()) {
      ss << '<' << attr << '>';
    }
  });
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
        return this->emitOpError().append("has more than one parameter named \"@", name, '"');
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

  std::string message = buildStringViaCallback([&inFunc, appendSelf](llvm::raw_ostream &ss) {
    ss << "\"@" << COMPONENT_NAME_MAIN << "\" component \"@" << inFunc.getSymName()
       << "\" function parameters must be one of: {";
    if (appendSelf) {
      ss << "!" << StructType::name << "<@" << COMPONENT_NAME_MAIN << ">, ";
    }
    ss << "!" << StructType::name << "<@" << COMPONENT_NAME_SIGNAL << ">, ";
    ss << "!" << ArrayType::name << "<.. x !" << StructType::name << "<@" << COMPONENT_NAME_SIGNAL
       << ">>}";
  });
  return inFunc.emitError(message);
}

inline LogicalResult verifyStructComputeConstrain(
    StructDefOp &structDef, FuncDefOp &computeFunc, FuncDefOp &constrainFunc
) {
  // ASSERT: The `SetFuncAllowAttrs` trait on StructDefOp set the attributes correctly.
  assert(constrainFunc.hasAllowConstraintAttr());
  assert(!computeFunc.hasAllowConstraintAttr());
  assert(!constrainFunc.hasAllowWitnessAttr());
  assert(computeFunc.hasAllowWitnessAttr());

  // Verify parameter types are valid. Skip the first parameter of the "constrain" function; it is
  // already checked via verifyFuncTypeConstrain() in Function/IR/Ops.cpp.
  ArrayRef<Type> computeParams = computeFunc.getFunctionType().getInputs();
  ArrayRef<Type> constrainParams = constrainFunc.getFunctionType().getInputs().drop_front();
  if (structDef.isMainComponent()) {
    // Verify that the Struct has no parameters.
    if (!isNullOrEmpty(structDef.getConstParamsAttr())) {
      return structDef.emitError().append(
          "The \"@", COMPONENT_NAME_MAIN, "\" component must have no parameters"
      );
    }
    // Verify the input parameter types are legal. The error message is explicit about what types
    // are allowed so there is no benefit to report multiple errors if more than one parameter in
    // the referenced function has an illegal type.
    for (Type t : computeParams) {
      if (failed(checkMainFuncParamType(t, computeFunc, false))) {
        return failure(); // checkMainFuncParamType() already emits a sufficient error message
      }
    }
    for (Type t : constrainParams) {
      if (failed(checkMainFuncParamType(t, constrainFunc, true))) {
        return failure(); // checkMainFuncParamType() already emits a sufficient error message
      }
    }
  }

  if (!typeListsUnify(computeParams, constrainParams)) {
    return constrainFunc.emitError()
        .append(
            "expected \"@", FUNC_NAME_CONSTRAIN,
            "\" function argument types (sans the first one) to match \"@", FUNC_NAME_COMPUTE,
            "\" function argument types"
        )
        .attachNote(computeFunc.getLoc())
        .append("\"@", FUNC_NAME_COMPUTE, "\" function defined here");
  }

  return success();
}

inline LogicalResult verifyStructProduct(StructDefOp &structDef, FuncDefOp &productFunc) {

  // ASSERT: The `SetFuncAllowAttrs` trait on StructDefOp set the attributes correctly
  assert(productFunc.hasAllowConstraintAttr());
  assert(productFunc.hasAllowWitnessAttr());

  // Verify parameter types are valid
  ArrayRef<Type> productParams = productFunc.getFunctionType().getInputs();
  if (structDef.isMainComponent()) {
    if (!isNullOrEmpty(structDef.getConstParamsAttr())) {
      return structDef.emitError().append(
          "The \"@", COMPONENT_NAME_MAIN, "\" component must have no parameters"
      );
    }
  }

  for (Type t : productParams) {
    if (failed(checkMainFuncParamType(t, productFunc, false))) {
      return failure();
    }
  }

  return success();
}

} // namespace

LogicalResult StructDefOp::verifyRegions() {
  assert(getBody().hasOneBlock()); // per ODS, SizedRegion<1>
  std::optional<FuncDefOp> foundCompute = std::nullopt;
  std::optional<FuncDefOp> foundConstrain = std::nullopt;
  std::optional<FuncDefOp> foundProduct = std::nullopt;
  {
    // Verify the following:
    // 1. The only ops within the body are field and function definitions
    // 2. The only functions defined in the struct are `@compute()` and `@constrain()`, or
    // `@product()`
    OwningEmitErrorFn emitError = getEmitOpErrFn(this);
    for (Operation &op : getBody().front()) {
      if (!llvm::isa<FieldDefOp>(op)) {
        if (FuncDefOp funcDef = llvm::dyn_cast<FuncDefOp>(op)) {
          if (funcDef.nameIsCompute()) {
            if (foundProduct) {
              return structFunDefError(funcDef.getOperation())
                     << "found both \"@" << FUNC_NAME_COMPUTE << "\" and \"@" << FUNC_NAME_PRODUCT
                     << "\" functions";
            }
            if (foundCompute) {
              return structFunDefError(funcDef.getOperation())
                     << "found multiple \"@" << FUNC_NAME_COMPUTE << "\" functions";
            }
            foundCompute = std::make_optional(funcDef);
          } else if (funcDef.nameIsConstrain()) {
            if (foundProduct) {
              return structFunDefError(funcDef.getOperation())
                     << "found both \"@" << FUNC_NAME_CONSTRAIN << "\" and \"@" << FUNC_NAME_PRODUCT
                     << "\" functions";
            }
            if (foundConstrain) {
              return structFunDefError(funcDef.getOperation())
                     << "found multiple \"@" << FUNC_NAME_CONSTRAIN << "\" functions";
            }
            foundConstrain = std::make_optional(funcDef);
          } else if (funcDef.nameIsProduct()) {
            if (foundCompute) {
              return structFunDefError(funcDef.getOperation())
                     << "found both \"@" << FUNC_NAME_COMPUTE << "\" and \"@" << FUNC_NAME_PRODUCT
                     << "\" functions";
            }
            if (foundConstrain) {
              return structFunDefError(funcDef.getOperation())
                     << "found both \"@" << FUNC_NAME_CONSTRAIN << "\" and \"@" << FUNC_NAME_PRODUCT
                     << "\" functions";
            }
            if (foundProduct) {
              return structFunDefError(funcDef.getOperation())
                     << "found multiple \"@" << FUNC_NAME_PRODUCT << "\" functions";
            }
            foundProduct = std::make_optional(funcDef);
          } else {
            // Must do a little more than a simple call to '?.emitOpError()' to
            // tag the error with correct location and correct op name.
            return structFunDefError(funcDef.getOperation())
                   << "found \"@" << funcDef.getSymName() << '"';
          }
        } else {
          return op.emitOpError() << "invalid operation in '" << StructDefOp::getOperationName()
                                  << "'; only '" << FieldDefOp::getOperationName() << '\''
                                  << " and '" << FuncDefOp::getOperationName()
                                  << "' operations are permitted";
        }
      }
    }
    if (!foundProduct.has_value() && !foundCompute.has_value()) {
      return structFunDefError(getOperation())
             << "missing \"@" << FUNC_NAME_COMPUTE << "\" or \"@" << FUNC_NAME_PRODUCT << "\"";
    }
    if (!foundProduct.has_value() && !foundConstrain.has_value()) {
      return structFunDefError(getOperation())
             << "missing \"@" << FUNC_NAME_CONSTRAIN << "\" or \"@" << FUNC_NAME_PRODUCT << "\"";
    }
  }

  if (foundCompute && foundConstrain) {
    return verifyStructComputeConstrain(*this, *foundCompute, *foundConstrain);
  }
  return verifyStructProduct(*this, *foundProduct);
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
    getOperation()->setAttr(PublicAttr::name, UnitAttr::get(getContext()));
  } else {
    getOperation()->removeAttr(PublicAttr::name);
  }
}

static LogicalResult
verifyFieldDefTypeImpl(Type fieldType, SymbolTableCollection &tables, Operation *origin) {
  if (StructType fieldStructType = llvm::dyn_cast<StructType>(fieldType)) {
    // Special case for StructType verifies that the field type can resolve and that it is NOT the
    // parent struct (i.e., struct fields cannot create circular references).
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
                              << tyStruct.getNameRef() << '"';
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
  // '!mapOperands.empty()' implies 'numDims.has_value()'
  assert(mapOperands.empty() || numDims.has_value());
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
    OpBuilder & /*odsBuilder*/, OperationState &odsState, TypeRange resultTypes,
    ValueRange operands, ArrayRef<NamedAttribute> attrs
) {
  odsState.addTypes(resultTypes);
  odsState.addOperands(operands);
  odsState.addAttributes(attrs);
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

} // namespace llzk::component
