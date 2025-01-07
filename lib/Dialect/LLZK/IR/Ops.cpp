#include "llzk/Dialect/LLZK/IR/Ops.h"
#include "llzk/Dialect/LLZK/IR/Types.h"
#include "llzk/Dialect/LLZK/Util/IncludeHelper.h"
#include "llzk/Dialect/LLZK/Util/SymbolHelper.h"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/Support/LogicalResult.h>

#include <llvm/ADT/StringSet.h>
#include <llvm/ADT/Twine.h>

// TableGen'd implementation files
#define GET_OP_CLASSES
#include "llzk/Dialect/LLZK/IR/OpInterfaces.cpp.inc"

// TableGen'd implementation files
#define GET_OP_CLASSES
#include "llzk/Dialect/LLZK/IR/Ops.cpp.inc"

namespace llzk {

using namespace mlir;

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
    FailureOr<StructDefOp> parentStruct = getParentOfType<StructDefOp>(parentFunc.getOperation());
    if (succeeded(parentStruct)) {
      if (parentFunc.getSymName().compare(funcName) == 0) {
        return true;
      }
    }
  }
  return false;
}

template <typename ConcreteType> LogicalResult InStruct<ConcreteType>::verifyTrait(Operation *op) {
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
// StructDefOp
//===------------------------------------------------------------------===//
namespace {

inline LogicalResult
msgOneFunction(function_ref<InFlightDiagnostic()> emitError, const Twine &name) {
  return emitError() << "must define exactly one '" << name << "' function";
}

} // namespace

StructType StructDefOp::getType(std::optional<ArrayAttr> constParams) {
  auto pathRes = getPathFromRoot(*this);
  assert(succeeded(pathRes)); // consistent with StructType::get() with invalid args
  return StructType::get(getContext(), pathRes.value(), constParams.value_or(getConstParamsAttr()));
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

LogicalResult StructDefOp::verifyRegions() {
  if (!getBody().hasOneBlock()) {
    return emitOpError() << "must contain exactly 1 block";
  }
  auto emitError = [this] { return this->emitOpError(); };
  std::optional<FuncOp> foundCompute = std::nullopt;
  std::optional<FuncOp> foundConstrain = std::nullopt;
  for (auto &op : getBody().front()) {
    if (!llvm::isa<FieldDefOp>(op)) {
      if (FuncOp funcDef = llvm::dyn_cast<FuncOp>(op)) {
        auto funcName = funcDef.getSymName();
        if (FUNC_NAME_COMPUTE == funcName) {
          if (foundCompute) {
            return msgOneFunction(emitError, FUNC_NAME_COMPUTE);
          }
          foundCompute = std::make_optional(funcDef);
        } else if (FUNC_NAME_CONSTRAIN == funcName) {
          if (foundConstrain) {
            return msgOneFunction(emitError, FUNC_NAME_CONSTRAIN);
          }
          foundConstrain = std::make_optional(funcDef);
        } else {
          // Must do a little more than a simple call to '?.emitOpError()' to
          // tag the error with correct location and correct op name.
          return op.emitError() << "'" << getOperationName() << "' op " << "must define only \"@"
                                << FUNC_NAME_COMPUTE << "\" and \"@" << FUNC_NAME_CONSTRAIN
                                << "\" functions;" << " found \"@" << funcName << "\"";
        }
      } else {
        return op.emitOpError() << "invalid operation in '" << StructDefOp::getOperationName()
                                << "'; only '" << FieldDefOp::getOperationName() << "'" << " and '"
                                << FuncOp::getOperationName() << "' operations are permitted";
      }
    }
  }
  if (!foundCompute.has_value()) {
    return msgOneFunction(emitError, FUNC_NAME_COMPUTE);
  }
  if (!foundConstrain.has_value()) {
    return msgOneFunction(emitError, FUNC_NAME_CONSTRAIN);
  }

  // Ensure function input types from compute and constrain match, sans the first parameter of
  // constrain which is the instance of the parent struct.
  if (!typeListsUnify(
          foundCompute.value().getFunctionType().getInputs(),
          foundConstrain.value().getFunctionType().getInputs().drop_front()
      )) {
    return foundConstrain.value()
        .emitError()
        .append(
            "expected \"@", FUNC_NAME_CONSTRAIN,
            "\" function argument types (sans the first one) to match \"@", FUNC_NAME_COMPUTE,
            "\" function argument types"
        )
        .attachNote(foundCompute.value().getLoc())
        .append("\"@", FUNC_NAME_COMPUTE, "\" function defined here");
  }

  return success();
}

FieldDefOp StructDefOp::getFieldDef(StringAttr fieldName) {
  // The Body Region was verified to have exactly one Block so only need to search front() Block.
  for (Operation &op : getBody().front()) {
    if (FieldDefOp fieldDef = llvm::dyn_cast_if_present<FieldDefOp>(op)) {
      if (fieldName.compare(fieldDef.getSymNameAttr()) == 0) {
        return fieldDef;
      }
    }
  }
  return nullptr;
}

//===------------------------------------------------------------------===//
// ConstReadOp
//===------------------------------------------------------------------===//

LogicalResult ConstReadOp::verifySymbolUses(SymbolTableCollection &tables) {
  FailureOr<StructDefOp> getParentRes = verifyInStruct(*this);
  if (failed(getParentRes)) {
    return failure(); // verifyInStruct() already emits a sufficient error message
  }
  if (!getParentRes->hasParamNamed(this->getConstNameAttr())) {
    return this->emitOpError()
        .append("references unknown symbol \"", this->getConstNameAttr(), "\"")
        .attachNote(getParentRes->getLoc())
        .append("must reference a parameter of this struct");
  }
  return success();
}

//===------------------------------------------------------------------===//
// FieldDefOp
//===------------------------------------------------------------------===//
bool FieldDefOp::hasPublicAttr() { return getOperation()->hasAttr(PublicAttr::name); }

LogicalResult FieldDefOp::verifySymbolUses(SymbolTableCollection &tables) {
  Type fieldType = this->getType();
  if (StructType fieldStructType = llvm::dyn_cast<StructType>(fieldType)) {
    // Special case for StructType verifies that the field type can resolve and that it is NOT the
    // parent struct (i.e. struct fields cannot create circular references).
    auto fieldTypeRes = verifyStructTypeResolution(tables, fieldStructType, *this);
    if (failed(fieldTypeRes)) {
      return failure(); // above already emits a sufficient error message
    }
    FailureOr<StructDefOp> parentRes = getParentOfType<StructDefOp>(*this);
    assert(succeeded(parentRes) && "FieldDefOp parent is always StructDefOp"); // per ODS def
    if (fieldTypeRes.value() == parentRes.value()) {
      return this->emitOpError()
          .append("type is circular")
          .attachNote(parentRes.value().getLoc())
          .append("references parent component defined here");
    }
    return success();
  } else {
    return verifyTypeResolution(tables, fieldType, *this);
  }
}

//===------------------------------------------------------------------===//
// FieldRefOp implementations
//===------------------------------------------------------------------===//
namespace {
FailureOr<SymbolLookupResult<FieldDefOp>>
getFieldDefOp(FieldRefOpInterface refOp, SymbolTableCollection &tables, StructType tyStruct) {
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
    return refOp->emitError() << "no '" << FieldDefOp::getOperationName() << "' named \"@"
                              << refOp.getFieldName() << "\" in \"" << tyStruct.getNameRef()
                              << "\"";
  }
  return std::move(res.value());
}

inline FailureOr<SymbolLookupResult<FieldDefOp>>
getFieldDefOp(FieldRefOpInterface refOp, SymbolTableCollection &tables) {
  return getFieldDefOp(refOp, tables, refOp.getStructType());
}

LogicalResult
verifySymbolUses(FieldRefOpInterface refOp, SymbolTableCollection &tables, Value compareTo) {
  StructType tyStruct = refOp.getStructType();
  if (failed(tyStruct.verifySymbolRef(tables, refOp.getOperation()))) {
    return failure();
  }
  auto field = getFieldDefOp(refOp, tables, tyStruct);
  if (failed(field)) {
    return field; // getFieldDefOp() already emits a sufficient error message
  }
  Type fieldType = field->get().getType();

  if (!typesUnify(compareTo.getType(), fieldType, field->getIncludeSymNames())) {
    return refOp->emitOpError() << "has wrong type; expected " << fieldType << ", got "
                                << compareTo.getType();
  }
  return success();
}
} // namespace

FailureOr<SymbolLookupResult<FieldDefOp>> FieldReadOp::getFieldDefOp(SymbolTableCollection &tables
) {
  return llzk::getFieldDefOp(*this, tables);
}

LogicalResult FieldReadOp::verifySymbolUses(SymbolTableCollection &tables) {
  return llzk::verifySymbolUses(*this, tables, getResult());
}

FailureOr<SymbolLookupResult<FieldDefOp>> FieldWriteOp::getFieldDefOp(SymbolTableCollection &tables
) {
  return llzk::getFieldDefOp(*this, tables);
}

LogicalResult FieldWriteOp::verifySymbolUses(SymbolTableCollection &tables) {
  FailureOr<StructDefOp> getParentRes = verifyInStruct(*this);
  if (failed(getParentRes)) {
    return failure(); // verifyInStruct() already emits a sufficient error message
  }
  if (failed(checkSelfType(tables, *getParentRes, this->getComponent().getType(), *this, "result")
      )) {
    return failure();
  }
  return llzk::verifySymbolUses(*this, tables, getVal());
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

void CreateArrayOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  setNameFn(getResult(), "array");
}

llvm::SmallVector<Type> CreateArrayOp::resultTypeToElementsTypes(Type resultType) {
  // The ODS restricts $result with LLZK_ArrayType so this cast is safe.
  ArrayType a = llvm::cast<ArrayType>(resultType);
  return llvm::SmallVector<Type>(a.getNumElements(), a.getElementType());
}

ParseResult CreateArrayOp::parseInferredArrayType(
    AsmParser &parser, llvm::SmallVector<Type, 1> &elementsTypes,
    ArrayRef<OpAsmParser::UnresolvedOperand> elements, Type resultType
) {
  assert(elementsTypes.size() == 0); // it was not yet initialized
  // If the '$elements' operand is not empty, then the expected type for the operand
  //  is computed to match the type of the '$result'. Otherwise, it remains empty.
  if (elements.size() > 0) {
    elementsTypes.append(resultTypeToElementsTypes(resultType));
  }
  return ParseResult::success();
}

void CreateArrayOp::printInferredArrayType(
    AsmPrinter &printer, CreateArrayOp, Operation::operand_range::type_range,
    Operation::operand_range, Type
) {
  // nothing to print, it's derived and therefore not represented in the output
}

//===------------------------------------------------------------------===//
// ReadArrayOp
//===------------------------------------------------------------------===//

LogicalResult ReadArrayOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ReadArrayOpAdaptor adaptor,
    ::llvm::SmallVectorImpl<Type> &inferredReturnTypes
) {
  inferredReturnTypes.resize(1);
  Type lvalType = adaptor.getLvalue().getType();
  assert(llvm::isa<ArrayType>(lvalType)); // per ODS spec of ReadArrayOp
  inferredReturnTypes[0] = llvm::cast<ArrayType>(lvalType).getElementType();
  return success();
}

bool ReadArrayOp::isCompatibleReturnTypes(TypeRange l, TypeRange r) {
  // There is a single return type per ODS spec of ReadArrayOp
  return l.size() == 1 && r.size() == 1 && typesUnify(l.front(), r.front());
}

//===------------------------------------------------------------------===//
// EmitEqualityOp
//===------------------------------------------------------------------===//

Type EmitEqualityOp::inferRHS(Type lhsType) { return lhsType; }

//===------------------------------------------------------------------===//
// EmitContainmentOp
//===------------------------------------------------------------------===//

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

} // namespace llzk
