#include "llzk/Dialect/LLZK/IR/Ops.h"
#include "llzk/Dialect/LLZK/IR/Types.h"
#include "llzk/Dialect/LLZK/Util/IncludeHelper.h"
#include "llzk/Dialect/LLZK/Util/SymbolHelper.h"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/Support/LogicalResult.h>

#include <llvm/ADT/Twine.h>

// TableGen'd implementation files
#define GET_OP_CLASSES
#include "llzk/Dialect/LLZK/IR/OpInterfaces.cpp.inc"

// TableGen'd implementation files
#define GET_OP_CLASSES
#include "llzk/Dialect/LLZK/IR/Ops.cpp.inc"

namespace llzk {

bool isInStruct(mlir::Operation *op) { return mlir::succeeded(getParentOfType<StructDefOp>(op)); }

mlir::LogicalResult verifyInStruct(mlir::Operation *op) {
  return isInStruct(op) ? mlir::success()
                        : op->emitOpError() << "only valid within a '"
                                            << getOperationName<StructDefOp>() << "' ancestor";
}

bool isInStructFunctionNamed(mlir::Operation *op, char const *funcName) {
  mlir::FailureOr<FuncOp> parentFuncOpt = getParentOfType<FuncOp>(op);
  if (mlir::succeeded(parentFuncOpt)) {
    FuncOp parentFunc = parentFuncOpt.value();
    mlir::FailureOr<StructDefOp> parentStruct =
        getParentOfType<StructDefOp>(parentFunc.getOperation());
    if (mlir::succeeded(parentStruct)) {
      if (parentFunc.getSymName().compare(funcName) == 0) {
        return true;
      }
    }
  }
  return false;
}

//===------------------------------------------------------------------===//
// IncludeOp (see IncludeHelper.cpp for other functions)
//===------------------------------------------------------------------===//

IncludeOp IncludeOp::create(mlir::Location loc, llvm::StringRef name, llvm::StringRef path) {
  return delegate_to_build<IncludeOp>(loc, name, path);
}

IncludeOp IncludeOp::create(mlir::Location loc, mlir::StringAttr name, mlir::StringAttr path) {
  return delegate_to_build<IncludeOp>(loc, name, path);
}

//===------------------------------------------------------------------===//
// StructDefOp
//===------------------------------------------------------------------===//
namespace {

using namespace mlir;

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
  mlir::FailureOr<mlir::SymbolRefAttr> pathToExpected = getPathFromRoot(*this);
  if (mlir::succeeded(pathToExpected)) {
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
      if (FlatSymbolRefAttr flatter = llvm::dyn_cast<FlatSymbolRefAttr>(attr)) {
        if (flatter.getRootReference() == find) {
          return true;
        }
      }
    }
  }
  return false;
}

mlir::LogicalResult StructDefOp::verifyRegions() {
  if (!getBody().hasOneBlock()) {
    return emitOpError() << "must contain exactly 1 block";
  }
  auto emitError = [this] { return this->emitOpError(); };
  bool foundCompute = false;
  bool foundConstrain = false;
  for (auto &op : getBody().front()) {
    if (!llvm::isa<FieldDefOp>(op)) {
      if (auto funcDef = llvm::dyn_cast<::llzk::FuncOp>(op)) {
        auto funcName = funcDef.getSymName();
        if (llzk::FUNC_NAME_COMPUTE == funcName) {
          if (foundCompute) {
            return msgOneFunction({emitError}, llzk::FUNC_NAME_COMPUTE);
          }
          foundCompute = true;
        } else if (llzk::FUNC_NAME_CONSTRAIN == funcName) {
          if (foundConstrain) {
            return msgOneFunction({emitError}, llzk::FUNC_NAME_CONSTRAIN);
          }
          foundConstrain = true;
        } else {
          // Must do a little more than a simple call to '?.emitOpError()' to
          // tag the error with correct location and correct op name.
          return op.emitError() << "'" << getOperationName() << "' op "
                                << "must define only 'compute' and 'constrain' functions;"
                                << " found '" << funcName << "'";
        }
      } else {
        return op.emitOpError() << "invalid operation in 'struct'; only 'field'"
                                << " and 'func' operations are permitted";
      }
    }
  }
  if (!foundCompute) {
    return msgOneFunction({emitError}, llzk::FUNC_NAME_COMPUTE);
  } else if (!foundConstrain) {
    return msgOneFunction({emitError}, llzk::FUNC_NAME_CONSTRAIN);
  }

  return mlir::success();
}

FieldDefOp StructDefOp::getFieldDef(mlir::StringAttr fieldName) {
  // The Body Region was verified to have exactly one Block so only need to search front() Block.
  for (mlir::Operation &op : getBody().front()) {
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

mlir::LogicalResult ConstReadOp::verifySymbolUses(SymbolTableCollection &tables) {
  FailureOr<StructDefOp> getParentRes = getParentOfType<StructDefOp>(*this);
  if (failed(getParentRes)) {
    return this->emitOpError().append(
        "only valid within a '", StructDefOp::getOperationName(), "'"
    );
  }
  if (!getParentRes->hasParamNamed(this->getConstNameAttr())) {
    return this->emitOpError()
        .append("references unknown symbol \"", this->getConstNameAttr(), "\"")
        .attachNote(getParentRes->getLoc())
        .append("must reference a parameter of this struct");
  }
  return mlir::success();
}

//===------------------------------------------------------------------===//
// FieldDefOp
//===------------------------------------------------------------------===//
bool FieldDefOp::hasPublicAttr() { return getOperation()->hasAttr(PublicAttr::name); }

mlir::LogicalResult FieldDefOp::verifySymbolUses(SymbolTableCollection &tables) {
  mlir::Type fieldType = this->getType();
  if (StructType fieldStructType = llvm::dyn_cast<StructType>(fieldType)) {
    // Special case for StructType verifies that the field type can resolve and that it is NOT the
    // parent struct (i.e. struct fields cannot create circular references).
    auto fieldTypeRes = verifyStructTypeResolution(tables, fieldStructType, *this);
    if (mlir::failed(fieldTypeRes)) {
      return mlir::failure(); // above already emits a sufficient error message
    }
    mlir::FailureOr<StructDefOp> parentRes = getParentOfType<StructDefOp>(*this);
    assert(mlir::succeeded(parentRes) && "FieldDefOp parent is always StructDefOp");
    if (fieldTypeRes.value() == parentRes.value()) {
      return this->emitOpError()
          .append("type is circular")
          .attachNote(parentRes.value().getLoc())
          .append("references parent component defined here");
    }
    return mlir::success();
  } else {
    return verifyTypeResolution(tables, fieldType, *this);
  }
}

//===------------------------------------------------------------------===//
// FieldRefOp implementations
//===------------------------------------------------------------------===//
namespace {
mlir::FailureOr<SymbolLookupResult<FieldDefOp>>
getFieldDefOp(FieldRefOpInterface refOp, mlir::SymbolTableCollection &tables, StructType tyStruct) {
  mlir::Operation *op = refOp.getOperation();
  auto structDef = tyStruct.getDefinition(tables, op);
  if (mlir::failed(structDef)) {
    return mlir::failure(); // getDefinition() already emits a sufficient error message
  }
  auto res = llzk::lookupSymbolIn<FieldDefOp>(
      tables, mlir::SymbolRefAttr::get(refOp->getContext(), refOp.getFieldName()),
      structDef.value().get(), op
  );
  if (mlir::failed(res)) {
    return refOp->emitError() << "no '" << FieldDefOp::getOperationName() << "' named \"@"
                              << refOp.getFieldName() << "\" in \"" << tyStruct.getNameRef()
                              << "\"";
  }
  return std::move(res.value());
}

inline mlir::FailureOr<SymbolLookupResult<FieldDefOp>>
getFieldDefOp(FieldRefOpInterface refOp, mlir::SymbolTableCollection &tables) {
  return getFieldDefOp(refOp, tables, refOp.getStructType());
}

mlir::LogicalResult verifySymbolUses(
    FieldRefOpInterface refOp, mlir::SymbolTableCollection &tables, mlir::Value compareTo
) {
  StructType tyStruct = refOp.getStructType();
  if (mlir::failed(tyStruct.verifySymbolRef(tables, refOp.getOperation()))) {
    return mlir::failure();
  }
  auto field = getFieldDefOp(refOp, tables, tyStruct);
  if (mlir::failed(field)) {
    return field; // getFieldDefOp() already emits a sufficient error message
  }
  mlir::Type fieldType = field->get().getType();

  if (!typesUnify(compareTo.getType(), fieldType, field->getIncludeSymNames())) {
    return refOp->emitOpError() << "has wrong type; expected " << fieldType << ", got "
                                << compareTo.getType();
  }
  return mlir::success();
}
} // namespace

mlir::FailureOr<SymbolLookupResult<FieldDefOp>>
FieldReadOp::getFieldDefOp(mlir::SymbolTableCollection &tables) {
  return llzk::getFieldDefOp(*this, tables);
}

mlir::LogicalResult FieldReadOp::verifySymbolUses(mlir::SymbolTableCollection &tables) {
  return llzk::verifySymbolUses(*this, tables, getResult());
}

mlir::FailureOr<SymbolLookupResult<FieldDefOp>>
FieldWriteOp::getFieldDefOp(mlir::SymbolTableCollection &tables) {
  return llzk::getFieldDefOp(*this, tables);
}

mlir::LogicalResult FieldWriteOp::verifySymbolUses(mlir::SymbolTableCollection &tables) {
  return llzk::verifySymbolUses(*this, tables, getVal());
}

//===------------------------------------------------------------------===//
// FeltConstantOp
//===------------------------------------------------------------------===//

void FeltConstantOp::getAsmResultNames(mlir::OpAsmSetValueNameFn setNameFn) {
  llvm::SmallString<32> buf;
  llvm::raw_svector_ostream os(buf);
  os << "felt_const_";
  getValue().getValue().toStringUnsigned(buf);
  setNameFn(getResult(), buf);
}

mlir::OpFoldResult FeltConstantOp::fold(FeltConstantOp::FoldAdaptor) { return getValue(); }

//===------------------------------------------------------------------===//
// FeltNonDetOp
//===------------------------------------------------------------------===//

void FeltNonDetOp::getAsmResultNames(mlir::OpAsmSetValueNameFn setNameFn) {
  setNameFn(getResult(), "felt_nondet");
}

//===------------------------------------------------------------------===//
// CreateArrayOp
//===------------------------------------------------------------------===//

void CreateArrayOp::getAsmResultNames(mlir::OpAsmSetValueNameFn setNameFn) {
  setNameFn(getResult(), "array");
}

llvm::SmallVector<mlir::Type> CreateArrayOp::resultTypeToElementsTypes(mlir::Type resultType) {
  // The ODS restricts $result with LLZK_ArrayType so this cast is safe.
  ArrayType a = llvm::cast<ArrayType>(resultType);
  return llvm::SmallVector<mlir::Type>(a.getNumElements(), a.getElementType());
}

mlir::ParseResult CreateArrayOp::parseInferredArrayType(
    mlir::AsmParser &parser, llvm::SmallVector<mlir::Type, 1> &elementsTypes,
    mlir::ArrayRef<mlir::OpAsmParser::UnresolvedOperand> elements, mlir::Type resultType
) {
  assert(elementsTypes.size() == 0); // it was not yet initialized
  // If the '$elements' operand is not empty, then the expected type for the operand
  //  is computed to match the type of the '$result'. Otherwise, it remains empty.
  if (elements.size() > 0) {
    elementsTypes.append(resultTypeToElementsTypes(resultType));
  }
  return mlir::ParseResult::success();
}

void CreateArrayOp::printInferredArrayType(
    mlir::AsmPrinter &printer, CreateArrayOp, mlir::Operation::operand_range::type_range,
    mlir::Operation::operand_range, mlir::Type
) {
  // nothing to print, it's derived and therefore not represented in the output
}

//===------------------------------------------------------------------===//
// ReadArrayOp
//===------------------------------------------------------------------===//

mlir::LogicalResult ReadArrayOp::inferReturnTypes(
    mlir::MLIRContext *context, std::optional<mlir::Location> location, ReadArrayOpAdaptor adaptor,
    ::llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes
) {
  inferredReturnTypes.resize(1);
  mlir::Type lvalType = adaptor.getLvalue().getType();
  assert(llvm::isa<ArrayType>(lvalType)); // per ODS spec of ReadArrayOp
  inferredReturnTypes[0] = llvm::cast<ArrayType>(lvalType).getElementType();
  return mlir::success();
}

bool ReadArrayOp::isCompatibleReturnTypes(mlir::TypeRange l, mlir::TypeRange r) {
  // There is a single return type per ODS spec of ReadArrayOp
  return l.size() == 1 && r.size() == 1 && typesUnify(l.front(), r.front());
}

//===------------------------------------------------------------------===//
// EmitEqualityOp
//===------------------------------------------------------------------===//

mlir::Type EmitEqualityOp::inferRHS(mlir::Type lhsType) { return lhsType; }

//===------------------------------------------------------------------===//
// EmitContainmentOp
//===------------------------------------------------------------------===//

mlir::Type EmitContainmentOp::inferRHS(mlir::Type lhsType) {
  assert(llvm::isa<ArrayType>(lhsType)); // per ODS spec of EmitContainmentOp
  return llvm::cast<ArrayType>(lhsType).getElementType();
}

//===------------------------------------------------------------------===//
// CreateStructOp
//===------------------------------------------------------------------===//

void CreateStructOp::getAsmResultNames(mlir::OpAsmSetValueNameFn setNameFn) {
  setNameFn(getResult(), "self");
}

} // namespace llzk
