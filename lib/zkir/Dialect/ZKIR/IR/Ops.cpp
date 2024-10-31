#include "zkir/Dialect/ZKIR/IR/Ops.h"
#include "zkir/Dialect/ZKIR/IR/Types.h"
#include "zkir/Dialect/ZKIR/Util/SymbolHelper.h"

#include <mlir/IR/Diagnostics.h>

#include <llvm/ADT/Twine.h>

// TableGen'd implementation files
#define GET_OP_CLASSES
#include "zkir/Dialect/ZKIR/IR/OpInterfaces.cpp.inc"

// TableGen'd implementation files
#define GET_OP_CLASSES
#include "zkir/Dialect/ZKIR/IR/Ops.cpp.inc"

namespace zkir {

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
// StructDefOp
//===------------------------------------------------------------------===//
namespace {

using namespace mlir;

inline LogicalResult
msgOneFunction(function_ref<InFlightDiagnostic()> emitError, const Twine &name) {
  return emitError() << "must define exactly one '" << name << "' function";
}

} // namespace

mlir::LogicalResult StructDefOp::verifyRegions() {
  if (!getBody().hasOneBlock()) {
    return emitOpError() << "must contain exactly 1 block";
  }
  auto emitError = [this] { return this->emitOpError(); };
  bool foundCompute = false;
  bool foundConstrain = false;
  for (auto &op : getBody().front()) {
    if (!llvm::isa<FieldDefOp>(op)) {
      if (auto funcDef = llvm::dyn_cast<::zkir::FuncOp>(op)) {
        auto funcName = funcDef.getSymName();
        if (zkir::FUNC_NAME_COMPUTE == funcName) {
          if (foundCompute) {
            return msgOneFunction({emitError}, zkir::FUNC_NAME_COMPUTE);
          }
          foundCompute = true;
        } else if (zkir::FUNC_NAME_CONSTRAIN == funcName) {
          if (foundConstrain) {
            return msgOneFunction({emitError}, zkir::FUNC_NAME_CONSTRAIN);
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
    return msgOneFunction({emitError}, zkir::FUNC_NAME_COMPUTE);
  } else if (!foundConstrain) {
    return msgOneFunction({emitError}, zkir::FUNC_NAME_CONSTRAIN);
  }

  return mlir::success();
}

FieldDefOp StructDefOp::getFieldDef(::mlir::StringAttr fieldName) {
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
// FieldDefOp
//===------------------------------------------------------------------===//
bool FieldDefOp::hasPublicAttr() { return getOperation()->hasAttr(PublicAttr::name); }

mlir::LogicalResult FieldDefOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  return verifyTypeResolution(symbolTable, this->getType(), *this);
}

//===------------------------------------------------------------------===//
// FieldRefOp implementations
//===------------------------------------------------------------------===//
namespace {
mlir::FailureOr<FieldDefOp> getFieldDefOp(
    FieldRefOpInterface refOp, mlir::SymbolTableCollection &symbolTable, StructType tyStruct
) {
  mlir::Operation *op = refOp.getOperation();
  mlir::FailureOr<StructDefOp> structDef = tyStruct.getDefinition(symbolTable, op);
  if (mlir::failed(structDef)) {
    return structDef; // getDefinition() already emits a sufficient error message
  }
  auto res = zkir::lookupSymbolIn<FieldDefOp, mlir::SymbolRefAttr>(
      symbolTable, mlir::SymbolRefAttr::get(refOp->getContext(), refOp.getFieldName()),
      structDef.value(), op
  );
  if (mlir::failed(res)) {
    return refOp->emitError() << "no '" << FieldDefOp::getOperationName() << "' named \"@"
                              << refOp.getFieldName() << "\" in \"" << tyStruct.getName() << "\"";
  }
  return res;
}

inline mlir::FailureOr<FieldDefOp>
getFieldDefOp(FieldRefOpInterface refOp, mlir::SymbolTableCollection &symbolTable) {
  return getFieldDefOp(refOp, symbolTable, refOp.getStructType());
}

mlir::LogicalResult verifySymbolUses(
    FieldRefOpInterface refOp, mlir::SymbolTableCollection &symbolTable, mlir::Value compareTo,
    const char *kind
) {
  StructType tyStruct = refOp.getStructType();
  if (mlir::failed(tyStruct.verifySymbol(symbolTable, refOp.getOperation()))) {
    return mlir::failure();
  }
  mlir::FailureOr<FieldDefOp> field = getFieldDefOp(refOp, symbolTable, tyStruct);
  if (mlir::failed(field)) {
    return field; // getFieldDefOp() already emits a sufficient error message
  }
  mlir::Type fieldType = field.value().getType();
  if (fieldType != compareTo.getType()) {
    return refOp->emitOpError() << "has wrong type; expected " << fieldType << ", got "
                                << compareTo.getType();
  }
  return mlir::success();
}
} // namespace

mlir::FailureOr<FieldDefOp> FieldReadOp::getFieldDefOp(mlir::SymbolTableCollection &symbolTable) {
  return zkir::getFieldDefOp(*this, symbolTable);
}

mlir::LogicalResult FieldReadOp::verifySymbolUses(::mlir::SymbolTableCollection &symbolTable) {
  return zkir::verifySymbolUses(*this, symbolTable, getResult(), "read");
}

mlir::FailureOr<FieldDefOp> FieldWriteOp::getFieldDefOp(mlir::SymbolTableCollection &symbolTable) {
  return zkir::getFieldDefOp(*this, symbolTable);
}

mlir::LogicalResult FieldWriteOp::verifySymbolUses(::mlir::SymbolTableCollection &symbolTable) {
  return zkir::verifySymbolUses(*this, symbolTable, getVal(), "write");
}

//===------------------------------------------------------------------===//
// FeltConstantOp
//===------------------------------------------------------------------===//

void FeltConstantOp::getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn) {
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

void FeltNonDetOp::getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn) {
  setNameFn(getResult(), "felt_nondet");
}

//===------------------------------------------------------------------===//
// CreateArrayOp
//===------------------------------------------------------------------===//

void CreateArrayOp::getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn) {
  setNameFn(getResult(), "array");
}

//===------------------------------------------------------------------===//
// CreateStructOp
//===------------------------------------------------------------------===//

void CreateStructOp::getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn) {
  setNameFn(getResult(), "self");
}

} // namespace zkir
