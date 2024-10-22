#include "Dialect/ZKIR/IR/Ops.h"
#include "Dialect/ZKIR/IR/Types.h"

#include <mlir/IR/Diagnostics.h>

#include <llvm/ADT/Twine.h>

// TableGen'd implementation files
#define GET_OP_CLASSES
#include "Dialect/ZKIR/IR/Ops.cpp.inc"

namespace zkir {

mlir::FailureOr<llvm::StringRef> getParentStructName(mlir::Operation *op) {
  if (zkir::StructDefOp sDef = op->getParentOfType<zkir::StructDefOp>()) {
    return sDef.getSymName();
  } else {
    return mlir::failure();
  }
}

mlir::FailureOr<llvm::StringRef> getParentFuncName(mlir::Operation *op) {
  if (zkir::FuncOp func = op->getParentOfType<zkir::FuncOp>()) {
    return func.getSymName();
  } else {
    return mlir::failure();
  }
}

// -----
// StructDefOp
// -----
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
      if (auto func_def = llvm::dyn_cast<::zkir::FuncOp>(op)) {
        auto func_name = func_def.getSymName();
        if (zkir::FUNC_NAME_COMPUTE == func_name) {
          if (foundCompute) {
            return msgOneFunction({emitError}, zkir::FUNC_NAME_COMPUTE);
          }
          foundCompute = true;
        } else if (zkir::FUNC_NAME_CONSTRAIN == func_name) {
          if (foundConstrain) {
            return msgOneFunction({emitError}, zkir::FUNC_NAME_CONSTRAIN);
          }
          foundConstrain = true;
        } else {
          // Must do a little more than a simple call to '?.emitOpError()' to
          // tag the error with correct location and correct op name.
          return op.emitError() << "'" << getOperationName() << "' op "
                                << "must define only 'compute' and 'constrain' functions;"
                                << " found '" << func_name << "'";
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

// -----
// FieldDefOp
// -----
bool FieldDefOp::hasPublicAttr() { return getOperation()->hasAttr(PublicAttr::name); }

// -----
// FieldRefOp implementations
// -----

namespace {
template <class T> zkir::StructType getStructType(T refOp) {
  return refOp->getComponent().getType().template cast<zkir::StructType>();
}

template <class T>
zkir::FieldDefOp getFieldDefOp(T refOp, mlir::SymbolTableCollection &symbolTable) {
  return llvm::dyn_cast_if_present<FieldDefOp>(symbolTable.lookupSymbolIn(
      refOp->getStructType().getDefinition(symbolTable, refOp->getOperation()),
      mlir::SymbolRefAttr::get(refOp->getContext(), refOp->getFieldName())
  ));
}
} // namespace

StructType FieldReadOp::getStructType() { return zkir::getStructType(this); }

FieldDefOp FieldReadOp::getFieldDefOp(mlir::SymbolTableCollection &symbolTable) {
  return zkir::getFieldDefOp(this, symbolTable);
}

mlir::LogicalResult FieldReadOp::verifySymbolUses(::mlir::SymbolTableCollection &symbolTable) {
  if (mlir::failed(getStructType().verifySymbol(symbolTable, getOperation()))) {
    return mlir::failure();
  }
  FieldDefOp field = getFieldDefOp(symbolTable);
  if (!field) {
    return emitOpError() << "undefined struct field: @" << getFieldName();
  }
  if (field.getType() != getResult().getType()) {
    return emitOpError() << "field read has wrong type; expected " << field.getType() << ", got "
                         << getResult().getType();
  }
  return mlir::success();
}

StructType FieldWriteOp::getStructType() { return zkir::getStructType(this); }

FieldDefOp FieldWriteOp::getFieldDefOp(mlir::SymbolTableCollection &symbolTable) {
  return zkir::getFieldDefOp(this, symbolTable);
}

mlir::LogicalResult FieldWriteOp::verifySymbolUses(::mlir::SymbolTableCollection &symbolTable) {
  if (mlir::failed(getStructType().verifySymbol(symbolTable, getOperation()))) {
    return mlir::failure();
  }
  FieldDefOp field = getFieldDefOp(symbolTable);
  if (!field) {
    return emitOpError() << "undefined struct field: @" << getFieldName();
  }
  if (field.getType() != getVal().getType()) {
    return emitOpError() << "field write has wrong type; expected " << field.getType() << ", got "
                         << getVal().getType();
  }
  return mlir::success();
}

// -----
// FeltConstantOp
// -----

void FeltConstantOp::getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn) {
  llvm::SmallString<32> buf;
  llvm::raw_svector_ostream os(buf);
  os << "felt_const_";
  getValue().getValue().toStringUnsigned(buf);
  setNameFn(getResult(), buf);
}

mlir::OpFoldResult FeltConstantOp::fold(FeltConstantOp::FoldAdaptor) { return getValue(); }

// -----
// FeltNonDetOp
// -----

void FeltNonDetOp::getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn) {
  setNameFn(getResult(), "felt_nondet");
}

// -----
// CreateArrayOp
// -----

void CreateArrayOp::getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn) {
  setNameFn(getResult(), "array");
}

// -----
// CreateStructOp
// -----

void CreateStructOp::getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn) {
  setNameFn(getResult(), "self");
}

} // namespace zkir
