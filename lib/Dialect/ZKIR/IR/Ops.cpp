#include "Dialect/ZKIR/IR/Ops.h"
#include "Dialect/ZKIR/IR/Types.h"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Diagnostics.h>

#include <llvm/ADT/Twine.h>

// TableGen'd implementation files
#define GET_OP_CLASSES
#include "Dialect/ZKIR/IR/Ops.cpp.inc"

namespace zkir {

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
      if (auto func_def = llvm::dyn_cast<mlir::func::FuncOp>(op)) {
        auto func_name = func_def.getSymName();
        if ("compute" == func_name) {
          if (foundCompute) {
            return msgOneFunction({emitError}, "compute");
          }
          foundCompute = true;
        } else if ("constrain" == func_name) {
          if (foundConstrain) {
            return msgOneFunction({emitError}, "constrain");
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
    return msgOneFunction({emitError}, "compute");
  } else if (!foundConstrain) {
    return msgOneFunction({emitError}, "constrain");
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
// RefFieldOp
// -----

StructType RefFieldOp::getStructType() { return getComponent().getType().cast<StructType>(); }

FieldDefOp RefFieldOp::getFieldDefOp(mlir::SymbolTableCollection &symbolTable) {
  return llvm::dyn_cast_if_present<FieldDefOp>(symbolTable.lookupSymbolIn(
      getStructType().getDefinition(symbolTable, getOperation()),
      mlir::SymbolRefAttr::get(getContext(), getFieldName())
  ));
}

mlir::LogicalResult RefFieldOp::verifySymbolUses(::mlir::SymbolTableCollection &symbolTable) {
  if (mlir::failed(getStructType().verifySymbol(symbolTable, getOperation()))) {
    return mlir::failure();
  }
  FieldDefOp field = getFieldDefOp(symbolTable);
  if (!field) {
    return emitOpError() << "undefined struct field: @" << getFieldName();
  }
  if (field.getType() != getResult().getType()) {
    return emitOpError() << "field ref has wrong type; expected " << field.getType() << ", got "
                         << getResult().getType();
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
// Emit*Op
// -----

namespace {

inline mlir::LogicalResult verifyEmitOp(Operation *op) {
  // No need for dyn_cast due to HasParent<"mlir::func::FuncOp"> trait
  auto func_name = llvm::cast<mlir::func::FuncOp>(op->getParentOp()).getSymName();
  if ("constrain" == func_name) {
    return mlir::success();
  } else {
    return op->emitOpError() << "'emit' operation is only allowed within 'constrain' functions.";
  }
}

} // namespace

mlir::LogicalResult EmitEqualityOp::verify() { return verifyEmitOp(getOperation()); }

mlir::LogicalResult EmitContainmentOp::verify() { return verifyEmitOp(getOperation()); }

} // namespace zkir
