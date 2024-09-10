#include "Dialect/ZKIR/IR/Ops.h"
#include "Dialect/ZKIR/IR/Types.h"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Diagnostics.h>

#include <llvm/ADT/Twine.h>

// TableGen'd implementation files
#define GET_OP_CLASSES
#include "Dialect/ZKIR/IR/Ops.cpp.inc"

namespace {

using namespace mlir;

inline LogicalResult
msgOneFunction(function_ref<InFlightDiagnostic()> emitError, const Twine &name) {
  return emitError() << "must define exactly one '" << name << "' function";
}

} // namespace

namespace zkir {

// -----
// StructDefOp
// -----

mlir::LogicalResult StructDefOp::verifyRegions() {
  if (!getBody().hasOneBlock()) {
    return emitOpError() << "must contain exactly 1 block";
  }
  auto emitError = [this] { return this->emitOpError(); };
  bool foundCompute = false;
  bool foundConstrain = false;
  for (auto &op : getBody().front()) {
    if (!llvm::isa<FieldOp>(op)) {
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
// CreateArrayOp
// -----

void CreateArrayOp::getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn) {
  setNameFn(getResult(), "array");
}

} // namespace zkir