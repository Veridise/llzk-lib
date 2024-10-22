#include "Dialect/ZKIR/IR/Ops.h"
#include "Dialect/ZKIR/IR/Types.h"
#include "Dialect/ZKIR/Util/SymbolHelper.h"

#include <mlir/IR/Diagnostics.h>

#include <llvm/ADT/Twine.h>

// TableGen'd implementation files
#define GET_OP_CLASSES
#include "Dialect/ZKIR/IR/OpInterfaces.cpp.inc"

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
      if (auto func_def = llvm::dyn_cast<::zkir::FuncOp>(op)) {
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
// FieldDefOp
// -----
bool FieldDefOp::hasPublicAttr() { return getOperation()->hasAttr(PublicAttr::name); }

// -----
// FieldRefOp implementations
// -----
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
    return refOp->emitError() << "no field named \"@" << refOp.getFieldName() << "\" in \""
                              << tyStruct.getName() << "\"";
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
    return refOp->emitOpError() << "field " << kind << " has wrong type; expected " << fieldType
                                << ", got " << compareTo.getType();
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

// -----
// Emit*Op
// -----

namespace {

inline mlir::LogicalResult verifyEmitOp(Operation *op) {
  // No need for dyn_cast due to HasParent<"::zkir::FuncOp"> trait
  auto func_name = llvm::cast<::zkir::FuncOp>(op->getParentOp()).getSymName();
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
