#pragma once

#include "Dialect/ZKIR/IR/Attrs.h"
#include "Dialect/ZKIR/IR/Dialect.h"
#include "Dialect/ZKIR/IR/Types.h"

#include <mlir/Bytecode/BytecodeOpInterface.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/Interfaces/CallInterfaces.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>
#include <mlir/Interfaces/DataLayoutInterfaces.h>
#include <mlir/Interfaces/FunctionInterfaces.h>
#include <mlir/Interfaces/InferTypeOpInterface.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>

#include <optional>

// Types that must come before the "Ops.h.inc" import
namespace zkir {

constexpr char FUNC_NAME_COMPUTE[] = "compute";
constexpr char FUNC_NAME_CONSTRAIN[] = "constrain";

mlir::FailureOr<llvm::StringRef> getParentStructName(mlir::Operation *op);
mlir::FailureOr<llvm::StringRef> getParentFuncName(mlir::Operation *op);

/// This only exists so the compiler doesn't complain about incomplete types.
template <class OpType> llvm::StringLiteral getOperationName() {
  return OpType::getOperationName();
}

/// This class provides a verifier for ops that are expecting to have
/// an ancestor zkir::StructDefOp.
template <typename ConcreteType>
class InStruct : public mlir::OpTrait::TraitBase<ConcreteType, InStruct> {
public:
  static mlir::LogicalResult verifyTrait(mlir::Operation *op) {
    mlir::FailureOr<llvm::StringRef> name = zkir::getParentStructName(op);
    if (mlir::failed(name)) {
      return op->emitOpError() << "can only be used within a '" << getOperationName<StructDefOp>()
                               << "' ancestor";
    }
    return mlir::success();
  }
};

/// This class provides a verifier for ops that are expecting to have
/// an ancestor zkir::FuncOp with the given name.
template <char const *func_name> struct InFunctionWithName {
  template <typename ConcreteType>
  class Impl : public mlir::OpTrait::TraitBase<ConcreteType, Impl> {
  public:
    static mlir::LogicalResult verifyTrait(mlir::Operation *op) {
      mlir::FailureOr<llvm::StringRef> name = zkir::getParentFuncName(op);
      if (mlir::failed(name) || name.value() != func_name) {
        return op->emitOpError() << "only valid within function named \"" << func_name << "\"";
      }
      return mlir::success();
    }
  };
};
} // namespace zkir

// Include TableGen'd declarations
#define GET_OP_CLASSES
#include "Dialect/ZKIR/IR/Ops.h.inc"
