#pragma once

#include "llzk/Dialect/LLZK/IR/Attrs.h"
#include "llzk/Dialect/LLZK/IR/Dialect.h"
#include "llzk/Dialect/LLZK/IR/Types.h"
#include "llzk/Dialect/LLZK/Util/SymbolLookup.h" // IWYU pragma: keep

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
namespace llzk {

/// Symbol name for the struct/component representing a signal. A "signal" has direct correspondence
/// to a circom signal or AIR/PLONK column, opposed to intermediate values or other expressions.
constexpr char COMPONENT_NAME_SIGNAL[] = "Signal";

/// Symbol name for the main entry point struct/component (if any). There are additional
/// restrictions on the struct with this name:
/// 1. It cannot have struct parameters.
/// 2. The parameter types of its functions (besides the required "self" parameter) can
///     only be `struct<Signal>` or `array<.. x struct<Signal>>`.
constexpr char COMPONENT_NAME_MAIN[] = "Main";

constexpr char FUNC_NAME_COMPUTE[] = "compute";
constexpr char FUNC_NAME_CONSTRAIN[] = "constrain";

/// Get the operation name, like "llzk.emit_op" for the given OpType.
/// This function can be used when the compiler would complain about
/// incomplete types if `OpType::getOperationName()` were called directly.
template <typename OpType> inline llvm::StringLiteral getOperationName() {
  return OpType::getOperationName();
}

/// Return the closest surrounding parent operation that is of type 'OpType'.
template <typename OpType> mlir::FailureOr<OpType> getParentOfType(mlir::Operation *op) {
  if (OpType p = op->getParentOfType<OpType>()) {
    return p;
  } else {
    return mlir::failure();
  }
}

/// Return true iff the given Operation is nested somewhere within a StructDefOp.
bool isInStruct(mlir::Operation *op);

/// If the given Operation is nested somewhere within a StructDefOp, return a success result
/// containing that StructDefOp. Otherwise emit an error and return a failure result.
mlir::FailureOr<StructDefOp> verifyInStruct(mlir::Operation *op);

/// This class provides a verifier for ops that are expected to have
/// an ancestor llzk::StructDefOp.
template <typename ConcreteType>
class InStruct : public mlir::OpTrait::TraitBase<ConcreteType, InStruct> {
public:
  static mlir::LogicalResult verifyTrait(mlir::Operation *op);
};

/// Return true iff the given Operation is contained within a FuncOp with the given name that is
/// itself contained within a StructDefOp.
bool isInStructFunctionNamed(mlir::Operation *op, char const *funcName);

/// Checks if the given Operation is contained within a FuncOp with the given name that is itself
/// contained within a StructDefOp, producing an error if not.
template <char const *FuncName, unsigned PrefixLen>
mlir::LogicalResult verifyInStructFunctionNamed(
    mlir::Operation *op, llvm::function_ref<llvm::SmallString<PrefixLen>()> prefix
) {
  return isInStructFunctionNamed(op, FuncName)
             ? mlir::success()
             : op->emitOpError(prefix()) << "only valid within a '" << getOperationName<FuncOp>()
                                         << "' named \"@" << FuncName << "\" within a '"
                                         << getOperationName<StructDefOp>() << "' definition";
}

/// This class provides a verifier for ops that are expecting to have
/// an ancestor llzk::FuncOp with the given name.
template <char const *FuncName> struct InStructFunctionNamed {
  template <typename ConcreteType>
  class Impl : public mlir::OpTrait::TraitBase<ConcreteType, Impl> {
  public:
    static mlir::LogicalResult verifyTrait(mlir::Operation *op) {
      return verifyInStructFunctionNamed<FuncName, 0>(op, [] { return llvm::SmallString<0>(); });
    }
  };
};

/// This class provides a verifier for ops that cannot appear within a "constrain" function.
template <typename ConcreteType>
class ComputeOnly : public mlir::OpTrait::TraitBase<ConcreteType, ComputeOnly> {
public:
  static mlir::LogicalResult verifyTrait(mlir::Operation *op) {
    return !isInStructFunctionNamed(op, FUNC_NAME_CONSTRAIN)
               ? mlir::success()
               : op->emitOpError()
                     << "is ComputeOnly so it cannot be used within a '"
                     << getOperationName<FuncOp>() << "' named \"@" << FUNC_NAME_CONSTRAIN
                     << "\" within a '" << getOperationName<StructDefOp>() << "' definition";
  }
};

template <typename OpType, typename... Args>
inline OpType delegate_to_build(mlir::Location location, Args &&...args) {
  mlir::OpBuilder builder(location->getContext());
  return builder.create<OpType>(location, std::forward<Args>(args)...);
}
} // namespace llzk

// Include TableGen'd declarations
#define GET_OP_CLASSES
#include "llzk/Dialect/LLZK/IR/OpInterfaces.h.inc"

// Include TableGen'd declarations
#define GET_OP_CLASSES
#include "llzk/Dialect/LLZK/IR/Ops.h.inc"

namespace llzk {

mlir::InFlightDiagnostic
genCompareErr(StructDefOp &expected, mlir::Operation *origin, const char *aspect);

mlir::LogicalResult checkSelfType(
    mlir::SymbolTableCollection &symbolTable, StructDefOp &expectedStruct, mlir::Type actualType,
    mlir::Operation *origin, const char *aspect
);

} // namespace llzk
