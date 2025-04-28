//===-- Ops.h ---------------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llzk/Dialect/LLZK/IR/Attrs.h"
#include "llzk/Dialect/LLZK/IR/Dialect.h"
#include "llzk/Dialect/LLZK/IR/Types.h"
#include "llzk/Util/AffineHelpers.h"
#include "llzk/Util/BuilderHelper.h"
#include "llzk/Util/Constants.h"
#include "llzk/Util/SymbolLookup.h"

#include <mlir/Bytecode/BytecodeOpInterface.h>
#include <mlir/Dialect/Affine/IR/AffineValueMap.h>
#include <mlir/IR/AffineMap.h>
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

// Types that must come before the "Ops.h.inc" import
namespace llzk {

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
template <typename TypeClass>
class InStruct : public mlir::OpTrait::TraitBase<TypeClass, InStruct> {
public:
  static mlir::LogicalResult verifyTrait(mlir::Operation *op);
};

/// Return true iff the given Operation is contained within a FuncDefOp with the given name that is
/// itself contained within a StructDefOp.
bool isInStructFunctionNamed(mlir::Operation *op, char const *funcName);

/// Checks if the given Operation is contained within a FuncDefOp with the given name that is itself
/// contained within a StructDefOp, producing an error if not.
template <char const *FuncName, unsigned PrefixLen>
mlir::LogicalResult verifyInStructFunctionNamed(
    mlir::Operation *op, llvm::function_ref<llvm::SmallString<PrefixLen>()> prefix
) {
  return isInStructFunctionNamed(op, FuncName)
             ? mlir::success()
             : op->emitOpError(prefix()) << "only valid within a '" << getOperationName<FuncDefOp>()
                                         << "' named \"@" << FuncName << "\" within a '"
                                         << getOperationName<StructDefOp>() << "' definition";
}

/// This class provides a verifier for ops that are expecting to have
/// an ancestor llzk::FuncDefOp with the given name.
template <char const *FuncName> struct InStructFunctionNamed {
  template <typename TypeClass> class Impl : public mlir::OpTrait::TraitBase<TypeClass, Impl> {
  public:
    static mlir::LogicalResult verifyTrait(mlir::Operation *op) {
      return verifyInStructFunctionNamed<FuncName, 0>(op, [] { return llvm::SmallString<0>(); });
    }
  };
};

/// Produces errors if there is an inconsistency in the various attributes/values that are used to
/// support affine map instantiation in the Op marked with this Trait.
template <int OperandSegmentIndex> struct VerifySizesForMultiAffineOps {
  template <typename TypeClass> class Impl : public mlir::OpTrait::TraitBase<TypeClass, Impl> {
    inline static mlir::LogicalResult verifyHelper(mlir::Operation *op, int32_t segmentSize) {
      TypeClass c = llvm::cast<TypeClass>(op);
      return affineMapHelpers::verifySizesForMultiAffineOps(
          op, segmentSize, c.getMapOpGroupSizesAttr(), c.getMapOperands(), c.getNumDimsPerMapAttr()
      );
    }

  public:
    static mlir::LogicalResult verifyTrait(mlir::Operation *op) {
      if (TypeClass::template hasTrait<mlir::OpTrait::AttrSizedOperandSegments>()) {
        // If the AttrSizedOperandSegments trait is present, must have `OperandSegmentIndex`.
        static_assert(
            OperandSegmentIndex >= 0,
            "When the `AttrSizedOperandSegments` trait is present, the index of `$mapOperands` "
            "within the `operandSegmentSizes` attribute must be specified."
        );
        mlir::DenseI32ArrayAttr segmentSizes = op->getAttrOfType<mlir::DenseI32ArrayAttr>(
            mlir::OpTrait::AttrSizedOperandSegments<TypeClass>::getOperandSegmentSizeAttr()
        );
        assert(
            OperandSegmentIndex < segmentSizes.size() &&
            "Parameter of `VerifySizesForMultiAffineOps` exceeds the number of ODS-declared "
            "operands"
        );
        return verifyHelper(op, segmentSizes[OperandSegmentIndex]);
      } else {
        // If the trait is not present, the `OperandSegmentIndex` is ignored. Pass `-1` to indicate
        // that the checks against `operandSegmentSizes` should be skipped.
        return verifyHelper(op, -1);
      }
    }
  };
};

/// This class provides a verifier for ops that cannot appear within a "constrain" function.
template <typename TypeClass>
class ComputeOnly : public mlir::OpTrait::TraitBase<TypeClass, ComputeOnly> {
public:
  static mlir::LogicalResult verifyTrait(mlir::Operation *op) {
    return !isInStructFunctionNamed(op, FUNC_NAME_CONSTRAIN)
               ? mlir::success()
               : op->emitOpError()
                     << "is ComputeOnly so it cannot be used within a '"
                     << getOperationName<FuncDefOp>() << "' named \"@" << FUNC_NAME_CONSTRAIN
                     << "\" within a '" << getOperationName<StructDefOp>() << "' definition";
  }
};

template <unsigned N>
inline mlir::ParseResult parseDimAndSymbolList(
    mlir::OpAsmParser &parser,
    mlir::SmallVector<mlir::OpAsmParser::UnresolvedOperand, N> &mapOperands,
    mlir::IntegerAttr &numDims
) {
  return affineMapHelpers::parseDimAndSymbolList(parser, mapOperands, numDims);
}

inline void printDimAndSymbolList(
    mlir::OpAsmPrinter &printer, mlir::Operation *op, mlir::OperandRange mapOperands,
    mlir::IntegerAttr numDims
) {
  return affineMapHelpers::printDimAndSymbolList(printer, op, mapOperands, numDims);
}

inline mlir::ParseResult parseMultiDimAndSymbolList(
    mlir::OpAsmParser &parser,
    mlir::SmallVector<mlir::SmallVector<mlir::OpAsmParser::UnresolvedOperand>> &multiMapOperands,
    mlir::DenseI32ArrayAttr &numDimsPerMap
) {
  return affineMapHelpers::parseMultiDimAndSymbolList(parser, multiMapOperands, numDimsPerMap);
}

inline void printMultiDimAndSymbolList(
    mlir::OpAsmPrinter &printer, mlir::Operation *op, mlir::OperandRangeRange multiMapOperands,
    mlir::DenseI32ArrayAttr numDimsPerMap
) {
  return affineMapHelpers::printMultiDimAndSymbolList(printer, op, multiMapOperands, numDimsPerMap);
}

inline mlir::ParseResult parseAttrDictWithWarnings(
    mlir::OpAsmParser &parser, mlir::NamedAttrList &extraAttrs, mlir::OperationState &state
) {
  return affineMapHelpers::parseAttrDictWithWarnings(parser, extraAttrs, state);
}

template <typename ConcreteOp>
inline void printAttrDictWithWarnings(
    mlir::OpAsmPrinter &printer, ConcreteOp op, mlir::DictionaryAttr extraAttrs,
    typename mlir::PropertiesSelector<ConcreteOp>::type state
) {
  return affineMapHelpers::printAttrDictWithWarnings(printer, op, extraAttrs, state);
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
