//===-- Ops.h ---------------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llzk/Util/AffineHelper.h"
#include "llzk/Util/Constants.h"

#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/Operation.h>
#include <mlir/Support/LogicalResult.h>

#include <llvm/ADT/SmallString.h>
#include <llvm/ADT/StringRef.h>

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
