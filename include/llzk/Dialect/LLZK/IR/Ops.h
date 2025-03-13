#pragma once

#include "llzk/Dialect/LLZK/IR/Attrs.h"
#include "llzk/Dialect/LLZK/IR/Dialect.h"
#include "llzk/Dialect/LLZK/IR/Types.h"
#include "llzk/Dialect/LLZK/Util/SymbolLookup.h" // IWYU pragma: keep

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

#include <numeric>
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

/// Group together all implementation related to AffineMap type parameters.
namespace affineMapHelpers {

/// Parses dimension and symbol list for an AffineMap instantiation.
template <unsigned N>
mlir::ParseResult parseDimAndSymbolList(
    mlir::OpAsmParser &parser,
    mlir::SmallVector<mlir::OpAsmParser::UnresolvedOperand, N> &mapOperands,
    mlir::IntegerAttr &numDims
);

/// Prints dimension and symbol list for an AffineMap instantiation.
void printDimAndSymbolList(
    mlir::OpAsmPrinter &printer, mlir::Operation *op, mlir::OperandRange mapOperands,
    mlir::IntegerAttr numDims
);

/// Parses comma-separated list of multiple AffineMap instantiations.
mlir::ParseResult parseMultiDimAndSymbolList(
    mlir::OpAsmParser &parser,
    mlir::SmallVector<mlir::SmallVector<mlir::OpAsmParser::UnresolvedOperand>> &multiMapOperands,
    mlir::DenseI32ArrayAttr &numDimsPerMap
);

/// Prints comma-separated list of multiple AffineMap instantiations.
void printMultiDimAndSymbolList(
    mlir::OpAsmPrinter &printer, mlir::Operation *op, mlir::OperandRangeRange multiMapOperands,
    mlir::DenseI32ArrayAttr numDimsPerMap
);

/// This custom parse/print AttrDictWithWarnings is necessary to directly check what 'attr-dict' is
/// parsed from the input. Waiting until the `verify()` function will not work because the generated
/// `parse()` function automatically computes and initializes the attributes.
mlir::ParseResult parseAttrDictWithWarnings(
    mlir::OpAsmParser &parser, mlir::NamedAttrList &extraAttrs, mlir::OperationState &state
);

template <typename ConcreteOp>
void printAttrDictWithWarnings(
    mlir::OpAsmPrinter &printer, ConcreteOp op, mlir::DictionaryAttr extraAttrs,
    typename ConcreteOp::Properties state
);

/// Implements the ODS trait with the same name. Produces errors if there is an inconsistency in the
/// various attributes/values that are used to support affine map instantiation in the op.
mlir::LogicalResult verifySizesForMultiAffineOps(
    mlir::Operation *op, int32_t segmentSize, mlir::ArrayRef<int32_t> mapOpGroupSizes,
    mlir::OperandRangeRange mapOperands, mlir::ArrayRef<int32_t> numDimsPerMap
);

/// Produces errors if there is an inconsistency between the attributes/values that are used to
/// support affine map instantiation in the op and the AffineMapAttr list collected from the type.
mlir::LogicalResult verifyAffineMapInstantiations(
    mlir::OperandRangeRange mapOps, mlir::ArrayRef<int32_t> numDimsPerMap,
    mlir::ArrayRef<mlir::AffineMapAttr> mapAttrs, mlir::Operation *origin
);

/// Utility for build() functions that initializes the `operandSegmentSizes`, `mapOpGroupSizes`, and
/// `numDimsPerMap` attributes for an Op that performs affine map instantiations.
///
/// Note: This function supports Ops with 2 ODS-defined operand segments with the second being the
/// size of the `mapOperands` segment and the first provided by the `firstSegmentSize` parameter.
template <typename OpType>
inline typename OpType::Properties &buildInstantiationAttrs(
    mlir::OpBuilder &odsBuilder, mlir::OperationState &odsState,
    mlir::ArrayRef<mlir::ValueRange> mapOperands, mlir::DenseI32ArrayAttr numDimsPerMap,
    int32_t firstSegmentSize = 0
) {
  int32_t mapOpsSegmentSize = 0;
  mlir::SmallVector<int32_t> rangeSegments;
  for (mlir::ValueRange r : mapOperands) {
    odsState.addOperands(r);
    int32_t s = static_cast<int32_t>(r.size());
    rangeSegments.push_back(s);
    mapOpsSegmentSize += s;
  }
  typename OpType::Properties &props = odsState.getOrAddProperties<typename OpType::Properties>();
  props.setMapOpGroupSizes(odsBuilder.getDenseI32ArrayAttr(rangeSegments));
  props.setOperandSegmentSizes({firstSegmentSize, mapOpsSegmentSize});
  if (numDimsPerMap) {
    props.setNumDimsPerMap(numDimsPerMap);
  }
  return props;
}

/// Utility for build() functions that initializes the `operandSegmentSizes`, `mapOpGroupSizes`, and
/// `numDimsPerMap` attributes for an Op that supports affine map instantiations but in the case
/// where there are none.
template <typename OpType>
inline typename OpType::Properties &buildInstantiationAttrsEmpty(
    mlir::OpBuilder &odsBuilder, mlir::OperationState &odsState, int32_t firstSegmentSize = 0
) {
  typename OpType::Properties &props = odsState.getOrAddProperties<typename OpType::Properties>();
  // `operandSegmentSizes` = [ firstSegmentSize, mapOperands.size ]
  props.setOperandSegmentSizes({firstSegmentSize, 0});
  // There are no affine map operands so initialize the related properties as empty arrays.
  props.setMapOpGroupSizes(odsBuilder.getDenseI32ArrayAttr({}));
  props.setNumDimsPerMap(odsBuilder.getDenseI32ArrayAttr({}));
  return props;
}

} // namespace affineMapHelpers

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
                     << getOperationName<FuncOp>() << "' named \"@" << FUNC_NAME_CONSTRAIN
                     << "\" within a '" << getOperationName<StructDefOp>() << "' definition";
  }
};

template <typename OpType, typename... Args>
inline OpType delegate_to_build(mlir::Location location, Args &&...args) {
  mlir::OpBuilder builder(location->getContext());
  return builder.create<OpType>(location, std::forward<Args>(args)...);
}

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
