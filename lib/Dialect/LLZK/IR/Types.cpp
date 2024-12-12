#include "llzk/Dialect/LLZK/IR/Ops.h"
#include "llzk/Dialect/LLZK/IR/Types.h"
#include "llzk/Dialect/LLZK/Util/SymbolHelper.h"

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/Support/LogicalResult.h>

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>

namespace llzk {

/// Checks if the type is a LLZK Array and it also contains
/// a valid LLZK type.
bool isValidArrayType(mlir::Type type) {
  return llvm::isa<llzk::ArrayType>(type) &&
         isValidType(llvm::cast<::llzk::ArrayType>(type).getElementType());
}

// valid types: I1, Index, LLZK_FeltType, LLZK_StructType, LLZK_ArrayType
bool isValidType(mlir::Type type) {
  return type.isSignlessInteger(1) || llvm::isa<mlir::IndexType>(type) ||
         llvm::isa<llzk::FeltType>(type) || llvm::isa<llzk::StructType>(type) ||
         isValidArrayType(type);
}

// valid types: I1, Index, LLZK_FeltType, LLZK_ArrayType
bool isValidEmitEqType(mlir::Type type) {
  return type.isSignlessInteger(1) || llvm::isa<mlir::IndexType>(type) ||
         llvm::isa<llzk::FeltType>(type) ||
         (llvm::isa<llzk::ArrayType>(type) &&
          isValidEmitEqType(llvm::cast<::llzk::ArrayType>(type).getElementType()));
}

namespace {
bool structParamAttrUnify(const mlir::Attribute &lhsAttr, const mlir::Attribute &rhsAttr) {
  // TODO: when TypeAttr is allowed as a parameter, this must use typesUnify() to compare TypeAttr.
  //
  // If either attribute is a symbol ref, we assume they unify because a later pass with a
  //  more involved value analysis is required to check if they are actually the same value.
  return lhsAttr == rhsAttr || lhsAttr.isa<mlir::SymbolRefAttr>() ||
         rhsAttr.isa<mlir::SymbolRefAttr>();
}

/// Return `true` iff the two ArrayAttr instances containing struct parameters are equivalent or
/// could be equivalent after full instantiation of struct parameters.
bool structParamsUnify(const mlir::ArrayAttr &lhsParams, const mlir::ArrayAttr &rhsParams) {
  if (lhsParams && rhsParams) {
    return (lhsParams.size() == rhsParams.size()) &&
           std::equal(lhsParams.begin(), lhsParams.end(), rhsParams.begin(), structParamAttrUnify);
  }
  // When one or the other is null, they're only equivalent if both are null
  return !lhsParams && !rhsParams;
}
} // namespace

bool structTypesUnify(
    const StructType &lhs, const StructType &rhs, std::vector<llvm::StringRef> rhsRevPrefix
) {
  // Check if it references the same StructDefOp, considering the additional RHS path prefix.
  llvm::SmallVector<mlir::StringRef> rhsNames = getNames(rhs.getNameRef());
  rhsNames.insert(rhsNames.begin(), rhsRevPrefix.rbegin(), rhsRevPrefix.rend());
  if (rhsNames != getNames(lhs.getNameRef())) {
    return false;
  }
  // Check if the parameters unify between the LHS and RHS
  return structParamsUnify(lhs.getParams(), rhs.getParams());
}

bool typesUnify(
    const mlir::Type &lhs, const mlir::Type &rhs, std::vector<llvm::StringRef> rhsRevPrefix
) {
  if (lhs == rhs) {
    return true;
  }
  if (llvm::isa<llzk::StructType>(lhs) && llvm::isa<llzk::StructType>(rhs)) {
    return structTypesUnify(llvm::cast<StructType>(lhs), llvm::cast<StructType>(rhs), rhsRevPrefix);
  }
  return false;
}

//===------------------------------------------------------------------===//
// StructType
//===------------------------------------------------------------------===//

mlir::LogicalResult StructType::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError, mlir::SymbolRefAttr nameRef,
    mlir::ArrayAttr params
) {
  if (params) {
    // Ensure the parameters in the StructType are only
    //  - Integer constants
    //  - SymbolRef (global constants defined in another module require non-flat ref)
    // TODO: must include TypeAttr here to support type parameters on structs
    for (mlir::Attribute p : params) {
      if (!p.isa<mlir::IntegerAttr>() && !p.isa<mlir::SymbolRefAttr>()) {
        return emitError() << "Unexpected struct parameter type: "
                           << p.getAbstractAttribute().getName();
      }
    }
  }
  return mlir::success();
}

mlir::FailureOr<SymbolLookupResult<StructDefOp>>
StructType::getDefinition(mlir::SymbolTableCollection &symbolTable, mlir::Operation *op) {
  // First ensure this StructType passes verification
  mlir::ArrayAttr typeParams = this->getParams();
  if (mlir::failed(StructType::verify([op] {
    return op->emitError();
  }, this->getNameRef(), typeParams))) {
    return mlir::failure();
  }
  // Perform lookup and ensure the symbol references a StructDefOp
  auto res = lookupTopLevelSymbol<StructDefOp>(symbolTable, getNameRef(), op);
  if (mlir::failed(res) || !res.value()) {
    return op->emitError() << "no '" << StructDefOp::getOperationName() << "' named \""
                           << getNameRef() << "\"";
  }
  // If this StructType contains parameters, make sure they match the number from the StructDefOp.
  if (typeParams) {
    auto defParams = res.value().get().getConstParams();
    size_t numExpected = defParams ? defParams->size() : 0;
    if (typeParams.size() != numExpected) {
      return op->emitError() << "'" << StructType::name << "' type has " << typeParams.size()
                             << " parameters but \"" << res.value().get().getSymName()
                             << "\" expects " << numExpected;
    }
  }
  return res;
}

mlir::LogicalResult
StructType::verifySymbolRef(mlir::SymbolTableCollection &symbolTable, mlir::Operation *op) {
  return getDefinition(symbolTable, op);
}

//===------------------------------------------------------------------===//
// ArrayType
//===------------------------------------------------------------------===//

namespace {

inline mlir::InFlightDiagnostic &&
invalidArrDim(llvm::function_ref<mlir::InFlightDiagnostic()> emitError, mlir::Attribute &a) {
  // TODO: this needs a test case
  return emitError() << "Unexpected array dimension type: " << a.getAbstractAttribute().getName();
}

} // namespace

mlir::LogicalResult
parseAttrVec(mlir::AsmParser &parser, llvm::SmallVector<mlir::Attribute> &value) {
  auto parseResult = mlir::FieldParser<llvm::SmallVector<mlir::Attribute>>::parse(parser);
  if (mlir::failed(parseResult)) {
    return parser.emitError(parser.getCurrentLocation(), "failed to parse attribute list");
  }
  value.insert(value.begin(), parseResult->begin(), parseResult->end());
  return mlir::success();
}

void printAttrVec(mlir::AsmPrinter &printer, llvm::ArrayRef<mlir::Attribute> value) {
  llvm::raw_ostream &stream = printer.getStream();
  llvm::interleave(value, stream, [&stream](mlir::Attribute a) { a.print(stream, true); }, ",");
}

mlir::LogicalResult parseDerivedShape(
    mlir::AsmParser &parser, llvm::SmallVector<int64_t> &value,
    llvm::SmallVector<mlir::Attribute> dimensionSizes
) {
  // This is not actually parsing. It's computing the derived
  //  `shape` from the `dimensionSizes` attributes.
  for (mlir::Attribute a : dimensionSizes) {
    if (auto p = a.dyn_cast<mlir::IntegerAttr>()) {
      value.push_back(p.getValue().getSExtValue());
    } else if (a.isa<mlir::SymbolRefAttr>()) {
      // The ShapedTypeInterface uses 'kDynamic' for dimensions with non-static size.
      value.push_back(mlir::ShapedType::kDynamic);
    } else {
      return invalidArrDim([&parser] { return parser.emitError(parser.getCurrentLocation()); }, a);
    }
  }
  return mlir::success();
}
void printDerivedShape(mlir::AsmPrinter &, llvm::ArrayRef<int64_t>, llvm::ArrayRef<mlir::Attribute>) {
  // nothing to print, it's derived and therefore not represented in the output
}

mlir::LogicalResult ArrayType::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError, mlir::Type elementType,
    llvm::ArrayRef<mlir::Attribute> dimensions, llvm::ArrayRef<int64_t> shape
) {
  // In LLZK, the number of array dimensions must always be known, i.e. `hasRank()==true`
  if (dimensions.empty()) {
    return emitError() << "array must have at least one dimension";
  }
  // Ensure the parameters in the ArrayType are only
  //  - Integer constants
  //  - SymbolRef (global constants defined in another module require non-flat ref)
  for (mlir::Attribute a : dimensions) {
    if (!a.isa<mlir::IntegerAttr>() && !a.isa<mlir::SymbolRefAttr>()) {
      // TODO: ensure symbol lookup succeeds
      return invalidArrDim(emitError, a);
    }
  }

  // An array can hold any LLZK type bar Arrays
  if (llvm::isa<llzk::ArrayType>(elementType)) {
    // TODO: this needs a test case
    return emitError() << "array element type cannot be array";
  }
  return checkValidType(emitError, elementType);
}

ArrayType
ArrayType::cloneWith(std::optional<llvm::ArrayRef<int64_t>> shape, mlir::Type elementType) const {
  llvm::ArrayRef<int64_t> newShape = shape.has_value() ? shape.value() : getShape();
  mlir::Builder builder(getContext());
  mlir::ArrayAttr newDimensions = builder.getIndexArrayAttr(newShape);
  auto emitError = [] { return mlir::emitError(mlir::Location(mlir::LocationAttr())); };
  return ArrayType::getChecked(emitError, getContext(), elementType, newDimensions, newShape);
}

int64_t ArrayType::getNumElements() const { return mlir::ShapedType::getNumElements(getShape()); }

} // namespace llzk
