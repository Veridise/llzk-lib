#include "llzk/Dialect/LLZK/IR/Ops.h"
#include "llzk/Dialect/LLZK/IR/Types.h"
#include "llzk/Dialect/LLZK/Util/SymbolHelper.h"

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/Support/LogicalResult.h>

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
  return type.isSignlessInteger(1) || llvm::isa<::mlir::IndexType>(type) ||
         llvm::isa<llzk::FeltType>(type) || llvm::isa<llzk::StructType>(type) ||
         isValidArrayType(type);
}

// valid types: I1, Index, LLZK_FeltType, LLZK_ArrayType
bool isValidEmitEqType(mlir::Type type) {
  return type.isSignlessInteger(1) || llvm::isa<::mlir::IndexType>(type) ||
         llvm::isa<llzk::FeltType>(type) ||
         (llvm::isa<llzk::ArrayType>(type) &&
          isValidEmitEqType(llvm::cast<::llzk::ArrayType>(type).getElementType()));
}

namespace {
bool structParamAttrUnify(const mlir::Attribute &lhsAttr, const mlir::Attribute &rhsAttr) {
  return lhsAttr == rhsAttr || lhsAttr.isa<mlir::FlatSymbolRefAttr>() ||
         rhsAttr.isa<mlir::FlatSymbolRefAttr>();
}
} // namespace

bool structParamsUnify(const mlir::ArrayAttr &lhsParams, const mlir::ArrayAttr &rhsParams) {
  if (lhsParams && rhsParams) {
    return (lhsParams.size() == rhsParams.size()) &&
           std::equal(lhsParams.begin(), lhsParams.end(), rhsParams.begin(), structParamAttrUnify);
  }
  // When one or the other is null, they're only equivalent if both are null
  return !lhsParams && !rhsParams;
}

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

mlir::LogicalResult StructType::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError, mlir::SymbolRefAttr nameRef,
    mlir::ArrayAttr params
) {
  if (params) {
    // Ensure the parameters in the StructType are only Integer constants or FlatSymbolRef.
    for (auto i = params.begin(); i != params.end(); ++i) {
      if (!i->isa<mlir::IntegerAttr>() && !i->isa<mlir::FlatSymbolRefAttr>()) {
        return emitError() << "Unexpected struct parameter type: "
                           << i->getAbstractAttribute().getName();
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

mlir::LogicalResult ArrayType::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError, mlir::Type elementType,
    llvm::ArrayRef<int64_t> shape
) {

  // If a user of LLZK needs the shape to be statically defined
  // it should check it here. How to communicate that need to the type is TBD.
  if (shape.size() <= 0) {
    return emitError() << "array must have a shape of at least one element";
  }
  // An array can hold any LLZK type bar Arrays
  auto typeCheckResult = checkValidType(emitError, elementType);
  if (mlir::succeeded(typeCheckResult)) {
    if (llvm::isa<llzk::ArrayType>(elementType)) {
      return emitError() << "array inner type cannot be array";
    }
  }
  return typeCheckResult;
}

bool ArrayType::hasRank() const {
  return true; // A LLZK Array is ranked by construction.
}

ArrayType
ArrayType::cloneWith(std::optional<llvm::ArrayRef<int64_t>> shape, mlir::Type elementType) const {
  llvm::ArrayRef<int64_t> newShape = getShape();
  if (shape.has_value()) {
    newShape = *shape;
  }
  return ArrayType::get(elementType.getContext(), elementType, newShape);
}

int64_t ArrayType::getNumElements() const { return mlir::ShapedType::getNumElements(getShape()); }

// The code for these two methods was based on
// the autogenerated code by TableGen

/// A LLZK Array has a similar format to tensors
/// and memref types: <$shape x $type>
/// i.e. !llzk.array<2x2x!llzk.felt>
///   This will produce a shape of [2,2]
///   and a type of LLZK's Felt
mlir::Type ArrayType::parse(mlir::AsmParser &parser) {
  ::mlir::Builder odsBuilder(parser.getContext());
  ::llvm::SMLoc loc = parser.getCurrentLocation();
  // Parse literal '<'
  if (parser.parseLess()) {
    return {};
  }

  // I worry this array may dissapear early can cause
  // an Use-After-Free but the MLIR code I studied did
  // it too so it may be fine.  -- Dani
  llvm::SmallVector<int64_t> parsedShape;

  // The default configuration is good for our purpose
  //   allowDynamic = true
  //    This allows ? values.
  //    Wether an unknown dimension size is allowed or
  //    not will depend on the semantics the array finds
  //    itself in.
  //   withTrailing = true
  //    The parser will consume a literal `x` token
  //    if its trailing after the rest of the shape has been
  //    parsed. This leaves the head right at the type declaration.
  auto _result_shape = parser.parseDimensionList(parsedShape);

  // Parse variable 'elementType'
  auto _result_elementType = ::mlir::FieldParser<::mlir::Type>::parse(parser);
  if (::mlir::failed(_result_elementType)) {
    parser.emitError(
        parser.getCurrentLocation(),
        "failed to parse LLZK_ArrayType parameter 'elementType' which is to be a `::mlir::Type`"
    );
    return {};
  }
  // Parse literal '>'
  if (parser.parseGreater()) {
    return {};
  }

  assert(::mlir::succeeded(_result_elementType));
  assert(::mlir::succeeded(_result_shape));

  return parser.getChecked<ArrayType>(
      loc, parser.getContext(), ::mlir::Type(*_result_elementType),
      ::llvm::ArrayRef<int64_t>(parsedShape)
  );
}

/// Prints the array type with the following format
/// <$shape x $type>
/// i.e. !llzk.array<2x2 x !llzk.felt>
void ArrayType::print(mlir::AsmPrinter &printer) const {
  mlir::Builder odsBuilder(getContext());
  printer << "<";
  printer.printDimensionList(getShape());
  printer << " x ";
  printer.printStrippedAttrOrType(getElementType());
  printer << ">";
}

} // namespace llzk
