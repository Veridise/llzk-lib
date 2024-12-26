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

//===------------------------------------------------------------------===//
// Helpers
//===------------------------------------------------------------------===//

namespace {
template <bool AllowStruct, bool AllowArray> bool isValidTypeImpl(mlir::Type type);

template <bool AllowStruct> bool isValidArrayElemTypeImpl(mlir::Type type) {
  // ArrayType element can be any valid type sans ArrayType itself.
  //  Pass through the flag indicating if StructType is allowed.
  return isValidTypeImpl<AllowStruct, false>(type);
}

template <bool AllowStruct> bool isValidArrayTypeImpl(mlir::Type type) {
  // Pass through the flag indicating if StructType is allowed.
  return llvm::isa<ArrayType>(type) &&
         isValidArrayElemTypeImpl<AllowStruct>(llvm::cast<ArrayType>(type).getElementType());
}

template <bool AllowStruct, bool AllowArray> bool isValidTypeImpl(mlir::Type type) {
  // This is the main check for allowed types.
  //  Allow StructType and ArrayType only if the respective flags are true.
  //  Pass through the flag indicating if StructType is allowed.
  return type.isSignlessInteger(1) || llvm::isa<mlir::IndexType, FeltType, TypeVarType>(type) ||
         (AllowStruct && llvm::isa<StructType>(type)) ||
         (AllowArray && isValidArrayTypeImpl<AllowStruct>(type));
}
} // namespace

bool isValidType(mlir::Type type) { return isValidTypeImpl<true, true>(type); }

bool isValidEmitEqType(mlir::Type type) { return isValidTypeImpl<false, true>(type); }

bool isValidArrayElemType(mlir::Type type) { return isValidArrayElemTypeImpl<true>(type); }

bool isValidArrayType(mlir::Type type) { return isValidArrayTypeImpl<true>(type); }

namespace {
bool paramAttrUnify(const mlir::Attribute &lhsAttr, const mlir::Attribute &rhsAttr) {
  assertValidAttrForParamOfType(lhsAttr);
  assertValidAttrForParamOfType(rhsAttr);
  // If either attribute is a symbol ref, we assume they unify because a later pass with a
  //  more involved value analysis is required to check if they are actually the same value.
  if (lhsAttr == rhsAttr || lhsAttr.isa<mlir::SymbolRefAttr>() ||
      rhsAttr.isa<mlir::SymbolRefAttr>()) {
    return true;
  }
  // If both are type refs, check for unification of the types.
  if (mlir::TypeAttr lhsTy = lhsAttr.dyn_cast<mlir::TypeAttr>()) {
    if (mlir::TypeAttr rhsTy = rhsAttr.dyn_cast<mlir::TypeAttr>()) {
      return typesUnify(lhsTy.getValue(), rhsTy.getValue());
    }
  }
  return false;
}

bool paramsUnify(
    const mlir::ArrayRef<mlir::Attribute> &lhsParams,
    const mlir::ArrayRef<mlir::Attribute> &rhsParams
) {
  return (lhsParams.size() == rhsParams.size()) &&
         std::equal(lhsParams.begin(), lhsParams.end(), rhsParams.begin(), paramAttrUnify);
}

/// Return `true` iff the two ArrayAttr instances containing StructType or ArrayType parameters
/// are equivalent or could be equivalent after full instantiation of struct parameters.
bool paramsUnify(const mlir::ArrayAttr &lhsParams, const mlir::ArrayAttr &rhsParams) {
  if (lhsParams && rhsParams) {
    return paramsUnify(lhsParams.getValue(), rhsParams.getValue());
  }
  // When one or the other is null, they're only equivalent if both are null
  return !lhsParams && !rhsParams;
}
} // namespace

bool arrayTypesUnify(ArrayType lhs, ArrayType rhs, std::vector<llvm::StringRef> rhsRevPrefix) {
  // Check if the element types of the two arrays can unify
  if (!typesUnify(lhs.getElementType(), rhs.getElementType(), rhsRevPrefix)) {
    return false;
  }
  // Check if the dimension size attributes unify between the LHS and RHS
  return paramsUnify(lhs.getDimensionSizes(), rhs.getDimensionSizes());
}

bool structTypesUnify(StructType lhs, StructType rhs, std::vector<llvm::StringRef> rhsRevPrefix) {
  // Check if it references the same StructDefOp, considering the additional RHS path prefix.
  llvm::SmallVector<mlir::StringRef> rhsNames = getNames(rhs.getNameRef());
  rhsNames.insert(rhsNames.begin(), rhsRevPrefix.rbegin(), rhsRevPrefix.rend());
  if (rhsNames != getNames(lhs.getNameRef())) {
    return false;
  }
  // Check if the parameters unify between the LHS and RHS
  return paramsUnify(lhs.getParams(), rhs.getParams());
}

bool typesUnify(mlir::Type lhs, mlir::Type rhs, std::vector<llvm::StringRef> rhsRevPrefix) {
  if (lhs == rhs) {
    return true;
  }
  if (llvm::isa<TypeVarType>(lhs) || llvm::isa<TypeVarType>(rhs)) {
    // A type variable can be any type, thus it unifies with anything.
    return true;
  }
  if (llvm::isa<StructType>(lhs) && llvm::isa<StructType>(rhs)) {
    return structTypesUnify(llvm::cast<StructType>(lhs), llvm::cast<StructType>(rhs), rhsRevPrefix);
  }
  if (llvm::isa<ArrayType>(lhs) && llvm::isa<ArrayType>(rhs)) {
    return arrayTypesUnify(llvm::cast<ArrayType>(lhs), llvm::cast<ArrayType>(rhs), rhsRevPrefix);
  }
  return false;
}

llvm::function_ref<mlir::InFlightDiagnostic()>
emitErrorUnknownLoc(mlir::MLIRContext *ctx, const char *msg) {
  return [ctx, msg] {
    return mlir::emitError(mlir::Location(mlir::UnknownLoc::get(ctx))).append(msg);
  };
}

namespace {

template <typename... Types> struct TypeList {
  // Returns a comma-separated list formatted string of the names of `Types`
  static std::string getNames() {
    std::string output;
    llvm::raw_string_ostream oss(output);
    oss << "[";
    append<Types...>(oss);
    oss << "]";
    return output;
  }

  // Checks if the provided value is an instance of any of `Types`
  template <typename T> static inline bool matches(const T &value) {
    return llvm::isa<Types...>(value);
  }

  static inline mlir::LogicalResult reportInvalid(
      llvm::function_ref<mlir::InFlightDiagnostic()> emitError, llvm::StringRef foundName,
      const char *aspect
  ) {
    return emitError() << aspect << " must be one of " << getNames() << " but found '" << foundName
                       << "'";
  }

  static inline mlir::LogicalResult reportInvalid(
      llvm::function_ref<mlir::InFlightDiagnostic()> emitError, const mlir::Attribute &found,
      const char *aspect
  ) {
    return reportInvalid(emitError, found.getAbstractAttribute().getName(), aspect);
  }

private:
  template <typename T> static inline void append(llvm::raw_string_ostream &stream) {
    stream << "'" << T::name << "'";
  }

  template <typename First, typename Second, typename... Rest>
  static void append(llvm::raw_string_ostream &stream) {
    append<First>(stream);
    stream << ", ";
    append<Second, Rest...>(stream);
  }
};

} // namespace

//===------------------------------------------------------------------===//
// StructType
//===------------------------------------------------------------------===//

namespace {

// Parameters in the StructType must be one of the following:
//  - Integer constants
//  - SymbolRef (global constants defined in another module require non-flat ref)
//  - Type
using StructParamTypes = TypeList<mlir::IntegerAttr, mlir::SymbolRefAttr, mlir::TypeAttr>;

} // namespace

mlir::LogicalResult StructType::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError, mlir::SymbolRefAttr nameRef,
    mlir::ArrayAttr params
) {
  if (params) {
    for (mlir::Attribute p : params) {
      if (!StructParamTypes::matches(p)) {
        return StructParamTypes::reportInvalid(emitError, p, "Struct parameter");
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

// Dimensions in the ArrayType must be one of the following:
//  - Integer constants
//  - SymbolRef (global constants defined in another module require non-flat ref)
using ArrayDimensionTypes = TypeList<mlir::IntegerAttr, mlir::SymbolRefAttr>;

} // namespace

mlir::LogicalResult computeDimsFromShape(
    mlir::MLIRContext *ctx, llvm::ArrayRef<int64_t> shape,
    llvm::SmallVector<mlir::Attribute> &dimensionSizes
) {
  mlir::Builder builder(ctx);
  auto attrs = llvm::map_range(shape, [&builder](int64_t v) -> mlir::Attribute {
    return builder.getIndexAttr(v);
  });
  dimensionSizes.insert(dimensionSizes.begin(), attrs.begin(), attrs.end());
  return mlir::success();
}

mlir::LogicalResult computeShapeFromDims(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError, mlir::MLIRContext *ctx,
    llvm::ArrayRef<mlir::Attribute> dimensionSizes, llvm::SmallVector<int64_t> &value
) {
  auto emitErrFunc =
      emitError ? emitError : emitErrorUnknownLoc(ctx, "computeShapeFromDims() failed: ");
  for (mlir::Attribute a : dimensionSizes) {
    if (!ArrayDimensionTypes::matches(a)) {
      return ArrayDimensionTypes::reportInvalid(emitErrFunc, a, "Array dimension");
    }
    if (auto p = a.dyn_cast<mlir::IntegerAttr>()) {
      value.push_back(p.getValue().getSExtValue());
    } else if (a.isa<mlir::SymbolRefAttr>()) {
      // The ShapedTypeInterface uses 'kDynamic' for dimensions with non-static size.
      value.push_back(mlir::ShapedType::kDynamic);
    } else {
      llvm::report_fatal_error("parseDerivedShape() is out of sync with ArrayDimensionTypes");
      return mlir::failure();
    }
  }
  return mlir::success();
}

mlir::ParseResult parseAttrVec(mlir::AsmParser &parser, llvm::SmallVector<mlir::Attribute> &value) {
  auto parseResult = mlir::FieldParser<llvm::SmallVector<mlir::Attribute>>::parse(parser);
  if (mlir::failed(parseResult)) {
    return parser.emitError(parser.getCurrentLocation(), "cannot parse array dimensions");
  }
  value.insert(value.begin(), parseResult->begin(), parseResult->end());
  return mlir::success();
}

void printAttrVec(mlir::AsmPrinter &printer, llvm::ArrayRef<mlir::Attribute> value) {
  llvm::raw_ostream &stream = printer.getStream();
  llvm::interleave(value, stream, [&stream](mlir::Attribute a) { a.print(stream, true); }, ",");
}

mlir::ParseResult parseDerivedShape(
    mlir::AsmParser &parser, llvm::SmallVector<int64_t> &value,
    llvm::SmallVector<mlir::Attribute> dimensionSizes
) {
  // This is not actually parsing. It's computing the derived
  //  `shape` from the `dimensionSizes` attributes.
  auto emitError = [&parser] { return parser.emitError(parser.getCurrentLocation()); };
  return computeShapeFromDims(emitError, parser.getContext(), dimensionSizes, value);
}
void printDerivedShape(mlir::AsmPrinter &, llvm::ArrayRef<int64_t>, llvm::ArrayRef<mlir::Attribute>) {
  // nothing to print, it's derived and therefore not represented in the output
}

mlir::LogicalResult ArrayType::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError, mlir::Type elementType,
    llvm::ArrayRef<mlir::Attribute> dimensionSizes, llvm::ArrayRef<int64_t> shape
) {
  // In LLZK, the number of array dimensions must always be known, i.e. `hasRank()==true`
  if (dimensionSizes.empty()) {
    // TODO: can't test this via an ArrayType created from the parser. Need another way.
    return emitError().append("array must have at least one dimension");
  }
  for (mlir::Attribute a : dimensionSizes) {
    if (!ArrayDimensionTypes::matches(a)) {
      return ArrayDimensionTypes::reportInvalid(emitError, a, "Array dimension");
    }
  }
  // Ensure array element type is valid
  if (!isValidArrayElemType(elementType)) {
    // Print proper message if `elementType` is not a valid LLZK type or
    //  if it's simply not the right kind of type for an array element.
    if (mlir::failed(checkValidType(emitError, elementType))) {
      return mlir::failure();
    } else {
      return emitError().append(
          "'", ArrayType::name, "' element type cannot be '",
          elementType.getAbstractType().getName(), "'"
      );
    }
  }
  return mlir::success();
}

ArrayType
ArrayType::cloneWith(std::optional<llvm::ArrayRef<int64_t>> shape, mlir::Type elementType) const {
  // TODO: can't test this via an ArrayType created from the parser. Need another way.
  llvm::ArrayRef<int64_t> newShape = shape.has_value() ? shape.value() : getShape();
  auto emitError = emitErrorUnknownLoc(getContext(), "ArrayType::cloneWith() failed: ");
  return ArrayType::getChecked(emitError, elementType, newShape);
}

int64_t ArrayType::getNumElements() const { return mlir::ShapedType::getNumElements(getShape()); }

} // namespace llzk
