#include "llzk/Dialect/LLZK/IR/Ops.h"
#include "llzk/Dialect/LLZK/IR/Types.h"
#include "llzk/Dialect/LLZK/Util/ErrorHelper.h"
#include "llzk/Dialect/LLZK/Util/SymbolHelper.h"

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/Support/LogicalResult.h>

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>

namespace llzk {

using namespace mlir;

//===------------------------------------------------------------------===//
// Helpers
//===------------------------------------------------------------------===//

namespace {

template <typename... Types> class TypeList {

  /// Helper class that handles appending the 'Types' names to some kind of stream
  template <typename StreamType> struct Appender {

    // single
    template <typename Ty> static inline void append(StreamType &stream) {
      stream << "'" << Ty::name << "'";
    }

    // multiple
    template <typename First, typename Second, typename... Rest>
    static void append(StreamType &stream) {
      append<First>(stream);
      stream << ", ";
      append<Second, Rest...>(stream);
    }

    // full list with wrapping brackets
    static inline void append(StreamType &stream) {
      stream << "[";
      append<Types...>(stream);
      stream << "]";
    }
  };

public:
  // Checks if the provided value is an instance of any of `Types`
  template <typename T> static inline bool matches(const T &value) {
    return llvm::isa<Types...>(value);
  }

  static void reportInvalid(EmitErrorFn emitError, StringRef foundName, const char *aspect) {
    // The implicit conversion from InFlightDiagnostic to LogicalResult in the return causes the
    // diagnostic to be printed.
    InFlightDiagnostic diag = emitError().append(aspect, " must be one of ");
    Appender<InFlightDiagnostic>::append(diag);
    diag.append(" but found '", foundName, "'").report();
  }

  static inline void
  reportInvalid(std::optional<EmitErrorFn> emitError, Attribute found, const char *aspect) {
    if (emitError.has_value()) {
      reportInvalid(*emitError, found.getAbstractAttribute().getName(), aspect);
    }
  }

  // Returns a comma-separated list formatted string of the names of `Types`
  static std::string getNames() {
    std::string output;
    llvm::raw_string_ostream oss(output);
    Appender<llvm::raw_string_ostream>::append(oss);
    return output;
  }
};

/// Helpers to compute the union of multiple TypeList without repetition.
/// Use as: TypeListUnion<TypeList<...>, TypeList<...>, ...>
template <class... Ts> struct make_unique {
  using type = TypeList<Ts...>;
};

template <class... Ts> struct make_unique<TypeList<>, Ts...> : make_unique<Ts...> {};

template <class U, class... Us, class... Ts>
struct make_unique<TypeList<U, Us...>, Ts...>
    : std::conditional_t<
          (std::is_same_v<U, Us> || ...) || (std::is_same_v<U, Ts> || ...),
          make_unique<TypeList<Us...>, Ts...>, make_unique<TypeList<Us...>, Ts..., U>> {};

template <class... Ts> using TypeListUnion = typename make_unique<Ts...>::type;

template <bool AllowStruct, bool AllowString, bool AllowArray, bool AllowTVar>
bool isValidTypeImpl(Type type);

// Dimensions in the ArrayType must be one of the following:
//  - Integer constants
//  - SymbolRef (global constants defined in another module require non-flat ref)
//  - AffineMap (for array created within a loop where size depends on loop variable)
using ArrayDimensionTypes = TypeList<IntegerAttr, SymbolRefAttr, AffineMapAttr>;

bool areValidArrayDimSizes(
    ArrayRef<Attribute> dimensionSizes, std::optional<EmitErrorFn> emitError = std::nullopt
) {
  // In LLZK, the number of array dimensions must always be known, i.e. `hasRank()==true`
  if (dimensionSizes.empty()) {
    if (emitError.has_value()) {
      (*emitError)().append("array must have at least one dimension").report();
    }
    return false;
  }
  // Rather than immediately returning on failure, we check all dimensions and aggregate to provide
  // as many errors are possible in a single verifier run.
  bool success = true;
  for (Attribute a : dimensionSizes) {
    if (!ArrayDimensionTypes::matches(a)) {
      ArrayDimensionTypes::reportInvalid(emitError, a, "Array dimension");
      success = false;
    }
  }
  return success;
}

template <bool AllowStruct, bool AllowString, bool AllowArray, bool AllowTVar>
bool isValidArrayElemTypeImpl(Type type) {
  // ArrayType element can be any valid type sans ArrayType itself.
  //  Pass through the flags indicating which types are allowed.
  return !llvm::isa<ArrayType>(type) &&
         isValidTypeImpl<AllowStruct, AllowString, AllowArray, AllowTVar>(type);
}

template <bool AllowStruct, bool AllowString, bool AllowArray, bool AllowTVar>
bool isValidArrayTypeImpl(
    Type elementType, ArrayRef<Attribute> dimensionSizes,
    std::optional<EmitErrorFn> emitError = std::nullopt
) {
  if (!areValidArrayDimSizes(dimensionSizes, emitError)) {
    return false;
  }

  // Ensure array element type is valid
  if (!isValidArrayElemTypeImpl<AllowStruct, AllowString, AllowArray, AllowTVar>(elementType)) {
    if (emitError.has_value()) {
      // Print proper message if `elementType` is not a valid LLZK type or
      //  if it's simply not the right kind of type for an array element.
      if (succeeded(checkValidType(*emitError, elementType))) {
        (*emitError)()
            .append(
                "'", ArrayType::name, "' element type cannot be '",
                elementType.getAbstractType().getName(), "'"
            )
            .report();
      }
    }
    return false;
  }
  return true;
}

template <bool AllowStruct, bool AllowString, bool AllowArray, bool AllowTVar>
bool isValidArrayTypeImpl(Type type) {
  //  Pass through the flags indicating which types are allowed as array element type.
  if (ArrayType arrTy = llvm::dyn_cast<ArrayType>(type)) {
    return isValidArrayTypeImpl<AllowStruct, AllowString, AllowArray, AllowTVar>(
        arrTy.getElementType(), arrTy.getDimensionSizes()
    );
  }
  return false;
}

// Parameters in the StructType must be one of the following:
//  - Integer constants
//  - SymbolRef (global constants defined in another module require non-flat ref)
//  - Type
//  - AffineMap (for array of non-homogeneous structs)
using StructParamTypes = TypeList<IntegerAttr, SymbolRefAttr, TypeAttr, AffineMapAttr>;

// The Allow* flags here refer to Types nested within a TypeAttr parameter.
template <bool AllowStruct, bool AllowString, bool AllowArray, bool AllowTVar>
bool areValidStructTypeParams(
    ArrayAttr params, std::optional<EmitErrorFn> emitError = std::nullopt
) {
  bool success = true;
  if (params) {
    for (Attribute p : params) {
      if (!StructParamTypes::matches(p)) {
        StructParamTypes::reportInvalid(emitError, p, "Struct parameter");
        success = false;
      } else if (TypeAttr tyAttr = llvm::dyn_cast<TypeAttr>(p)) {
        if (!isValidTypeImpl<AllowStruct, AllowString, AllowArray, AllowTVar>(tyAttr.getValue())) {
          if (emitError.has_value()) {
            (*emitError)()
                .append("expected a valid LLZK type but found ", tyAttr.getValue())
                .report();
          }
          success = false;
        }
      }
    }
  }
  return success;
}

template <bool AllowStruct, bool AllowString, bool AllowArray, bool AllowTVar>
bool isValidStructTypeImpl(Type type) {
  //  Pass through the flags indicating which types are allowed if there is a TypeAttr.
  if (StructType sType = llvm::dyn_cast<StructType>(type)) {
    return areValidStructTypeParams<AllowStruct, AllowString, AllowArray, AllowTVar>(
        sType.getParams()
    );
  }
  return false;
}

template <bool AllowStruct, bool AllowString, bool AllowArray, bool AllowTVar>
bool isValidTypeImpl(Type type) {
  // This is the main check for allowed types.
  //  Pass through the flags indicating which types are allowed.
  return type.isSignlessInteger(1) || llvm::isa<IndexType, FeltType>(type) ||
         (AllowString && llvm::isa<StringType>(type)) ||
         (AllowTVar && llvm::isa<TypeVarType>(type)) ||
         (AllowStruct && isValidStructTypeImpl<true, AllowString, AllowArray, AllowTVar>(type)) ||
         (AllowArray && isValidArrayTypeImpl<AllowStruct, AllowString, true, AllowTVar>(type));
}

} // namespace

bool isValidType(Type type) { return isValidTypeImpl<true, true, true, true>(type); }

bool isValidEmitEqType(Type type) { return isValidTypeImpl<false, false, true, true>(type); }

// Allowed types must align with StructParamTypes (defined below)
bool isValidConstReadType(Type type) { return isValidTypeImpl<false, false, false, true>(type); }

bool isValidArrayElemType(Type type) {
  return isValidArrayElemTypeImpl<true, true, true, true>(type);
}

bool isValidArrayType(Type type) { return isValidArrayTypeImpl<true, true, true, true>(type); }

bool isConcreteType(Type type) { return isValidTypeImpl<true, true, true, false>(type); }

bool isSignalType(Type type) {
  if (auto structParamTy = llvm::dyn_cast<StructType>(type)) {
    // Only check the leaf part of the reference (i.e. just the struct name itself) to allow cases
    // where the `COMPONENT_NAME_SIGNAL` struct may be placed within some nesting of modules, as
    // happens when it's imported via an IncludeOp.
    return structParamTy.getNameRef().getLeafReference() == COMPONENT_NAME_SIGNAL;
  }
  return false;
}

namespace {
bool paramAttrUnify(Attribute lhsAttr, Attribute rhsAttr) {
  assertValidAttrForParamOfType(lhsAttr);
  assertValidAttrForParamOfType(rhsAttr);
  // IntegerAttr and AffineMapAttr only unify via equality and the others may. Additionally,
  //  if either attribute is a symbol ref, we assume they unify because a later pass with a
  //  more involved value analysis is required to check if they are actually the same value.
  if (lhsAttr == rhsAttr || lhsAttr.isa<SymbolRefAttr>() || rhsAttr.isa<SymbolRefAttr>()) {
    return true;
  }
  // If both are type refs, check for unification of the types.
  if (TypeAttr lhsTy = lhsAttr.dyn_cast<TypeAttr>()) {
    if (TypeAttr rhsTy = rhsAttr.dyn_cast<TypeAttr>()) {
      return typesUnify(lhsTy.getValue(), rhsTy.getValue());
    }
  }
  return false;
}
} // namespace

bool typeParamsUnify(const ArrayRef<Attribute> &lhsParams, const ArrayRef<Attribute> &rhsParams) {
  return (lhsParams.size() == rhsParams.size()) &&
         std::equal(lhsParams.begin(), lhsParams.end(), rhsParams.begin(), paramAttrUnify);
}

/// Return `true` iff the two ArrayAttr instances containing StructType or ArrayType parameters
/// are equivalent or could be equivalent after full instantiation of struct parameters.
bool typeParamsUnify(const ArrayAttr &lhsParams, const ArrayAttr &rhsParams) {
  if (lhsParams && rhsParams) {
    return typeParamsUnify(lhsParams.getValue(), rhsParams.getValue());
  }
  // When one or the other is null, they're only equivalent if both are null
  return !lhsParams && !rhsParams;
}

bool arrayTypesUnify(ArrayType lhs, ArrayType rhs, ArrayRef<StringRef> rhsRevPrefix) {
  // Check if the element types of the two arrays can unify
  if (!typesUnify(lhs.getElementType(), rhs.getElementType(), rhsRevPrefix)) {
    return false;
  }
  // Check if the dimension size attributes unify between the LHS and RHS
  return typeParamsUnify(lhs.getDimensionSizes(), rhs.getDimensionSizes());
}

bool structTypesUnify(StructType lhs, StructType rhs, ArrayRef<StringRef> rhsRevPrefix) {
  // Check if it references the same StructDefOp, considering the additional RHS path prefix.
  SmallVector<StringRef> rhsNames = getNames(rhs.getNameRef());
  rhsNames.insert(rhsNames.begin(), rhsRevPrefix.rbegin(), rhsRevPrefix.rend());
  if (rhsNames != getNames(lhs.getNameRef())) {
    return false;
  }
  // Check if the parameters unify between the LHS and RHS
  return typeParamsUnify(lhs.getParams(), rhs.getParams());
}

bool typesUnify(Type lhs, Type rhs, ArrayRef<StringRef> rhsRevPrefix) {
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

//===------------------------------------------------------------------===//
// StructType
//===------------------------------------------------------------------===//

LogicalResult StructType::verify(EmitErrorFn emitError, SymbolRefAttr nameRef, ArrayAttr params) {
  return success(areValidStructTypeParams<true, true, true, true>(params, emitError));
}

FailureOr<SymbolLookupResult<StructDefOp>>
StructType::getDefinition(SymbolTableCollection &symbolTable, Operation *op) {
  // First ensure this StructType passes verification
  ArrayAttr typeParams = this->getParams();
  if (failed(StructType::verify([op] { return op->emitError(); }, getNameRef(), typeParams))) {
    return failure();
  }
  // Perform lookup and ensure the symbol references a StructDefOp
  auto res = lookupTopLevelSymbol<StructDefOp>(symbolTable, getNameRef(), op);
  if (failed(res) || !res.value()) {
    return op->emitError() << "could not find '" << StructDefOp::getOperationName() << "' named \""
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

LogicalResult StructType::verifySymbolRef(SymbolTableCollection &symbolTable, Operation *op) {
  return getDefinition(symbolTable, op);
}

//===------------------------------------------------------------------===//
// ArrayType
//===------------------------------------------------------------------===//

LogicalResult computeDimsFromShape(
    MLIRContext *ctx, ArrayRef<int64_t> shape, SmallVector<Attribute> &dimensionSizes
) {
  assert(dimensionSizes.empty()); // fully computed by this function
  Builder builder(ctx);
  auto attrs = llvm::map_range(shape, [&builder](int64_t v) -> Attribute {
    return builder.getI64IntegerAttr(v);
  });
  dimensionSizes.insert(dimensionSizes.begin(), attrs.begin(), attrs.end());
  assert(dimensionSizes.size() == shape.size()); // fully computed by this function
  return success();
}

LogicalResult computeShapeFromDims(
    EmitErrorFn emitError, MLIRContext *ctx, ArrayRef<Attribute> dimensionSizes,
    SmallVector<int64_t> &shape
) {
  assert(shape.empty()); // fully computed by this function

  // Ensure all Attributes are valid Attribute classes for ArrayType.
  // In the case where `emitError==null`, we mirror how the verification failure is handled by
  // `*Type::get()` via `StorageUserBase` (i.e. use DefaultDiagnosticEmitFn and assert). See:
  //  https://github.com/llvm/llvm-project/blob/0897373f1a329a7a02f8ce3c501a05d2f9c89390/mlir/include/mlir/IR/StorageUniquerSupport.h#L179-L180
  auto errFunc = emitError ? llvm::unique_function<InFlightDiagnostic()>(emitError)
                           : mlir::detail::getDefaultDiagnosticEmitFn(ctx);
  if (!areValidArrayDimSizes(dimensionSizes, errFunc)) {
    assert(emitError);
    return failure();
  }

  // Convert the Attributes to int64_t
  for (Attribute a : dimensionSizes) {
    if (auto p = a.dyn_cast<IntegerAttr>()) {
      shape.push_back(p.getValue().getSExtValue());
    } else if (a.isa<SymbolRefAttr, AffineMapAttr>()) {
      // The ShapedTypeInterface uses 'kDynamic' for dimensions with non-static size.
      shape.push_back(ShapedType::kDynamic);
    } else {
      // For every Attribute class in ArrayDimensionTypes, there should be a case here.
      llvm::report_fatal_error("computeShapeFromDims() is out of sync with ArrayDimensionTypes");
      return failure();
    }
  }
  assert(shape.size() == dimensionSizes.size()); // fully computed by this function
  return success();
}

ParseResult parseAttrVec(AsmParser &parser, SmallVector<Attribute> &value) {
  auto parseResult = FieldParser<SmallVector<Attribute>>::parse(parser);
  if (failed(parseResult)) {
    return parser.emitError(parser.getCurrentLocation(), "cannot parse array dimensions");
  }
  value.insert(value.begin(), parseResult->begin(), parseResult->end());
  return success();
}

void printAttrVec(AsmPrinter &printer, ArrayRef<Attribute> value) {
  llvm::raw_ostream &stream = printer.getStream();
  llvm::interleave(value, stream, [&stream](Attribute a) { a.print(stream, true); }, ",");
}

ParseResult parseDerivedShape(
    AsmParser &parser, SmallVector<int64_t> &shape, SmallVector<Attribute> dimensionSizes
) {
  // This is not actually parsing. It's computing the derived
  //  `shape` from the `dimensionSizes` attributes.
  auto emitError = [&parser] { return parser.emitError(parser.getCurrentLocation()); };
  return computeShapeFromDims(emitError, parser.getContext(), dimensionSizes, shape);
}
void printDerivedShape(AsmPrinter &, ArrayRef<int64_t>, ArrayRef<Attribute>) {
  // nothing to print, it's derived and therefore not represented in the output
}

LogicalResult ArrayType::verify(
    EmitErrorFn emitError, Type elementType, ArrayRef<Attribute> dimensionSizes,
    ArrayRef<int64_t> shape
) {
  return success(
      isValidArrayTypeImpl<true, true, true, true>(elementType, dimensionSizes, emitError)
  );
}

ArrayType ArrayType::cloneWith(std::optional<ArrayRef<int64_t>> shape, Type elementType) const {
  return ArrayType::get(elementType, shape.has_value() ? shape.value() : getShape());
}

int64_t ArrayType::getNumElements() const { return ShapedType::getNumElements(getShape()); }

//===------------------------------------------------------------------===//
// Additional Helpers
//===------------------------------------------------------------------===//

void assertValidAttrForParamOfType(Attribute attr) {
  // Must be the union of valid attribute types within ArrayType, StructType, and TypeVarType.
  using TypeVarAttrs = TypeList<SymbolRefAttr>; // per ODS spec of TypeVarType
  if (!TypeListUnion<ArrayDimensionTypes, StructParamTypes, TypeVarAttrs>::matches(attr)) {
    llvm::report_fatal_error(
        "Legal type parameters are inconsistent. Encountered " +
        attr.getAbstractAttribute().getName()
    );
  }
}

} // namespace llzk
