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

using namespace mlir;

//===------------------------------------------------------------------===//
// Helpers
//===------------------------------------------------------------------===//

namespace {
template <bool AllowStruct, bool AllowString, bool AllowArray> bool isValidTypeImpl(Type type);

template <bool AllowStruct, bool AllowString> bool isValidArrayElemTypeImpl(Type type) {
  // ArrayType element can be any valid type sans ArrayType itself.
  //  Pass through the flags indicating which types are allowed.
  // [TH]: Maybe the array type should not be excluded beyond the immediate element type itself.
  //  i.e. should `!llzk.array<2 x !llzk.struct<@A<[!llzk.array<2 x i1>]>>>` be allowed?
  return isValidTypeImpl<AllowStruct, AllowString, false>(type);
}

template <bool AllowStruct, bool AllowString> bool isValidArrayTypeImpl(Type type) {
  //  Pass through the flags indicating which types are allowed.
  return llvm::isa<ArrayType>(type) && isValidArrayElemTypeImpl<AllowStruct, AllowString>(
                                           llvm::cast<ArrayType>(type).getElementType()
                                       );
}

template <bool AllowStruct, bool AllowString, bool AllowArray> bool isValidTypeImpl(Type type) {
  // This is the main check for allowed types.
  //  Allow StructType and ArrayType only if the respective flags are true.
  //  Pass through the flags indicating which types are allowed.
  return type.isSignlessInteger(1) || llvm::isa<IndexType, FeltType, TypeVarType>(type) ||
         (AllowStruct && llvm::isa<StructType>(type)) ||
         (AllowString && llvm::isa<StringType>(type)) ||
         (AllowArray && isValidArrayTypeImpl<AllowStruct, AllowString>(type));
}
} // namespace

bool isValidType(Type type) { return isValidTypeImpl<true, true, true>(type); }

bool isValidEmitEqType(Type type) { return isValidTypeImpl<false, false, true>(type); }

// Allowed types must align with StructParamTypes (defined below)
bool isValidConstReadType(Type type) { return isValidTypeImpl<false, false, false>(type); }

bool isValidArrayElemType(Type type) { return isValidArrayElemTypeImpl<true, true>(type); }

bool isValidArrayType(Type type) { return isValidArrayTypeImpl<true, true>(type); }

bool isSignalType(Type type) {
  if (auto structParamTy = mlir::dyn_cast<StructType>(type)) {
    // Only check the leaf part of the reference (i.e. just the struct name itself) to allow cases
    // where the `COMPONENT_NAME_SIGNAL` struct may be placed within some nesting of modules, as
    // happens when it's imported via an IncludeOp.
    return structParamTy.getNameRef().getLeafReference() == COMPONENT_NAME_SIGNAL;
  }
  return false;
}

namespace {
bool paramAttrUnify(const Attribute &lhsAttr, const Attribute &rhsAttr) {
  assertValidAttrForParamOfType(lhsAttr);
  assertValidAttrForParamOfType(rhsAttr);
  // If either attribute is a symbol ref, we assume they unify because a later pass with a
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

bool paramsUnify(const ArrayRef<Attribute> &lhsParams, const ArrayRef<Attribute> &rhsParams) {
  return (lhsParams.size() == rhsParams.size()) &&
         std::equal(lhsParams.begin(), lhsParams.end(), rhsParams.begin(), paramAttrUnify);
}

/// Return `true` iff the two ArrayAttr instances containing StructType or ArrayType parameters
/// are equivalent or could be equivalent after full instantiation of struct parameters.
bool paramsUnify(const ArrayAttr &lhsParams, const ArrayAttr &rhsParams) {
  if (lhsParams && rhsParams) {
    return paramsUnify(lhsParams.getValue(), rhsParams.getValue());
  }
  // When one or the other is null, they're only equivalent if both are null
  return !lhsParams && !rhsParams;
}
} // namespace

bool arrayTypesUnify(ArrayType lhs, ArrayType rhs, ArrayRef<llvm::StringRef> rhsRevPrefix) {
  // Check if the element types of the two arrays can unify
  if (!typesUnify(lhs.getElementType(), rhs.getElementType(), rhsRevPrefix)) {
    return false;
  }
  // Check if the dimension size attributes unify between the LHS and RHS
  return paramsUnify(lhs.getDimensionSizes(), rhs.getDimensionSizes());
}

bool structTypesUnify(StructType lhs, StructType rhs, ArrayRef<llvm::StringRef> rhsRevPrefix) {
  // Check if it references the same StructDefOp, considering the additional RHS path prefix.
  llvm::SmallVector<StringRef> rhsNames = getNames(rhs.getNameRef());
  rhsNames.insert(rhsNames.begin(), rhsRevPrefix.rbegin(), rhsRevPrefix.rend());
  if (rhsNames != getNames(lhs.getNameRef())) {
    return false;
  }
  // Check if the parameters unify between the LHS and RHS
  return paramsUnify(lhs.getParams(), rhs.getParams());
}

bool typesUnify(Type lhs, Type rhs, ArrayRef<llvm::StringRef> rhsRevPrefix) {
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

  // This always returns failure()
  static inline LogicalResult reportInvalid(
      llvm::function_ref<InFlightDiagnostic()> emitError, llvm::StringRef foundName,
      const char *aspect
  ) {
    // The implicit conversion from InFlightDiagnostic to LogicalResult in the return causes the
    // diagnostic to be printed.
    auto diag = emitError() << aspect << " must be one of ";
    Appender<InFlightDiagnostic>::append(diag);
    return diag << " but found '" << foundName << "'";
  }

  // This always returns failure()
  static inline LogicalResult reportInvalid(
      llvm::function_ref<InFlightDiagnostic()> emitError, const Attribute &found, const char *aspect
  ) {
    return reportInvalid(emitError, found.getAbstractAttribute().getName(), aspect);
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

} // namespace

//===------------------------------------------------------------------===//
// StructType
//===------------------------------------------------------------------===//

namespace {

// Parameters in the StructType must be one of the following:
//  - Integer constants
//  - SymbolRef (global constants defined in another module require non-flat ref)
//  - Type
using StructParamTypes = TypeList<IntegerAttr, SymbolRefAttr, TypeAttr>;

} // namespace

LogicalResult StructType::verify(
    llvm::function_ref<InFlightDiagnostic()> emitError, SymbolRefAttr nameRef, ArrayAttr params
) {
  if (params) {
    for (Attribute p : params) {
      if (!StructParamTypes::matches(p)) {
        return StructParamTypes::reportInvalid(emitError, p, "Struct parameter");
      }
    }
  }
  return success();
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

namespace {

// Dimensions in the ArrayType must be one of the following:
//  - Integer constants
//  - SymbolRef (global constants defined in another module require non-flat ref)
using ArrayDimensionTypes = TypeList<IntegerAttr, SymbolRefAttr>;

LogicalResult verifyArrayDimensionSizes(
    llvm::function_ref<InFlightDiagnostic()> emitError, llvm::ArrayRef<Attribute> dimensionSizes
) {
  // In LLZK, the number of array dimensions must always be known, i.e. `hasRank()==true`
  if (dimensionSizes.empty()) {
    return emitError().append("array must have at least one dimension");
  }
  // Rather than immediately returning on failure, we check all dimensions and aggregate to provide
  // as many errors are possible in a single verifier run.
  LogicalResult aggregateResult = success();
  for (Attribute a : dimensionSizes) {
    if (!ArrayDimensionTypes::matches(a)) {
      aggregateResult = ArrayDimensionTypes::reportInvalid(emitError, a, "Array dimension");
      assert(failed(aggregateResult)); // reportInvalid() always returns failure()
    }
  }
  return aggregateResult;
}

} // namespace

LogicalResult computeDimsFromShape(
    MLIRContext *ctx, llvm::ArrayRef<int64_t> shape, llvm::SmallVector<Attribute> &dimensionSizes
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
    llvm::function_ref<InFlightDiagnostic()> emitError, MLIRContext *ctx,
    llvm::ArrayRef<Attribute> dimensionSizes, llvm::SmallVector<int64_t> &shape
) {
  assert(shape.empty()); // fully computed by this function
  // Ensure all Attributes are valid Attribute classes for ArrayType.
  // In the case where `emitError==null`, we mirror how the verification failure is handled by
  // `*Type::get()` via `StorageUserBase` (i.e. use DefaultDiagnosticEmitFn and assert). See:
  //  https://github.com/llvm/llvm-project/blob/0897373f1a329a7a02f8ce3c501a05d2f9c89390/mlir/include/mlir/IR/StorageUniquerSupport.h#L179-L180
  auto errFunc = emitError ? llvm::unique_function<InFlightDiagnostic()>(emitError)
                           : mlir::detail::getDefaultDiagnosticEmitFn(ctx);
  if (failed(verifyArrayDimensionSizes(errFunc, dimensionSizes))) {
    assert(emitError);
    return failure();
  }

  // Convert the Attributes to int64_t
  for (Attribute a : dimensionSizes) {
    if (auto p = a.dyn_cast<IntegerAttr>()) {
      shape.push_back(p.getValue().getSExtValue());
    } else if (a.isa<SymbolRefAttr>()) {
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

ParseResult parseAttrVec(AsmParser &parser, llvm::SmallVector<Attribute> &value) {
  auto parseResult = FieldParser<llvm::SmallVector<Attribute>>::parse(parser);
  if (failed(parseResult)) {
    return parser.emitError(parser.getCurrentLocation(), "cannot parse array dimensions");
  }
  value.insert(value.begin(), parseResult->begin(), parseResult->end());
  return success();
}

void printAttrVec(AsmPrinter &printer, llvm::ArrayRef<Attribute> value) {
  llvm::raw_ostream &stream = printer.getStream();
  llvm::interleave(value, stream, [&stream](Attribute a) { a.print(stream, true); }, ",");
}

ParseResult parseDerivedShape(
    AsmParser &parser, llvm::SmallVector<int64_t> &shape,
    llvm::SmallVector<Attribute> dimensionSizes
) {
  // This is not actually parsing. It's computing the derived
  //  `shape` from the `dimensionSizes` attributes.
  auto emitError = [&parser] { return parser.emitError(parser.getCurrentLocation()); };
  return computeShapeFromDims(emitError, parser.getContext(), dimensionSizes, shape);
}
void printDerivedShape(AsmPrinter &, llvm::ArrayRef<int64_t>, llvm::ArrayRef<Attribute>) {
  // nothing to print, it's derived and therefore not represented in the output
}

LogicalResult ArrayType::verify(
    llvm::function_ref<InFlightDiagnostic()> emitError, Type elementType,
    llvm::ArrayRef<Attribute> dimensionSizes, llvm::ArrayRef<int64_t> shape
) {
  if (failed(verifyArrayDimensionSizes(emitError, dimensionSizes))) {
    return failure();
  }

  // Ensure array element type is valid
  if (!isValidArrayElemType(elementType)) {
    // Print proper message if `elementType` is not a valid LLZK type or
    //  if it's simply not the right kind of type for an array element.
    if (failed(checkValidType(emitError, elementType))) {
      return failure();
    }
    return emitError().append(
        "'", ArrayType::name, "' element type cannot be '", elementType.getAbstractType().getName(),
        "'"
    );
  }
  return success();
}

ArrayType
ArrayType::cloneWith(std::optional<llvm::ArrayRef<int64_t>> shape, Type elementType) const {
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
