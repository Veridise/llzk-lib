#include "llzk/Dialect/LLZK/IR/Ops.h"
#include "llzk/Dialect/LLZK/IR/Types.h"
#include "llzk/Dialect/LLZK/Util/AttributeHelper.h"
#include "llzk/Dialect/LLZK/Util/ErrorHelper.h"
#include "llzk/Dialect/LLZK/Util/StreamHelper.h"
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

void ShortTypeStringifier::appendSymName(StringRef str) {
  if (str.empty()) {
    ss << "@?";
  } else {
    ss << "@" << str;
  }
}

void ShortTypeStringifier::appendSymRef(SymbolRefAttr sa) {
  appendSymName(sa.getRootReference().getValue());
  for (FlatSymbolRefAttr nestedRef : sa.getNestedReferences()) {
    ss << "::";
    appendSymName(nestedRef.getValue());
  }
}

void ShortTypeStringifier::appendAnyAttr(Attribute a) {
  // Adapted from AsmPrinter::Impl::printAttributeImpl()
  if (llvm::isa<IntegerAttr>(a)) {
    IntegerAttr ia = llvm::cast<IntegerAttr>(a);
    Type ty = ia.getType();
    bool isUnsigned = ty.isUnsignedInteger() || ty.isSignlessInteger(1);
    ia.getValue().print(ss, !isUnsigned);
  } else if (llvm::isa<SymbolRefAttr>(a)) {
    appendSymRef(llvm::cast<SymbolRefAttr>(a));
  } else if (llvm::isa<TypeAttr>(a)) {
    append(llvm::cast<TypeAttr>(a).getValue());
  } else if (llvm::isa<AffineMapAttr>(a)) {
    ss << "!m<";
    // Filter to remove spaces
    filtered_raw_ostream fs(ss, [](char c) { return c == ' '; });
    llvm::cast<AffineMapAttr>(a).getValue().print(fs);
    fs.flush();
    ss << ">";
  } else if (llvm::isa<ArrayAttr>(a)) {
    append(llvm::cast<ArrayAttr>(a).getValue());
  } else {
    // All valid/legal cases must be covered above
    assertValidAttrForParamOfType(a);
  }
}

ShortTypeStringifier &ShortTypeStringifier::append(ArrayRef<Attribute> attrs) {
  llvm::interleave(attrs, ss, [this](Attribute a) { appendAnyAttr(a); }, "_");
  return *this;
}

ShortTypeStringifier &ShortTypeStringifier::append(Type type) {
  // Cases must be consistent with isValidTypeImpl() below.
  if (type.isSignlessInteger(1)) {
    ss << "b";
  } else if (llvm::isa<IndexType>(type)) {
    ss << "i";
  } else if (llvm::isa<FeltType>(type)) {
    ss << "f";
  } else if (llvm::isa<StringType>(type)) {
    ss << "s";
  } else if (llvm::isa<TypeVarType>(type)) {
    ss << "!v<";
    appendSymName(llvm::cast<TypeVarType>(type).getRefName());
    ss << ">";
  } else if (llvm::isa<ArrayType>(type)) {
    ArrayType at = llvm::cast<ArrayType>(type);
    ss << "!a<";
    append(at.getElementType());
    ss << ":";
    append(at.getDimensionSizes());
    ss << ">";
    // return ret;
  } else if (llvm::isa<StructType>(type)) {
    StructType st = llvm::cast<StructType>(type);
    ss << "!s<";
    appendSymRef(st.getNameRef());
    ss << "_";
    if (ArrayAttr params = st.getParams()) {
      append(params.getValue());
    }
    ss << ">";
  } else {
    ss << "!INVALID";
  }
  return *this;
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

  static void reportInvalid(EmitErrorFn emitError, StringRef foundName, const char *aspect) {
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

// Dimensions in the ArrayType must be one of the following:
//  - Integer constants
//  - SymbolRef (flat ref for struct params, non-flat for global constants from another module)
//  - AffineMap (for array created within a loop where size depends on loop variable)
using ArrayDimensionTypes = TypeList<IntegerAttr, SymbolRefAttr, AffineMapAttr>;

// Parameters in the StructType must be one of the following:
//  - Integer constants
//  - SymbolRef (flat ref for struct params, non-flat for global constants from another module)
//  - Type
//  - AffineMap (for array of non-homogeneous structs)
using StructParamTypes = TypeList<IntegerAttr, SymbolRefAttr, TypeAttr, AffineMapAttr>;

class AllowedTypes {
  bool _noFelt, _noString, _noStruct, _noArray, _noVar;
  bool _noStructParams;

public:
  AllowedTypes()
      : _noFelt(false), _noString(false), _noStruct(false), _noArray(false), _noVar(false),
        _noStructParams(false) {}

  AllowedTypes &noFelt() {
    _noFelt = true;
    return *this;
  }

  AllowedTypes &noString() {
    _noString = true;
    return *this;
  }

  AllowedTypes &noStruct() {
    _noStruct = true;
    return *this;
  }

  AllowedTypes &noArray() {
    _noArray = true;
    return *this;
  }

  AllowedTypes &noVar() {
    _noVar = true;
    return *this;
  }

  AllowedTypes &noStructParams(bool noStructParams = true) {
    _noStructParams = noStructParams;
    return *this;
  }

  AllowedTypes &onlyInt() { return noFelt().noString().noStruct().noArray().noVar(); }

  // This is the main check for allowed types.
  bool isValidTypeImpl(Type type);

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
    // Rather than immediately returning on failure, we check all dimensions and aggregate to
    // provide as many errors are possible in a single verifier run.
    bool success = true;
    for (Attribute a : dimensionSizes) {
      if (!ArrayDimensionTypes::matches(a)) {
        ArrayDimensionTypes::reportInvalid(emitError, a, "Array dimension");
        success = false;
      } else if (_noVar && !llvm::isa<IntegerAttr>(a)) {
        TypeList<IntegerAttr>::reportInvalid(emitError, a, "Concrete array dimension");
        success = false;
      } else if (failed(verifyIntAttrType(emitError, a))) {
        success = false;
      }
    }
    return success;
  }

  bool isValidArrayElemTypeImpl(Type type) {
    // ArrayType element can be any valid type sans ArrayType itself.
    return !llvm::isa<ArrayType>(type) && isValidTypeImpl(type);
  }

  bool isValidArrayTypeImpl(
      Type elementType, ArrayRef<Attribute> dimensionSizes,
      std::optional<EmitErrorFn> emitError = std::nullopt
  ) {
    if (!areValidArrayDimSizes(dimensionSizes, emitError)) {
      return false;
    }

    // Ensure array element type is valid
    if (!isValidArrayElemTypeImpl(elementType)) {
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

  bool isValidArrayTypeImpl(Type type) {
    if (ArrayType arrTy = llvm::dyn_cast<ArrayType>(type)) {
      return isValidArrayTypeImpl(arrTy.getElementType(), arrTy.getDimensionSizes());
    }
    return false;
  }

  // Note: The `_no*` flags here refer to Types nested within a TypeAttr parameter (if any) except
  // for the `_noStructParams` flag which requires that `params` is null or empty.
  bool
  areValidStructTypeParams(ArrayAttr params, std::optional<EmitErrorFn> emitError = std::nullopt) {
    bool success = true;
    if (!isNullOrEmpty(params)) {
      if (_noStructParams) {
        return false;
      }
      for (Attribute p : params) {
        if (!StructParamTypes::matches(p)) {
          StructParamTypes::reportInvalid(emitError, p, "Struct parameter");
          success = false;
        } else if (TypeAttr tyAttr = llvm::dyn_cast<TypeAttr>(p)) {
          if (!isValidTypeImpl(tyAttr.getValue())) {
            if (emitError.has_value()) {
              (*emitError)()
                  .append("expected a valid LLZK type but found ", tyAttr.getValue())
                  .report();
            }
            success = false;
          }
        } else if (_noVar && !llvm::isa<IntegerAttr>(p)) {
          TypeList<IntegerAttr>::reportInvalid(emitError, p, "Concrete struct parameter");
          success = false;
        } else if (failed(verifyIntAttrType(emitError, p))) {
          success = false;
        }
      }
    }
    return success;
  }

  // Note: The `_no*` flags here refer to Types nested within a TypeAttr parameter.
  bool isValidStructTypeImpl(Type type) {
    if (StructType sType = llvm::dyn_cast<StructType>(type)) {
      return areValidStructTypeParams(sType.getParams());
    }
    return false;
  }
};

bool AllowedTypes::isValidTypeImpl(Type type) {
  return type.isSignlessInteger(1) || llvm::isa<IndexType>(type) ||
         (!_noFelt && llvm::isa<FeltType>(type)) || (!_noString && llvm::isa<StringType>(type)) ||
         (!_noVar && llvm::isa<TypeVarType>(type)) || (!_noStruct && isValidStructTypeImpl(type)) ||
         (!_noArray && isValidArrayTypeImpl(type));
}

} // namespace

bool isValidType(Type type) { return AllowedTypes().isValidTypeImpl(type); }

bool isValidEmitEqType(Type type) {
  return AllowedTypes().noString().noStruct().isValidTypeImpl(type);
}

// Allowed types must align with StructParamTypes (defined below)
bool isValidConstReadType(Type type) {
  return AllowedTypes().noString().noStruct().noArray().isValidTypeImpl(type);
}

bool isValidArrayElemType(Type type) { return AllowedTypes().isValidArrayElemTypeImpl(type); }

bool isValidArrayType(Type type) { return AllowedTypes().isValidArrayTypeImpl(type); }

bool isConcreteType(Type type, bool allowStructParams) {
  return AllowedTypes().noVar().noStructParams(!allowStructParams).isValidTypeImpl(type);
}

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

void updateMap(UnificationMap *unifications, Side side, SymbolRefAttr symRef, Attribute attr) {
  if (unifications) {
    auto key = std::make_pair(symRef, side);
    auto it = unifications->find(key);
    if (it != unifications->end()) {
      it->second = nullptr;
    } else {
      unifications->try_emplace(key, attr);
    }
  }
}

bool paramAttrUnify(UnificationMap *unifications, Attribute lhsAttr, Attribute rhsAttr) {
  assertValidAttrForParamOfType(lhsAttr);
  assertValidAttrForParamOfType(rhsAttr);
  // IntegerAttr and AffineMapAttr only unify via equality and the others may. Additionally,
  //  if either attribute is a symbol ref, we assume they unify because a later pass with a
  //  more involved value analysis is required to check if they are actually the same value.
  if (lhsAttr == rhsAttr) {
    return true;
  }
  if (SymbolRefAttr lhsSymRef = lhsAttr.dyn_cast<SymbolRefAttr>()) {
    updateMap(unifications, Side::LHS, lhsSymRef, rhsAttr);
    return true;
  }
  if (SymbolRefAttr rhsSymRef = rhsAttr.dyn_cast<SymbolRefAttr>()) {
    updateMap(unifications, Side::RHS, rhsSymRef, lhsAttr);
    return true;
  }
  // If both are type refs, check for unification of the types.
  if (TypeAttr lhsTy = lhsAttr.dyn_cast<TypeAttr>()) {
    if (TypeAttr rhsTy = rhsAttr.dyn_cast<TypeAttr>()) {
      return typesUnify(lhsTy.getValue(), rhsTy.getValue(), {}, unifications);
    }
  }
  return false;
}
} // namespace

bool typeParamsUnify(
    const ArrayRef<Attribute> &lhsParams, const ArrayRef<Attribute> &rhsParams,
    UnificationMap *unifications
) {
  return (lhsParams.size() == rhsParams.size()) &&
         std::equal(
             lhsParams.begin(), lhsParams.end(), rhsParams.begin(),
             std::bind_front(paramAttrUnify, unifications)
         );
}

/// Return `true` iff the two ArrayAttr instances containing StructType or ArrayType parameters
/// are equivalent or could be equivalent after full instantiation of struct parameters.
bool typeParamsUnify(
    const ArrayAttr &lhsParams, const ArrayAttr &rhsParams, UnificationMap *unifications
) {
  if (lhsParams && rhsParams) {
    return typeParamsUnify(lhsParams.getValue(), rhsParams.getValue(), unifications);
  }
  // When one or the other is null, they're only equivalent if both are null
  return !lhsParams && !rhsParams;
}

bool arrayTypesUnify(
    ArrayType lhs, ArrayType rhs, ArrayRef<StringRef> rhsRevPrefix, UnificationMap *unifications
) {
  // Check if the element types of the two arrays can unify
  if (!typesUnify(lhs.getElementType(), rhs.getElementType(), rhsRevPrefix, unifications)) {
    return false;
  }
  // Check if the dimension size attributes unify between the LHS and RHS
  return typeParamsUnify(lhs.getDimensionSizes(), rhs.getDimensionSizes(), unifications);
}

bool structTypesUnify(
    StructType lhs, StructType rhs, ArrayRef<StringRef> rhsRevPrefix, UnificationMap *unifications
) {
  // Check if it references the same StructDefOp, considering the additional RHS path prefix.
  SmallVector<StringRef> rhsNames = getNames(rhs.getNameRef());
  rhsNames.insert(rhsNames.begin(), rhsRevPrefix.rbegin(), rhsRevPrefix.rend());
  if (rhsNames != getNames(lhs.getNameRef())) {
    return false;
  }
  // Check if the parameters unify between the LHS and RHS
  return typeParamsUnify(lhs.getParams(), rhs.getParams(), unifications);
}

bool typesUnify(
    Type lhs, Type rhs, ArrayRef<StringRef> rhsRevPrefix, UnificationMap *unifications
) {
  if (lhs == rhs) {
    return true;
  }
  // A type variable can be any type, thus it unifies with anything.
  if (TypeVarType lhsTvar = llvm::dyn_cast<TypeVarType>(lhs)) {
    updateMap(unifications, Side::LHS, lhsTvar.getNameRef(), TypeAttr::get(rhs));
    return true;
  }
  if (TypeVarType rhsTvar = llvm::dyn_cast<TypeVarType>(rhs)) {
    updateMap(unifications, Side::RHS, rhsTvar.getNameRef(), TypeAttr::get(lhs));
    return true;
  }
  if (llvm::isa<StructType>(lhs) && llvm::isa<StructType>(rhs)) {
    return structTypesUnify(
        llvm::cast<StructType>(lhs), llvm::cast<StructType>(rhs), rhsRevPrefix, unifications
    );
  }
  if (llvm::isa<ArrayType>(lhs) && llvm::isa<ArrayType>(rhs)) {
    return arrayTypesUnify(
        llvm::cast<ArrayType>(lhs), llvm::cast<ArrayType>(rhs), rhsRevPrefix, unifications
    );
  }
  return false;
}

SmallVector<Attribute> forceIntAttrType(ArrayRef<Attribute> attrList) {
  return map_to_vector(attrList, [](Attribute a) -> Attribute {
    if (IntegerAttr intAttr = dyn_cast<IntegerAttr>(a)) {
      Type attrTy = intAttr.getType();
      if (!AllowedTypes().onlyInt().isValidTypeImpl(attrTy)) {
        return IntegerAttr::get(IndexType::get(intAttr.getContext()), intAttr.getValue());
      }
    }
    return a;
  });
}

LogicalResult verifyIntAttrType(std::optional<EmitErrorFn> emitError, Attribute in) {
  if (IntegerAttr intAttr = llvm::dyn_cast<IntegerAttr>(in)) {
    Type attrTy = intAttr.getType();
    if (!AllowedTypes().onlyInt().isValidTypeImpl(attrTy)) {
      if (emitError.has_value()) {
        (*emitError)()
            .append("IntegerAttr must have type 'index' or 'i1' but found '", attrTy, "'")
            .report();
      }
      return failure();
    }
  }
  return success();
}

ParseResult parseAttrVec(AsmParser &parser, SmallVector<Attribute> &value) {
  auto parseResult = FieldParser<SmallVector<Attribute>>::parse(parser);
  if (failed(parseResult)) {
    return parser.emitError(parser.getCurrentLocation(), "failed to parse array dimensions");
  }
  value = forceIntAttrType(*parseResult);
  return success();
}

namespace {

// Adapted from AsmPrinter::printStrippedAttrOrType(), but without printing type.
void printAttrs(AsmPrinter &printer, ArrayRef<Attribute> attrs, const StringRef &separator) {
  llvm::interleave(attrs, printer.getStream(), [&printer](Attribute a) {
    if (succeeded(printer.printAlias(a))) {
      return;
    }
    raw_ostream &os = printer.getStream();
    uint64_t posPrior = os.tell();
    printer.printAttributeWithoutType(a);
    // Fallback to printing with prefix if the above failed to write anything to the output stream.
    if (posPrior == os.tell()) {
      printer << a;
    }
  }, separator);
}

} // namespace

void printAttrVec(AsmPrinter &printer, ArrayRef<Attribute> value) {
  printAttrs(printer, value, ",");
}

ParseResult parseStructParams(AsmParser &parser, ArrayAttr &value) {
  auto parseResult = FieldParser<ArrayAttr>::parse(parser);
  if (failed(parseResult)) {
    return parser.emitError(parser.getCurrentLocation(), "failed to parse struct parameters");
  }
  SmallVector<Attribute> own = forceIntAttrType(parseResult->getValue());
  value = parser.getBuilder().getArrayAttr(own);
  return success();
}
void printStructParams(AsmPrinter &printer, ArrayAttr value) {
  printer << '[';
  printAttrs(printer, value.getValue(), ", ");
  printer << ']';
}

//===------------------------------------------------------------------===//
// StructType
//===------------------------------------------------------------------===//

LogicalResult StructType::verify(EmitErrorFn emitError, SymbolRefAttr nameRef, ArrayAttr params) {
  return success(AllowedTypes().areValidStructTypeParams(params, emitError));
}

FailureOr<SymbolLookupResult<StructDefOp>>
StructType::getDefinition(SymbolTableCollection &symbolTable, Operation *op) const {
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
  Builder builder(ctx);
  dimensionSizes = llvm::map_to_vector(shape, [&builder](int64_t v) -> Attribute {
    return builder.getIndexAttr(v);
  });
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
  if (!AllowedTypes().areValidArrayDimSizes(dimensionSizes, errFunc)) {
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
  return success(AllowedTypes().isValidArrayTypeImpl(elementType, dimensionSizes, emitError));
}

int64_t ArrayType::getNumElements() const { return ShapedType::getNumElements(getShape()); }

ArrayType ArrayType::cloneWith(std::optional<ArrayRef<int64_t>> shape, Type elementType) const {
  return ArrayType::get(elementType, shape.has_value() ? shape.value() : getShape());
}

ArrayType
ArrayType::cloneWith(Type elementType, std::optional<ArrayRef<Attribute>> dimensions) const {
  return ArrayType::get(
      elementType, dimensions.has_value() ? dimensions.value() : getDimensionSizes()
  );
}

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
