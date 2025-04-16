//===-- Types.cpp - LLZK type implementations -------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/LLZK/IR/Ops.h"
#include "llzk/Dialect/LLZK/IR/Types.h"
#include "llzk/Dialect/LLZK/Util/AttributeHelper.h"
#include "llzk/Dialect/LLZK/Util/ErrorHelper.h"
#include "llzk/Dialect/LLZK/Util/StreamHelper.h"
#include "llzk/Dialect/LLZK/Util/SymbolHelper.h"

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/Support/LogicalResult.h>

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>

#include <cassert>

namespace llzk {

using namespace mlir;

//===------------------------------------------------------------------===//
// Helpers
//===------------------------------------------------------------------===//

void ShortTypeStringifier::appendSymName(StringRef str) {
  if (str.empty()) {
    ss << '?';
  } else {
    ss << '@' << str;
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
    ss << '>';
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
    ss << 'b';
  } else if (llvm::isa<IndexType>(type)) {
    ss << 'i';
  } else if (llvm::isa<FeltType>(type)) {
    ss << 'f';
  } else if (llvm::isa<StringType>(type)) {
    ss << 's';
  } else if (llvm::isa<TypeVarType>(type)) {
    ss << "!t<";
    appendSymName(llvm::cast<TypeVarType>(type).getRefName());
    ss << '>';
  } else if (llvm::isa<ArrayType>(type)) {
    ArrayType at = llvm::cast<ArrayType>(type);
    ss << "!a<";
    append(at.getElementType());
    ss << ':';
    append(at.getDimensionSizes());
    ss << '>';
  } else if (llvm::isa<StructType>(type)) {
    StructType st = llvm::cast<StructType>(type);
    ss << "!s<";
    appendSymRef(st.getNameRef());
    if (ArrayAttr params = st.getParams()) {
      ss << '_';
      append(params.getValue());
    }
    ss << '>';
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
      stream << '\'' << Ty::name << '\'';
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
      stream << '[';
      append<Types...>(stream);
      stream << ']';
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

  static inline void reportInvalid(EmitErrorFn emitError, Attribute found, const char *aspect) {
    if (emitError) {
      reportInvalid(emitError, found.getAbstractAttribute().getName(), aspect);
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
  struct ColumnCheckData {
    SymbolTableCollection *symbolTable = nullptr;
    Operation *op = nullptr;
  };

  bool no_felt : 1 = false;
  bool no_string : 1 = false;
  bool no_non_signal_struct : 1 = false;
  bool no_signal_struct : 1 = false;
  bool no_array : 1 = false;
  bool no_var : 1 = false;
  bool no_int : 1 = false;
  bool no_struct_params : 1 = false;
  bool must_be_column : 1 = false;

  ColumnCheckData columnCheck;

  /// Validates that, if columns are a requirement, the struct type has columns.
  /// If columns are not a requirement returns true early since the pointers required
  /// for lookup may be null.
  bool validColumns(StructType s) {
    if (!must_be_column) {
      return true;
    }
    assert(columnCheck.symbolTable);
    assert(columnCheck.op);
    return succeeded(s.hasColumns(*columnCheck.symbolTable, columnCheck.op));
  }

public:
  constexpr AllowedTypes &noFelt() {
    no_felt = true;
    return *this;
  }

  constexpr AllowedTypes &noString() {
    no_string = true;
    return *this;
  }

  constexpr AllowedTypes &noStruct() {
    no_non_signal_struct = true;
    no_signal_struct = true;
    return *this;
  }

  constexpr AllowedTypes &noStructExceptSignal() {
    no_non_signal_struct = true;
    no_signal_struct = false;
    return *this;
  }

  constexpr AllowedTypes &noArray() {
    no_array = true;
    return *this;
  }

  constexpr AllowedTypes &noVar() {
    no_var = true;
    return *this;
  }

  constexpr AllowedTypes &noInt() {
    no_int = true;
    return *this;
  }

  constexpr AllowedTypes &noStructParams(bool noStructParams = true) {
    no_struct_params = noStructParams;
    return *this;
  }

  constexpr AllowedTypes &onlyInt() {
    no_int = false;
    return noFelt().noString().noStruct().noArray().noVar();
  }

  constexpr AllowedTypes &mustBeColumn(SymbolTableCollection &symbolTable, Operation *op) {
    must_be_column = true;
    columnCheck.symbolTable = &symbolTable;
    columnCheck.op = op;
    return *this;
  }

  // This is the main check for allowed types.
  bool isValidTypeImpl(Type type);

  bool areValidArrayDimSizes(ArrayRef<Attribute> dimensionSizes, EmitErrorFn emitError = nullptr) {
    // In LLZK, the number of array dimensions must always be known, i.e. `hasRank()==true`
    if (dimensionSizes.empty()) {
      if (emitError) {
        emitError().append("array must have at least one dimension").report();
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
      } else if (no_var && !llvm::isa<IntegerAttr>(a)) {
        TypeList<IntegerAttr>::reportInvalid(emitError, a, "Concrete array dimension");
        success = false;
      } else if (failed(verifyAffineMapAttrType(emitError, a))) {
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
      Type elementType, ArrayRef<Attribute> dimensionSizes, EmitErrorFn emitError = nullptr
  ) {
    if (!areValidArrayDimSizes(dimensionSizes, emitError)) {
      return false;
    }

    // Ensure array element type is valid
    if (!isValidArrayElemTypeImpl(elementType)) {
      if (emitError) {
        // Print proper message if `elementType` is not a valid LLZK type or
        //  if it's simply not the right kind of type for an array element.
        if (succeeded(checkValidType(emitError, elementType))) {
          emitError()
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

  // Note: The `no*` flags here refer to Types nested within a TypeAttr parameter (if any) except
  // for the `no_struct_params` flag which requires that `params` is null or empty.
  bool areValidStructTypeParams(ArrayAttr params, EmitErrorFn emitError = nullptr) {
    if (isNullOrEmpty(params)) {
      return true;
    }
    if (no_struct_params) {
      return false;
    }
    bool success = true;
    for (Attribute p : params) {
      if (!StructParamTypes::matches(p)) {
        StructParamTypes::reportInvalid(emitError, p, "Struct parameter");
        success = false;
      } else if (TypeAttr tyAttr = llvm::dyn_cast<TypeAttr>(p)) {
        if (!isValidTypeImpl(tyAttr.getValue())) {
          if (emitError) {
            emitError().append("expected a valid LLZK type but found ", tyAttr.getValue()).report();
          }
          success = false;
        }
      } else if (no_var && !llvm::isa<IntegerAttr>(p)) {
        TypeList<IntegerAttr>::reportInvalid(emitError, p, "Concrete struct parameter");
        success = false;
      } else if (failed(verifyAffineMapAttrType(emitError, p))) {
        success = false;
      } else if (failed(verifyIntAttrType(emitError, p))) {
        success = false;
      }
    }

    return success;
  }

  // Note: The `no*` flags here refer to Types nested within a TypeAttr parameter.
  bool isValidStructTypeImpl(Type type, bool allowSignalStruct, bool allowNonSignalStruct) {
    if (!allowSignalStruct && !allowNonSignalStruct) {
      return false;
    }
    if (StructType sType = llvm::dyn_cast<StructType>(type); sType && validColumns(sType)) {
      return (allowSignalStruct && isSignalType(sType)) ||
             (allowNonSignalStruct && areValidStructTypeParams(sType.getParams()));
    }
    return false;
  }
};

bool AllowedTypes::isValidTypeImpl(Type type) {
  assert(
      !(no_int && no_felt && no_string && no_var && no_non_signal_struct && no_signal_struct &&
        no_array) &&
      "All types have been deactivated"
  );
  return (!no_int && type.isSignlessInteger(1)) || (!no_int && llvm::isa<IndexType>(type)) ||
         (!no_felt && llvm::isa<FeltType>(type)) || (!no_string && llvm::isa<StringType>(type)) ||
         (!no_var && llvm::isa<TypeVarType>(type)) || (!no_array && isValidArrayTypeImpl(type)) ||
         isValidStructTypeImpl(type, !no_signal_struct, !no_non_signal_struct);
}

} // namespace

bool isValidType(Type type) { return AllowedTypes().isValidTypeImpl(type); }

bool isValidColumnType(Type type, SymbolTableCollection &symbolTable, Operation *op) {
  return AllowedTypes().noString().noInt().mustBeColumn(symbolTable, op).isValidTypeImpl(type);
}

bool isValidGlobalType(Type type) { return AllowedTypes().noVar().isValidTypeImpl(type); }

bool isValidEmitEqType(Type type) {
  return AllowedTypes().noString().noStructExceptSignal().isValidTypeImpl(type);
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
    return isSignalType(structParamTy);
  }
  return false;
}

bool isSignalType(StructType sType) {
  // Only check the leaf part of the reference (i.e. just the struct name itself) to allow cases
  // where the `COMPONENT_NAME_SIGNAL` struct may be placed within some nesting of modules, as
  // happens when it's imported via an IncludeOp.
  return sType.getNameRef().getLeafReference() == COMPONENT_NAME_SIGNAL;
}

bool hasAffineMapAttr(Type type) {
  bool encountered = false;
  type.walk([&](AffineMapAttr a) {
    encountered = true;
    return WalkResult::interrupt();
  });
  return encountered;
}

namespace {

/// Optional result from type unifications. Maps `AffineMapAttr` appearing in one type to the
/// associated `IntegerAttr` from the other type at the same nested position. The `Side` enum in the
/// key indicates which input expression the `AffineMapAttr` is from. Additionally, if a conflict is
/// found (i.e. multiple occurances of a specific `AffineMapAttr` on the same side map to different
/// `IntegerAttr` from the other side), the mapped value will be `nullptr`.
using AffineInstantiations = DenseMap<std::pair<AffineMapAttr, Side>, IntegerAttr>;

struct UnifierImpl {
  ArrayRef<StringRef> rhsRevPrefix;
  UnificationMap *unifications;
  AffineInstantiations *affineToIntTracker;
  // This optional function can be used to provide an exception to the standard unification
  // rules and return a true/success result when it otherwise may not.
  llvm::function_ref<bool(Type oldTy, Type newTy)> overrideSuccess;

  UnifierImpl(UnificationMap *unificationMap, ArrayRef<StringRef> rhsReversePrefix = {})
      : rhsRevPrefix(rhsReversePrefix), unifications(unificationMap), affineToIntTracker(nullptr),
        overrideSuccess(nullptr) {}

  bool typeParamsUnify(const ArrayRef<Attribute> &lhsParams, const ArrayRef<Attribute> &rhsParams) {
    return (lhsParams.size() == rhsParams.size()) &&
           std::equal(
               lhsParams.begin(), lhsParams.end(), rhsParams.begin(),
               std::bind_front(&UnifierImpl::paramAttrUnify, this)
           );
  }

  UnifierImpl &trackAffineToInt(AffineInstantiations *tracker) {
    this->affineToIntTracker = tracker;
    return *this;
  }

  UnifierImpl &withOverrides(llvm::function_ref<bool(Type oldTy, Type newTy)> overrides) {
    this->overrideSuccess = overrides;
    return *this;
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

  bool arrayTypesUnify(ArrayType lhs, ArrayType rhs) {
    // Check if the element types of the two arrays can unify
    if (!typesUnify(lhs.getElementType(), rhs.getElementType())) {
      return false;
    }
    // Check if the dimension size attributes unify between the LHS and RHS
    return typeParamsUnify(lhs.getDimensionSizes(), rhs.getDimensionSizes());
  }

  bool structTypesUnify(StructType lhs, StructType rhs) {
    // Check if it references the same StructDefOp, considering the additional RHS path prefix.
    SmallVector<StringRef> rhsNames = getNames(rhs.getNameRef());
    rhsNames.insert(rhsNames.begin(), rhsRevPrefix.rbegin(), rhsRevPrefix.rend());
    if (rhsNames != getNames(lhs.getNameRef())) {
      return false;
    }
    // Check if the parameters unify between the LHS and RHS
    return typeParamsUnify(lhs.getParams(), rhs.getParams());
  }

  bool typesUnify(Type lhs, Type rhs) {
    if (lhs == rhs) {
      return true;
    }
    if (overrideSuccess && overrideSuccess(lhs, rhs)) {
      return true;
    }
    // A type variable can be any type, thus it unifies with anything.
    if (TypeVarType lhsTvar = llvm::dyn_cast<TypeVarType>(lhs)) {
      track(Side::LHS, lhsTvar.getNameRef(), TypeAttr::get(rhs));
      return true;
    }
    if (TypeVarType rhsTvar = llvm::dyn_cast<TypeVarType>(rhs)) {
      track(Side::RHS, rhsTvar.getNameRef(), TypeAttr::get(lhs));
      return true;
    }
    if (llvm::isa<StructType>(lhs) && llvm::isa<StructType>(rhs)) {
      return structTypesUnify(llvm::cast<StructType>(lhs), llvm::cast<StructType>(rhs));
    }
    if (llvm::isa<ArrayType>(lhs) && llvm::isa<ArrayType>(rhs)) {
      return arrayTypesUnify(llvm::cast<ArrayType>(lhs), llvm::cast<ArrayType>(rhs));
    }
    return false;
  }

private:
  template <typename Tracker, typename Key, typename Val>
  inline void track(Tracker &tracker, Side side, Key keyHead, Val val) {
    auto key = std::make_pair(keyHead, side);
    auto it = tracker.find(key);
    if (it != tracker.end()) {
      it->second = nullptr;
    } else {
      tracker.try_emplace(key, val);
    }
  }

  void track(Side side, SymbolRefAttr symRef, Attribute attr) {
    if (unifications) {
      track(*unifications, side, symRef, attr);
    }
  }

  void track(Side side, AffineMapAttr affineAttr, IntegerAttr intAttr) {
    if (affineToIntTracker) {
      track(*affineToIntTracker, side, affineAttr, intAttr);
    }
  }

  bool paramAttrUnify(Attribute lhsAttr, Attribute rhsAttr) {
    assertValidAttrForParamOfType(lhsAttr);
    assertValidAttrForParamOfType(rhsAttr);
    // Straightforward equality check.
    if (lhsAttr == rhsAttr) {
      return true;
    }
    // AffineMapAttr can unify with IntegerAttr because struct parameter instantiation will result
    // in conversion of AffineMapAttr to IntegerAttr.
    if (AffineMapAttr lhsAffine = lhsAttr.dyn_cast<AffineMapAttr>()) {
      if (IntegerAttr rhsInt = rhsAttr.dyn_cast<IntegerAttr>()) {
        track(Side::LHS, lhsAffine, rhsInt);
        return true;
      }
    }
    if (AffineMapAttr rhsAffine = rhsAttr.dyn_cast<AffineMapAttr>()) {
      if (IntegerAttr lhsInt = lhsAttr.dyn_cast<IntegerAttr>()) {
        track(Side::RHS, rhsAffine, lhsInt);
        return true;
      }
    }
    // If either side is a SymbolRefAttr, assume they unify because either flattening or a pass with
    // a more involved value analysis is required to check if they are actually the same value.
    if (SymbolRefAttr lhsSymRef = lhsAttr.dyn_cast<SymbolRefAttr>()) {
      track(Side::LHS, lhsSymRef, rhsAttr);
      return true;
    }
    if (SymbolRefAttr rhsSymRef = rhsAttr.dyn_cast<SymbolRefAttr>()) {
      track(Side::RHS, rhsSymRef, lhsAttr);
      return true;
    }
    // If both are type refs, check for unification of the types.
    if (TypeAttr lhsTy = lhsAttr.dyn_cast<TypeAttr>()) {
      if (TypeAttr rhsTy = rhsAttr.dyn_cast<TypeAttr>()) {
        return typesUnify(lhsTy.getValue(), rhsTy.getValue());
      }
    }
    // Otherwise, they do not unify.
    return false;
  }
};

} // namespace

bool typeParamsUnify(
    const ArrayRef<Attribute> &lhsParams, const ArrayRef<Attribute> &rhsParams,
    UnificationMap *unifications
) {
  return UnifierImpl(unifications).typeParamsUnify(lhsParams, rhsParams);
}

/// Return `true` iff the two ArrayAttr instances containing StructType or ArrayType parameters
/// are equivalent or could be equivalent after full instantiation of struct parameters.
bool typeParamsUnify(
    const ArrayAttr &lhsParams, const ArrayAttr &rhsParams, UnificationMap *unifications
) {
  return UnifierImpl(unifications).typeParamsUnify(lhsParams, rhsParams);
}

bool arrayTypesUnify(
    ArrayType lhs, ArrayType rhs, ArrayRef<StringRef> rhsReversePrefix, UnificationMap *unifications
) {
  return UnifierImpl(unifications, rhsReversePrefix).arrayTypesUnify(lhs, rhs);
}

bool structTypesUnify(
    StructType lhs, StructType rhs, ArrayRef<StringRef> rhsReversePrefix,
    UnificationMap *unifications
) {
  return UnifierImpl(unifications, rhsReversePrefix).structTypesUnify(lhs, rhs);
}

bool typesUnify(
    Type lhs, Type rhs, ArrayRef<StringRef> rhsReversePrefix, UnificationMap *unifications
) {
  return UnifierImpl(unifications, rhsReversePrefix).typesUnify(lhs, rhs);
}

bool isMoreConcreteUnification(
    Type oldTy, Type newTy, llvm::function_ref<bool(Type oldTy, Type newTy)> knownOldToNew
) {
  UnificationMap unifications;
  AffineInstantiations affineInstantiations;
  // Run type unification with the addition that affine map can become integer in the new type.
  if (!UnifierImpl(&unifications)
           .trackAffineToInt(&affineInstantiations)
           .withOverrides(knownOldToNew)
           .typesUnify(oldTy, newTy)) {
    return false;
  }

  // If either map contains RHS-keyed mappings then the old type is "more concrete" than the new.
  // In the UnificationMap, a RHS key would indicate that the new type contains a SymbolRef (i.e.
  // the "least concrete" attribute kind) where the old type contained any other attribute. In the
  // AffineInstantiations map, a RHS key would indicate that the new type contains an AffineMapAttr
  // where the old type contains an IntegerAttr.
  auto entryIsRHS = [](const auto &entry) { return entry.first.second == Side::RHS; };
  return !llvm::any_of(unifications, entryIsRHS) && !llvm::any_of(affineInstantiations, entryIsRHS);
}

IntegerAttr forceIntType(IntegerAttr attr) {
  if (AllowedTypes().onlyInt().isValidTypeImpl(attr.getType())) {
    return attr;
  }
  return IntegerAttr::get(IndexType::get(attr.getContext()), attr.getValue());
}

Attribute forceIntAttrType(Attribute attr) {
  if (IntegerAttr intAttr = dyn_cast<IntegerAttr>(attr)) {
    return forceIntType(intAttr);
  }
  return attr;
}

SmallVector<Attribute> forceIntAttrTypes(ArrayRef<Attribute> attrList) {
  return llvm::map_to_vector(attrList, forceIntAttrType);
}

LogicalResult verifyIntAttrType(EmitErrorFn emitError, Attribute in) {
  if (IntegerAttr intAttr = llvm::dyn_cast<IntegerAttr>(in)) {
    Type attrTy = intAttr.getType();
    if (!AllowedTypes().onlyInt().isValidTypeImpl(attrTy)) {
      if (emitError) {
        emitError()
            .append("IntegerAttr must have type 'index' or 'i1' but found '", attrTy, "'")
            .report();
      }
      return failure();
    }
  }
  return success();
}

LogicalResult verifyAffineMapAttrType(EmitErrorFn emitError, Attribute in) {
  if (AffineMapAttr affineAttr = llvm::dyn_cast<AffineMapAttr>(in)) {
    AffineMap map = affineAttr.getValue();
    if (map.getNumResults() != 1) {
      if (emitError) {
        emitError()
            .append(
                "AffineMapAttr must yield a single result, but found ", map.getNumResults(),
                " results"
            )
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
  value = forceIntAttrTypes(*parseResult);
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
  SmallVector<Attribute> own = forceIntAttrTypes(parseResult->getValue());
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

LogicalResult StructType::hasColumns(SymbolTableCollection &symbolTable, Operation *op) const {
  auto lookup = getDefinition(symbolTable, op);
  if (failed(lookup)) {
    return lookup;
  }
  return lookup->get().hasColumns();
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
