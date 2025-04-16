//===-- Types.h -------------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llzk/Dialect/LLZK/IR/Dialect.h"
#include "llzk/Dialect/LLZK/Util/ErrorHelper.h"
#include "llzk/Dialect/LLZK/Util/SymbolLookup.h" // IWYU pragma: keep

#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/IR/Types.h>

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/STLFunctionalExtras.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/TypeSwitch.h>

#include <vector>

// forward-declare ops
#define GET_OP_FWD_DEFINES
#include "llzk/Dialect/LLZK/IR/Ops.h.inc"

// Include TableGen'd declarations
#define GET_TYPEDEF_CLASSES
#include "llzk/Dialect/LLZK/IR/Types.h.inc"

namespace llzk {

/// Note: If any symbol refs in an input Type/Attribute use any of the special characters that this
/// class generates, they are not escaped. That means these string representations are not safe to
/// reverse back into a Type. It's only intended to produce a unique name for instantiated structs
/// that may give some hint when debugging regarding the original struct name and the params used.
class ShortTypeStringifier {
  std::string ret;
  llvm::raw_string_ostream ss;

public:
  ShortTypeStringifier() : ret(), ss(ret) {}
  std::string str() const { return ret; }
  ShortTypeStringifier &append(mlir::Type);
  ShortTypeStringifier &append(mlir::ArrayRef<mlir::Attribute>);

private:
  void appendAnyAttr(mlir::Attribute);
  void appendSymRef(mlir::SymbolRefAttr);
  void appendSymName(mlir::StringRef);
};

/// Return a brief string representation of the given LLZK type.
static inline std::string shortString(mlir::Type type) {
  return ShortTypeStringifier().append(type).str();
}

/// Return a brief string representation of the attribute list from a parameterized type.
static inline std::string shortString(mlir::ArrayRef<mlir::Attribute> attrs) {
  return ShortTypeStringifier().append(attrs).str();
}

// This function asserts that the given Attribute kind is legal within the LLZK types that can
// contain Attribute parameters (i.e. ArrayType, StructType, and TypeVarType). This should be used
// in any function that examines the attribute parameters within parameterized LLZK types to ensure
// that the function handles all possible cases properly, especially if more legal attributes are
// added in the future. Throw a fatal error if anything illegal is found, indicating that the caller
// of this function should be updated.
void assertValidAttrForParamOfType(mlir::Attribute attr);

/// valid types: {I1, Index, String, FeltType, StructType, ArrayType, TypeVarType}
bool isValidType(mlir::Type type);

/// valid types: {FeltType, StructType (with columns), ArrayType (that contains a valid column
/// type)}
bool isValidColumnType(
    mlir::Type type, mlir::SymbolTableCollection &symbolTable, mlir::Operation *op
);

/// valid types: isValidType() - {TypeVarType} - {types with variable parameters}
bool isValidGlobalType(mlir::Type type);

/// valid types: isValidType() - {String, StructType} (excluded via any type parameter nesting)
bool isValidEmitEqType(mlir::Type type);

/// valid types: {I1, Index, FeltType, TypeVarType}
bool isValidConstReadType(mlir::Type type);

/// valid types: isValidType() - {ArrayType}
bool isValidArrayElemType(mlir::Type type);

/// Checks if the type is a LLZK Array and it also contains a valid LLZK type.
bool isValidArrayType(mlir::Type type);

/// Return `false` iff the type contains any `TypeVarType`
bool isConcreteType(mlir::Type type, bool allowStructParams = true);

inline mlir::LogicalResult checkValidType(EmitErrorFn emitError, mlir::Type type) {
  if (!isValidType(type)) {
    return emitError() << "expected a valid LLZK type but found " << type;
  } else {
    return mlir::success();
  }
}

/// Return `true` iff the given type is a StructType referencing the `COMPONENT_NAME_SIGNAL` struct.
bool isSignalType(mlir::Type type);

/// Return `true` iff the given StructType is referencing the `COMPONENT_NAME_SIGNAL` struct.
bool isSignalType(StructType sType);

/// @brief Return `true` iff the given type contains an AffineMapAttr.
bool hasAffineMapAttr(mlir::Type type);

enum class Side { EMPTY = 0, LHS, RHS, TOMB };
static inline mlir::raw_ostream &operator<<(mlir::raw_ostream &os, const Side &val) {
  switch (val) {
  case Side::EMPTY:
    os << "EMPTY";
    break;
  case Side::TOMB:
    os << "TOMB";
    break;
  case Side::LHS:
    os << "LHS";
    break;
  case Side::RHS:
    os << "RHS";
    break;
  }
  return os;
}
} // namespace llzk

namespace llvm {
template <> struct DenseMapInfo<llzk::Side> {
  using T = llzk::Side;
  static inline T getEmptyKey() { return T::EMPTY; }
  static inline T getTombstoneKey() { return T::TOMB; }
  static unsigned getHashValue(const T &val) {
    using UT = std::underlying_type_t<T>;
    return llvm::DenseMapInfo<UT>::getHashValue(static_cast<UT>(val));
  }
  static bool isEqual(const T &lhs, const T &rhs) { return lhs == rhs; }
};
} // namespace llvm

namespace llzk {

/// Optional result from type unifications. Maps `SymbolRefAttr` appearing in one type to the
/// associated `Attribute` from the other type at the same nested position. The `Side` enum in the
/// key indicates which input expression the `SymbolRefAttr` is from. Additionally, if a conflict is
/// found (i.e. multiple occurances of a specific `SymbolRefAttr` on the same side map to different
/// Attributes from the other side). The mapped value will be `nullptr`.
using UnificationMap = mlir::DenseMap<std::pair<mlir::SymbolRefAttr, Side>, mlir::Attribute>;

/// Return `true` iff the two ArrayRef instances containing StructType or ArrayType parameters
/// are equivalent or could be equivalent after full instantiation of struct parameters.
bool typeParamsUnify(
    const mlir::ArrayRef<mlir::Attribute> &lhsParams,
    const mlir::ArrayRef<mlir::Attribute> &rhsParams, UnificationMap *unifications = nullptr
);

/// Return `true` iff the two ArrayAttr instances containing StructType or ArrayType parameters
/// are equivalent or could be equivalent after full instantiation of struct parameters.
bool typeParamsUnify(
    const mlir::ArrayAttr &lhsParams, const mlir::ArrayAttr &rhsParams,
    UnificationMap *unifications = nullptr
);

/// Return `true` iff the two ArrayType instances are equivalent or could be equivalent after full
/// instantiation of struct parameters.
bool arrayTypesUnify(
    ArrayType lhs, ArrayType rhs, mlir::ArrayRef<llvm::StringRef> rhsReversePrefix = {},
    UnificationMap *unifications = nullptr
);

/// Return `true` iff the two StructType instances are equivalent or could be equivalent after full
/// instantiation of struct parameters.
bool structTypesUnify(
    StructType lhs, StructType rhs, mlir::ArrayRef<llvm::StringRef> rhsReversePrefix = {},
    UnificationMap *unifications = nullptr
);

/// Return `true` iff the two Type instances are equivalent or could be equivalent after full
/// instantiation of struct parameters (if applicable within the given types).
bool typesUnify(
    mlir::Type lhs, mlir::Type rhs, mlir::ArrayRef<llvm::StringRef> rhsReversePrefix = {},
    UnificationMap *unifications = nullptr
);

/// Return `true` iff the two lists of Type instances are equivalent or could be equivalent after
/// full instantiation of struct parameters (if applicable within the given types).
template <typename Iter1, typename Iter2>
inline bool typeListsUnify(
    Iter1 lhs, Iter2 rhs, mlir::ArrayRef<llvm::StringRef> rhsReversePrefix = {},
    UnificationMap *unifications = nullptr
) {
  return (lhs.size() == rhs.size()) &&
         std::equal(lhs.begin(), lhs.end(), rhs.begin(), [&](mlir::Type a, mlir::Type b) {
    return typesUnify(a, b, rhsReversePrefix, unifications);
  });
}

template <typename Iter1, typename Iter2>
inline bool singletonTypeListsUnify(
    Iter1 lhs, Iter2 rhs, mlir::ArrayRef<llvm::StringRef> rhsReversePrefix = {},
    UnificationMap *unifications = nullptr
) {
  return lhs.size() == 1 && rhs.size() == 1 &&
         typesUnify(lhs.front(), rhs.front(), rhsReversePrefix, unifications);
}

/// Return `true` iff the types unify and `newTy` is "more concrete" than `oldTy`.
///
/// The types `i1`, `index`, `llzk.felt`, and `llzk.string` are concrete whereas `llzk.tvar` is not
/// (because it may be substituted with any type during struct instantiation). When considering the
/// attributes with `llzk.array` and `llzk.struct` types, we define IntegerAttr and TypeAttr as
/// concrete, AffineMapAttr as less concrete than those, and SymbolRefAttr as least concrete.
bool isMoreConcreteUnification(
    mlir::Type oldTy, mlir::Type newTy,
    llvm::function_ref<bool(mlir::Type oldTy, mlir::Type newTy)> knownOldToNew = nullptr
);

template <typename TypeClass> inline TypeClass getIfSingleton(mlir::TypeRange types) {
  return (types.size() == 1) ? llvm::dyn_cast<TypeClass>(types.front()) : nullptr;
}

template <typename TypeClass> inline TypeClass getAtIndex(mlir::TypeRange types, size_t index) {
  return (types.size() > index) ? llvm::dyn_cast<TypeClass>(types[index]) : nullptr;
}

/// Convert an IntegerAttr with a type other than IndexType to use IndexType.
mlir::IntegerAttr forceIntType(mlir::IntegerAttr attr);

/// Convert any IntegerAttr with a type other than IndexType to use IndexType.
mlir::Attribute forceIntAttrType(mlir::Attribute attr);

/// Convert any IntegerAttr with a type other than IndexType to use IndexType.
llvm::SmallVector<mlir::Attribute> forceIntAttrTypes(llvm::ArrayRef<mlir::Attribute> attrList);

/// Verify that all IntegerAttr have type IndexType.
mlir::LogicalResult verifyIntAttrType(EmitErrorFn emitError, mlir::Attribute in);

/// Verify that all AffineMapAttr only have a single result.
mlir::LogicalResult verifyAffineMapAttrType(EmitErrorFn emitError, mlir::Attribute in);

mlir::ParseResult parseAttrVec(mlir::AsmParser &parser, llvm::SmallVector<mlir::Attribute> &value);
void printAttrVec(mlir::AsmPrinter &printer, llvm::ArrayRef<mlir::Attribute> value);

mlir::ParseResult parseStructParams(mlir::AsmParser &parser, mlir::ArrayAttr &value);
void printStructParams(mlir::AsmPrinter &printer, mlir::ArrayAttr value);

mlir::LogicalResult computeDimsFromShape(
    mlir::MLIRContext *ctx, llvm::ArrayRef<int64_t> shape,
    llvm::SmallVector<mlir::Attribute> &dimensionSizes
);

mlir::LogicalResult computeShapeFromDims(
    EmitErrorFn emitError, mlir::MLIRContext *ctx, llvm::ArrayRef<mlir::Attribute> dimensionSizes,
    llvm::SmallVector<int64_t> &shape
);

mlir::ParseResult parseDerivedShape(
    mlir::AsmParser &parser, llvm::SmallVector<int64_t> &shape,
    llvm::SmallVector<mlir::Attribute> dimensionSizes
);
void printDerivedShape(
    mlir::AsmPrinter &printer, llvm::ArrayRef<int64_t> shape,
    llvm::ArrayRef<mlir::Attribute> dimensionSizes
);

} // namespace llzk
