#pragma once

#include "llzk/Dialect/LLZK/IR/Dialect.h"
#include "llzk/Dialect/LLZK/Util/ErrorHelper.h"
#include "llzk/Dialect/LLZK/Util/SymbolLookup.h" // IWYU pragma: keep

#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/IR/Types.h>

#include <llvm/ADT/ArrayRef.h>
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
/// reverse back into a Type.
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
inline std::string shortString(mlir::Type type) {
  return ShortTypeStringifier().append(type).str();
}

/// Return a brief string representation of the attribute list from a parameterized type.
inline std::string shortString(mlir::ArrayRef<mlir::Attribute> attrs) {
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

/// Return `true` iff the two ArrayRef instances containing StructType or ArrayType parameters
/// are equivalent or could be equivalent after full instantiation of struct parameters.
bool typeParamsUnify(
    const mlir::ArrayRef<mlir::Attribute> &lhsParams,
    const mlir::ArrayRef<mlir::Attribute> &rhsParams
);

/// Return `true` iff the two ArrayAttr instances containing StructType or ArrayType parameters
/// are equivalent or could be equivalent after full instantiation of struct parameters.
bool typeParamsUnify(const mlir::ArrayAttr &lhsParams, const mlir::ArrayAttr &rhsParams);

/// Return `true` iff the two ArrayType instances are equivalent or could be equivalent after full
/// instantiation of struct parameters.
bool arrayTypesUnify(
    ArrayType lhs, ArrayType rhs, mlir::ArrayRef<llvm::StringRef> rhsRevPrefix = {}
);

/// Return `true` iff the two StructType instances are equivalent or could be equivalent after full
/// instantiation of struct parameters.
bool structTypesUnify(
    StructType lhs, StructType rhs, mlir::ArrayRef<llvm::StringRef> rhsRevPrefix = {}
);

/// Return `true` iff the two Type instances are equivalent or could be equivalent after full
/// instantiation of struct parameters (if applicable within the given types).
bool typesUnify(mlir::Type lhs, mlir::Type rhs, mlir::ArrayRef<llvm::StringRef> rhsRevPrefix = {});

/// Return `true` iff the two lists of Type instances are equivalent or could be equivalent after
/// full instantiation of struct parameters (if applicable within the given types).
template <typename Iter1, typename Iter2>
inline bool
typeListsUnify(Iter1 lhs, Iter2 rhs, mlir::ArrayRef<llvm::StringRef> rhsRevPrefix = {}) {
  return (lhs.size() == rhs.size()) &&
         std::equal(
             lhs.begin(), lhs.end(), rhs.begin(),
             [&rhsRevPrefix](mlir::Type a, mlir::Type b) { return typesUnify(a, b, rhsRevPrefix); }
         );
}

template <typename Iter1, typename Iter2>
inline bool
singletonTypeListsUnify(Iter1 lhs, Iter2 rhs, mlir::ArrayRef<llvm::StringRef> rhsRevPrefix = {}) {
  return lhs.size() == 1 && rhs.size() == 1 && typesUnify(lhs.front(), rhs.front());
}

template <typename ConcreteType> inline ConcreteType getIfSingleton(mlir::TypeRange types) {
  return (types.size() == 1) ? llvm::dyn_cast<ConcreteType>(types.front()) : nullptr;
}

template <typename ConcreteType>
inline ConcreteType getAtIndex(mlir::TypeRange types, size_t index) {
  return (types.size() > index) ? llvm::dyn_cast<ConcreteType>(types[index]) : nullptr;
}

mlir::LogicalResult computeDimsFromShape(
    mlir::MLIRContext *ctx, llvm::ArrayRef<int64_t> shape,
    llvm::SmallVector<mlir::Attribute> &dimensionSizes
);

mlir::LogicalResult computeShapeFromDims(
    EmitErrorFn emitError, mlir::MLIRContext *ctx, llvm::ArrayRef<mlir::Attribute> dimensionSizes,
    llvm::SmallVector<int64_t> &shape
);

mlir::ParseResult parseAttrVec(mlir::AsmParser &parser, llvm::SmallVector<mlir::Attribute> &value);
void printAttrVec(mlir::AsmPrinter &printer, llvm::ArrayRef<mlir::Attribute> value);

mlir::ParseResult parseDerivedShape(
    mlir::AsmParser &parser, llvm::SmallVector<int64_t> &shape,
    llvm::SmallVector<mlir::Attribute> dimensionSizes
);
void printDerivedShape(
    mlir::AsmPrinter &printer, llvm::ArrayRef<int64_t> shape,
    llvm::ArrayRef<mlir::Attribute> dimensionSizes
);

} // namespace llzk
