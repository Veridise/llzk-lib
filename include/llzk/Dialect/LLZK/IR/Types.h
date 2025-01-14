#pragma once

#include "llzk/Dialect/LLZK/IR/Dialect.h"
#include "llzk/Dialect/LLZK/Util/Debug.h"
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

/// The allowed attribute types in ArrayType, StructType, and TypeVarType are IntegerAttr,
/// SymbolRefAttr, and TypeAttr. Throw a fatal error if anything else if found indicating that the
/// caller of this function should be updated.
inline void assertValidAttrForParamOfType(mlir::Attribute attr) {
  if (!llvm::isa<mlir::IntegerAttr, mlir::SymbolRefAttr, mlir::TypeAttr>(attr)) {
    llvm::report_fatal_error(
        "Legal type parameters are inconsistent. Encountered " +
        attr.getAbstractAttribute().getName()
    );
  }
}

/// valid types: {I1, Index, LLZK_FeltType, LLZK_StructType, LLZK_ArrayType, LLZK_TypeVarType}
bool isValidType(mlir::Type type);

/// valid types: isValidType() - {LLZK_StructType (including within LLZK_ArrayType)}
bool isValidEmitEqType(mlir::Type type);

/// valid types: isValidType() - {LLZK_ArrayType}
bool isValidArrayElemType(mlir::Type type);

/// Checks if the type is a LLZK Array and it also contains a valid LLZK type.
bool isValidArrayType(mlir::Type type);

inline mlir::LogicalResult
checkValidType(llvm::function_ref<mlir::InFlightDiagnostic()> emitError, mlir::Type type) {
  if (!isValidType(type)) {
    return emitError() << "expected a valid LLZK type but found " << type;
  } else {
    return mlir::success();
  }
}

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

mlir::LogicalResult computeDimsFromShape(
    mlir::MLIRContext *ctx, llvm::ArrayRef<int64_t> shape,
    llvm::SmallVector<mlir::Attribute> &dimensionSizes
);

mlir::LogicalResult computeShapeFromDims(
    llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, mlir::MLIRContext *ctx,
    llvm::ArrayRef<mlir::Attribute> dimensionSizes, llvm::SmallVector<int64_t> &shape
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
