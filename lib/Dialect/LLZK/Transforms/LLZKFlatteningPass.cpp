//===-- LLZKFlatteningPass.cpp - Implements -llzk-flatten pass --*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the `-llzk-flatten` pass.
///
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/LLZK/IR/Dialect.h"
#include "llzk/Dialect/LLZK/IR/Ops.h"
#include "llzk/Dialect/LLZK/Transforms/LLZKTransformationPasses.h"
#include "llzk/Dialect/LLZK/Util/AttributeHelper.h"
#include "llzk/Dialect/LLZK/Util/Debug.h"
#include "llzk/Dialect/LLZK/Util/SymbolHelper.h"
#include <llzk/Dialect/LLZK/IR/Attrs.h>

#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Affine/LoopUtils.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/SCF/Transforms/Patterns.h>
#include <mlir/Dialect/SCF/Utils/Utils.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Interfaces/InferTypeOpInterface.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <llvm/ADT/APInt.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DepthFirstIterator.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/Debug.h>

// Include the generated base pass class definitions.
namespace llzk {
#define GEN_PASS_DEF_FLATTENINGPASS
#include "llzk/Dialect/LLZK/Transforms/LLZKTransformationPasses.h.inc"
} // namespace llzk

using namespace llzk;
using namespace mlir;

#define DEBUG_TYPE "llzk-flatten"

namespace {

class ConversionTracker {
  /// Tracks if some step performed a modification of the code such that another pass should be run.
  bool modified;
  /// Maps original remote (i.e. use site) type to new remote type.
  /// Note: The keys are always parameterized StructType and the values are no-parameter StructType.
  DenseMap<StructType, StructType> structInstantiations;
  /// Contains the reverse of mappings in `structInstantiations` for use in legal conversion check.
  DenseMap<StructType, StructType> reverseInstantiations;
  /// Maps new remote type (i.e. the values in 'structInstantiations') to a list of Diagnostic
  /// to report at the location(s) of the compute() that causes the instantiation to the StructType.
  DenseMap<StructType, SmallVector<Diagnostic>> delayedDiagnostics;

public:
  bool isModified() const { return modified; }
  void resetModifiedFlag() { modified = false; }
  void updateModifiedFlag(bool currStepModified) { modified |= currStepModified; }

  void recordInstantiation(StructType oldType, StructType newType) {
    // Assert invariant required by `structInstantiations`
    assert(!isNullOrEmpty(oldType.getParams()));
    assert(isNullOrEmpty(newType.getParams()));

    auto forwardResult = structInstantiations.try_emplace(oldType, newType);
    if (forwardResult.second) {
      // Insertion was successful
      // ASSERT: The reverse map does not contain this mapping either
      assert(!reverseInstantiations.contains(newType));
      reverseInstantiations[newType] = oldType;
      // Set the modified flag
      modified = true;
    } else {
      // ASSERT: If a mapping already existed for `oldType` it must be `newType`
      assert(forwardResult.first->getSecond() == newType);
      // ASSERT: The reverse mapping is already present as well
      assert(reverseInstantiations.lookup(newType) == oldType);
    }
    assert(structInstantiations.size() == reverseInstantiations.size());
  }

  /// Return the instantiated type of the given StructType, if any.
  std::optional<StructType> getInstantiation(StructType oldType) const {
    auto cachedResult = structInstantiations.find(oldType);
    if (cachedResult != structInstantiations.end()) {
      return cachedResult->second;
    }
    return std::nullopt;
  }

  /// Collect the fully-qualified names of all structs that were instantiated.
  DenseSet<SymbolRefAttr> getInstantiatedStructNames() const {
    DenseSet<SymbolRefAttr> instantiatedNames;
    for (const auto &[origRemoteTy, _] : structInstantiations) {
      instantiatedNames.insert(origRemoteTy.getNameRef());
    }
    return instantiatedNames;
  }

  void reportDelayedDiagnostics(StructType newType, CallOp caller) {
    auto res = delayedDiagnostics.find(newType);
    if (res == delayedDiagnostics.end()) {
      return;
    }

    DiagnosticEngine &engine = caller.getContext()->getDiagEngine();
    for (Diagnostic &diag : res->second) {
      // Update any notes referencing an UnknownLoc to use the CallOp location.
      for (Diagnostic &note : diag.getNotes()) {
        assert(note.getNotes().empty() && "notes cannot have notes attached");
        if (llvm::isa<UnknownLoc>(note.getLocation())) {
          note = std::move(Diagnostic(caller.getLoc(), note.getSeverity()).append(note.str()));
        }
      }
      // Report. Based on InFlightDiagnostic::report().
      engine.emit(std::move(diag));
    }
    // Emiting a Diagnostic consumes it (per DiagnosticEngine::emit) so remove them from the map.
    // Unfortunately, this means if the key StructType is the result of instantiation at multiple
    // `compute()` calls it will only be reported at one of those locations, not all.
    delayedDiagnostics.erase(newType);
  }

  SmallVector<Diagnostic> &delayedDiagnosticSet(StructType newType) {
    return delayedDiagnostics[newType];
  }

  /// Check if the type conversion is legal, i.e. the new type unifies with and is more concrete
  /// than the old type with additional allowance for the results of struct flattening conversions.
  bool isLegalConversion(Type oldType, Type newType, const char *patName) const {
    std::function<bool(Type, Type)> checkInstantiations = [&](Type oTy, Type nTy) {
      // Check if `oTy` is a struct with a known instantiation to `nTy`
      if (StructType oldStructType = llvm::dyn_cast<StructType>(oTy)) {
        // Note: The values in `structInstantiations` must be no-parameter struct types
        // so there is no need for recursive check, simple equality is sufficient.
        if (this->structInstantiations.lookup(oldStructType) == nTy) {
          return true;
        }
      }
      // Check if `nTy` is the result of a struct instantiation and if the pre-image of
      // that instantiation (i.e. the parameterized version of the instantiated struct)
      // is a more concrete unification of `oTy`.
      if (StructType newStructType = llvm::dyn_cast<StructType>(nTy)) {
        if (auto preImage = this->reverseInstantiations.lookup(newStructType)) {
          if (isMoreConcreteUnification(oTy, preImage, checkInstantiations)) {
            return true;
          }
        }
      }
      return false;
    };

    if (isMoreConcreteUnification(oldType, newType, checkInstantiations)) {
      return true;
    }
    LLVM_DEBUG(llvm::dbgs() << "[" << patName << "] Cannot replace old type " << oldType
                            << " with new type " << newType
                            << " because it does not define a compatible and more concrete type.\n";
    );
    return false;
  }

  template <typename T, typename U>
  inline bool areLegalConversions(T oldTypes, U newTypes, const char *patName) const {
    return llvm::all_of(
        llvm::zip_equal(oldTypes, newTypes),
        [this, &patName](std::tuple<Type, Type> oldThenNew) {
      return this->isLegalConversion(std::get<0>(oldThenNew), std::get<1>(oldThenNew), patName);
    }
    );
  }
};

/// Patterns can use this listener and call notifyMatchFailure(..) for failures where the entire
/// pass must fail, i.e. where instantiation would introduce an illegal type conversion.
struct MatchFailureListener : public RewriterBase::Listener {
  bool hadFailure = false;

  ~MatchFailureListener() override {}

  LogicalResult
  notifyMatchFailure(Location loc, function_ref<void(Diagnostic &)> reasonCallback) override {
    hadFailure = true;

    InFlightDiagnostic diag = emitError(loc);
    reasonCallback(*diag.getUnderlyingDiagnostic());
    return diag; // implicitly calls `diag.report()`
  }
};

static LogicalResult
applyAndFoldGreedily(ModuleOp modOp, ConversionTracker &tracker, RewritePatternSet &&patterns) {
  bool currStepModified = false;
  MatchFailureListener failureListener;
  LogicalResult result = applyPatternsAndFoldGreedily(
      modOp->getRegion(0), std::move(patterns), GreedyRewriteConfig {.listener = &failureListener},
      &currStepModified
  );
  tracker.updateModifiedFlag(currStepModified);
  return failure(result.failed() || failureListener.hadFailure);
}

/// Wrapper for PatternRewriter.replaceOpWithNewOp() that automatically copies discardable
/// attributes (i.e. attributes other than those specifically defined as part of the Op in ODS).
template <typename OpTy, typename Rewriter, typename... Args>
inline OpTy replaceOpWithNewOp(Rewriter &rewriter, Operation *op, Args &&...args) {
  DictionaryAttr attrs = op->getDiscardableAttrDictionary();
  OpTy newOp = rewriter.template replaceOpWithNewOp<OpTy>(op, std::forward<Args>(args)...);
  newOp->setDiscardableAttrs(attrs);
  return newOp;
}

/// Lists all Op classes that may contain a StructType in their results or attributes.
static struct {
  /// Subset that define the general builder function:
  /// `build(OpBuilder&, OperationState&, TypeRange, ValueRange, ArrayRef<NamedAttribute>)`
  const std::tuple<
      FieldDefOp, FieldWriteOp, FieldReadOp, CreateStructOp, FuncOp, ReturnOp, InsertArrayOp,
      ExtractArrayOp, ReadArrayOp, WriteArrayOp, EmitContainmentOp>
      WithGeneralBuilder {};
  /// Subset that do NOT define the general builder function. These cannot use
  /// `GeneralTypeReplacePattern` and must have a `OpConversionPattern` defined if
  /// they need to be converted.
  const std::tuple<CallOp, CreateArrayOp> NoGeneralBuilder {};
} OpClassesWithStructTypes;

// NOTE: This pattern will produce a compile error if `OpTy` does not define the general
// `build(OpBuilder&, OperationState&, TypeRange, ValueRange, ArrayRef<NamedAttribute>)` function
// because that function is required by the `replaceOpWithNewOp()` call.
template <typename OpTy> class GeneralTypeReplacePattern : public OpConversionPattern<OpTy> {
public:
  using OpConversionPattern<OpTy>::OpConversionPattern;

  LogicalResult matchAndRewrite(OpTy op, OpTy::Adaptor adaptor, ConversionPatternRewriter &rewriter)
      const override {
    const TypeConverter *converter = OpConversionPattern<OpTy>::getTypeConverter();
    assert(converter);
    // Convert result types
    SmallVector<Type> newResultTypes;
    if (failed(converter->convertTypes(op->getResultTypes(), newResultTypes))) {
      return op->emitError("Could not convert Op result types.");
    }
    // ASSERT: 'adaptor.getAttributes()' is empty or subset of 'op->getAttrDictionary()' so the
    // former can be ignored without losing anything.
    assert(
        adaptor.getAttributes().empty() ||
        llvm::all_of(
            adaptor.getAttributes(),
            [d = op->getAttrDictionary()](NamedAttribute a) { return d.contains(a.getName()); }
        )
    );
    // Convert any TypeAttr in the attribute list.
    SmallVector<NamedAttribute> newAttrs(op->getAttrDictionary().getValue());
    for (NamedAttribute &n : newAttrs) {
      if (TypeAttr t = llvm::dyn_cast<TypeAttr>(n.getValue())) {
        if (Type newType = converter->convertType(t.getValue())) {
          n.setValue(TypeAttr::get(newType));
        } else {
          return op->emitError().append("Could not convert type in attribute: ", t);
        }
      }
    }
    // Build a new Op in place of the current one
    replaceOpWithNewOp<OpTy>(
        rewriter, op, TypeRange(newResultTypes), adaptor.getOperands(), ArrayRef(newAttrs)
    );
    return success();
  }
};

class CreateArrayOpTypeReplacePattern : public OpConversionPattern<CreateArrayOp> {
public:
  using OpConversionPattern<CreateArrayOp>::OpConversionPattern;

  LogicalResult match(CreateArrayOp op) const override {
    if (Type newType = getTypeConverter()->convertType(op.getType())) {
      return success();
    } else {
      return op->emitError("Could not convert Op result type.");
    }
  }

  void
  rewrite(CreateArrayOp op, OpAdaptor adapter, ConversionPatternRewriter &rewriter) const override {
    Type newType = getTypeConverter()->convertType(op.getType());
    assert(llvm::isa<ArrayType>(newType) && "CreateArrayOp must produce ArrayType result");
    DenseI32ArrayAttr numDimsPerMap = op.getNumDimsPerMapAttr();
    if (isNullOrEmpty(numDimsPerMap)) {
      replaceOpWithNewOp<CreateArrayOp>(
          rewriter, op, llvm::cast<ArrayType>(newType), adapter.getElements()
      );
    } else {
      replaceOpWithNewOp<CreateArrayOp>(
          rewriter, op, llvm::cast<ArrayType>(newType), adapter.getMapOperands(), numDimsPerMap
      );
    }
  }
};

template <typename I, typename NextOpType, typename... OtherOpTypes>
inline void applyToMoreTypes(I inserter) {
  std::apply(inserter, std::tuple<NextOpType, OtherOpTypes...> {});
}
template <typename I> inline void applyToMoreTypes(I inserter) {}

/// Return a new `RewritePatternSet` that includes a `GeneralTypeReplacePattern` for all of
/// `OpClassesWithStructTypes.WithGeneralBuilder` and `AdditionalOpTypes`.
/// Note: `GeneralTypeReplacePattern` uses the default benefit (1) so additional patterns with a
/// higher priority can be added for any of the Ops already included and that will take precedence.
template <typename... AdditionalOpTypes>
RewritePatternSet
newGeneralRewritePatternSet(TypeConverter &tyConv, MLIRContext *ctx, ConversionTarget &target) {
  RewritePatternSet patterns(ctx);
  auto inserter = [&](auto... opClasses) {
    patterns.add<GeneralTypeReplacePattern<decltype(opClasses)>...>(tyConv, ctx);
  };
  std::apply(inserter, OpClassesWithStructTypes.WithGeneralBuilder);
  applyToMoreTypes<decltype(inserter), AdditionalOpTypes...>(inserter);
  // Special case for CreateArrayOp since GeneralTypeReplacePattern does not work
  patterns.add<CreateArrayOpTypeReplacePattern>(tyConv, ctx);
  // Add builtin FunctionType converter
  populateFunctionOpInterfaceTypeConversionPattern<FuncOp>(patterns, tyConv);
  scf::populateSCFStructuralTypeConversionsAndLegality(tyConv, patterns, target);
  return patterns;
}

/// Return a new `ConversionTarget` allowing all LLZK-required dialects.
ConversionTarget newBaseTarget(MLIRContext *ctx) {
  ConversionTarget target(*ctx);
  target.addLegalDialect<LLZKDialect, arith::ArithDialect, scf::SCFDialect>();
  target.addLegalOp<ModuleOp>();
  return target;
}

static bool defaultLegalityCheck(const TypeConverter &tyConv, Operation *op) {
  // Check operand types and result types
  if (!tyConv.isLegal(op)) {
    return false;
  }
  // Check type attributes
  for (NamedAttribute n : op->getAttrDictionary().getValue()) {
    if (TypeAttr tyAttr = llvm::dyn_cast<TypeAttr>(n.getValue())) {
      Type t = tyAttr.getValue();
      if (FunctionType funcTy = llvm::dyn_cast<FunctionType>(t)) {
        if (!tyConv.isSignatureLegal(funcTy)) {
          return false;
        }
      } else {
        if (!tyConv.isLegal(t)) {
          return false;
        }
      }
    }
  }
  return true;
}

// Default to true if the check is not for that particular operation type.
template <typename Check> bool runCheck(Operation *op, Check check) {
  if (auto specificOp =
          mlir::dyn_cast_if_present<typename llvm::function_traits<Check>::template arg_t<0>>(op)) {
    return check(specificOp);
  }
  return true;
}

/// Return a new `ConversionTarget` allowing all LLZK-required dialects and defining Op legality
/// based on the given `TypeConverter` for Ops listed in both fields of `OpClassesWithStructTypes`
/// and in `AdditionalOpTypes`.
/// Additional legality checks can be included for certain ops that will run along with the default
/// check. For an op to be considered legal all checks (default plus additional checks if any) must
/// return true.
///
template <typename... AdditionalOpTypes, typename... AdditionalChecks>
ConversionTarget
newConverterDefinedTarget(TypeConverter &tyConv, MLIRContext *ctx, AdditionalChecks &&...checks) {
  ConversionTarget target = newBaseTarget(ctx);
  auto inserter = [&](auto... opClasses) {
    target.addDynamicallyLegalOp<decltype(opClasses)...>([&tyConv, &checks...](Operation *op) {
      return defaultLegalityCheck(tyConv, op) && (runCheck<AdditionalChecks>(op, checks) && ...);
    });
  };
  std::apply(inserter, OpClassesWithStructTypes.NoGeneralBuilder);
  std::apply(inserter, OpClassesWithStructTypes.WithGeneralBuilder);
  applyToMoreTypes<decltype(inserter), AdditionalOpTypes...>(inserter);
  return target;
}

template <bool AllowStructParams = true> bool isConcreteAttr(Attribute a) {
  if (TypeAttr tyAttr = dyn_cast<TypeAttr>(a)) {
    return isConcreteType(tyAttr.getValue(), AllowStructParams);
  }
  return llvm::isa<IntegerAttr>(a);
}

namespace Step1_InstantiateStructs {

static inline bool tableOffsetIsntSymbol(FieldReadOp op) {
  return !mlir::isa_and_present<SymbolRefAttr>(op.getTableOffset().value_or(nullptr));
}

class StructCloner {
  ConversionTracker &tracker_;
  ModuleOp rootMod;
  SymbolTableCollection symTables;

  class MappedTypeConverter : public TypeConverter {
    StructType origTy;
    StructType newTy;
    const DenseMap<Attribute, Attribute> &paramNameToValue;

  public:
    MappedTypeConverter(
        StructType originalType, StructType newType,
        /// Instantiated values for the parameter names in `originalType`
        const DenseMap<Attribute, Attribute> &paramNameToInstantiatedValue
    )
        : TypeConverter(), origTy(originalType), newTy(newType),
          paramNameToValue(paramNameToInstantiatedValue) {

      addConversion([](Type inputTy) { return inputTy; });

      addConversion([this](StructType inputTy) {
        // Check for replacement of the full type
        if (inputTy == this->origTy) {
          return this->newTy;
        }
        // Check for replacement of parameter symbol names with concrete values
        if (ArrayAttr inputTyParams = inputTy.getParams()) {
          SmallVector<Attribute> updated;
          for (Attribute a : inputTyParams) {
            auto res = this->paramNameToValue.find(a);
            updated.push_back((res != this->paramNameToValue.end()) ? res->second : a);
          }
          return StructType::get(
              inputTy.getNameRef(), ArrayAttr::get(inputTy.getContext(), updated)
          );
        }
        // Otherwise, return the type unchanged
        return inputTy;
      });

      addConversion([this](ArrayType inputTy) {
        // Check for replacement of parameter symbol names with concrete values
        ArrayRef<Attribute> dimSizes = inputTy.getDimensionSizes();
        if (!dimSizes.empty()) {
          SmallVector<Attribute> updated;
          for (Attribute a : dimSizes) {
            auto res = this->paramNameToValue.find(a);
            updated.push_back((res != this->paramNameToValue.end()) ? res->second : a);
          }
          return ArrayType::get(this->convertType(inputTy.getElementType()), updated);
        }
        // Otherwise, return the type unchanged
        return inputTy;
      });

      addConversion([this](TypeVarType inputTy) -> Type {
        // Check for replacement of parameter symbol name with a concrete type
        auto res = this->paramNameToValue.find(inputTy.getNameRef());
        if (res != this->paramNameToValue.end()) {
          if (TypeAttr tyAttr = llvm::dyn_cast<TypeAttr>(res->second)) {
            return tyAttr.getValue();
          }
        }
        return inputTy;
      });
    }
  };

  class ClonedStructCallOpPattern : public OpConversionPattern<CallOp> {
  public:
    ClonedStructCallOpPattern(TypeConverter &converter, MLIRContext *ctx)
        // future proof: use higher priority than GeneralTypeReplacePattern
        : OpConversionPattern<CallOp>(converter, ctx, 2) {}

    LogicalResult matchAndRewrite(CallOp op, OpAdaptor adapter, ConversionPatternRewriter &rewriter)
        const override {
      // Convert the result types of the CallOp
      SmallVector<Type> newResultTypes;
      if (failed(getTypeConverter()->convertTypes(op.getResultTypes(), newResultTypes))) {
        return op->emitError("Could not convert Op result types.");
      }
      replaceOpWithNewOp<CallOp>(
          rewriter, op, newResultTypes, op.getCalleeAttr(), adapter.getMapOperands(),
          op.getNumDimsPerMapAttr(), adapter.getArgOperands()
      );
      return success();
    }
  };

  template <typename Impl, typename Op, typename... HandledAttrs>
  class SymbolUserHelper : public OpConversionPattern<Op> {
  private:
    const DenseMap<Attribute, Attribute> &paramNameToValue;

    SymbolUserHelper(
        TypeConverter &converter, MLIRContext *ctx, unsigned Benefit,
        const DenseMap<Attribute, Attribute> &paramNameToInstantiatedValue
    )
        : OpConversionPattern<Op>(converter, ctx, Benefit),
          paramNameToValue(paramNameToInstantiatedValue) {}

  public:
    using OpAdaptor = typename mlir::OpConversionPattern<Op>::OpAdaptor;

    virtual Attribute getNameAttr(Op) const = 0;

    virtual LogicalResult handleDefaultRewrite(
        Attribute, Op op, OpAdaptor, ConversionPatternRewriter &, Attribute a
    ) const {
      return op->emitOpError().append("expected value with type ", op.getType(), " but found ", a);
    }

    LogicalResult
    matchAndRewrite(Op op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
      auto res = this->paramNameToValue.find(getNameAttr(op));
      if (res == this->paramNameToValue.end()) {
        return op->emitOpError("missing instantiation");
      }
      llvm::TypeSwitch<Attribute, LogicalResult> TS(res->second);
      llvm::TypeSwitch<Attribute, LogicalResult> *ptr = &TS;

      ((ptr = &(ptr->template Case<HandledAttrs>([&](HandledAttrs a) {
        return static_cast<const Impl *>(this)->handleRewrite(res->first, op, adaptor, rewriter, a);
      }))),
       ...);

      return TS.Default([&](Attribute a) {
        return handleDefaultRewrite(res->first, op, adaptor, rewriter, a);
      });
    }
    friend Impl;
  };

  class ClonedStructConstReadOpPattern
      : public SymbolUserHelper<
            ClonedStructConstReadOpPattern, ConstReadOp, IntegerAttr, FeltConstAttr> {
    SmallVector<Diagnostic> &diagnostics;

    using super =
        SymbolUserHelper<ClonedStructConstReadOpPattern, ConstReadOp, IntegerAttr, FeltConstAttr>;

  public:
    ClonedStructConstReadOpPattern(
        TypeConverter &converter, MLIRContext *ctx,
        const DenseMap<Attribute, Attribute> &paramNameToInstantiatedValue,
        SmallVector<Diagnostic> &instantiationDiagnostics
    )
        // future proof: use higher priority than GeneralTypeReplacePattern
        : super(converter, ctx, /*Benefit=*/2, paramNameToInstantiatedValue),
          diagnostics(instantiationDiagnostics) {}

    Attribute getNameAttr(ConstReadOp op) const override { return op.getConstNameAttr(); }

    LogicalResult handleRewrite(
        Attribute sym, ConstReadOp op, OpAdaptor, ConversionPatternRewriter &rewriter, IntegerAttr a
    ) const {
      APInt attrValue = a.getValue();
      Type origResTy = op.getType();
      if (llvm::isa<FeltType>(origResTy)) {
        replaceOpWithNewOp<FeltConstantOp>(
            rewriter, op, FeltConstAttr::get(getContext(), attrValue)
        );
        return success();
      }

      if (llvm::isa<IndexType>(origResTy)) {
        replaceOpWithNewOp<arith::ConstantIndexOp>(rewriter, op, fromAPInt(attrValue));
        return success();
      }

      if (origResTy.isSignlessInteger(1)) {
        // Treat 0 as false and any other value as true (but give a warning if it's not 1)
        if (attrValue.isZero()) {
          replaceOpWithNewOp<arith::ConstantIntOp>(rewriter, op, false, origResTy);
          return success();
        }
        if (!attrValue.isOne()) {
          Location opLoc = op.getLoc();
          Diagnostic diag(opLoc, DiagnosticSeverity::Warning);
          diag << "Interpretting non-zero value " << stringWithoutType(a) << " as true";
          if (getContext()->shouldPrintOpOnDiagnostic()) {
            diag.attachNote(opLoc) << "see current operation: " << *op;
          }
          diag.attachNote(UnknownLoc::get(getContext()))
              << "when instantiating " << StructDefOp::getOperationName() << " parameter \"" << sym
              << "\" for this call";
          diagnostics.push_back(std::move(diag));
        }
        replaceOpWithNewOp<arith::ConstantIntOp>(rewriter, op, true, origResTy);
        return success();
      }
      return LogicalResult(op->emitOpError().append("unexpected result type ", origResTy));
    }

    LogicalResult handleRewrite(
        Attribute, ConstReadOp op, OpAdaptor, ConversionPatternRewriter &rewriter, FeltConstAttr a
    ) const {
      replaceOpWithNewOp<FeltConstantOp>(rewriter, op, a);
      return success();
    }
  };

  class ClonedStructFieldReadOpPattern
      : public SymbolUserHelper<
            ClonedStructFieldReadOpPattern, FieldReadOp, IntegerAttr, FeltConstAttr> {
    using super =
        SymbolUserHelper<ClonedStructFieldReadOpPattern, FieldReadOp, IntegerAttr, FeltConstAttr>;

  public:
    ClonedStructFieldReadOpPattern(
        TypeConverter &converter, MLIRContext *ctx,
        const DenseMap<Attribute, Attribute> &paramNameToInstantiatedValue
    )
        // future proof: use higher priority than GeneralTypeReplacePattern
        : super(converter, ctx, /*Benefit=*/3, paramNameToInstantiatedValue) {}

    Attribute getNameAttr(FieldReadOp op) const override {
      return op.getTableOffset().value_or(nullptr);
    }

    template <typename Attr>
    LogicalResult handleRewrite(
        Attribute, FieldReadOp op, OpAdaptor, ConversionPatternRewriter &rewriter, Attr a
    ) const {
      rewriter.modifyOpInPlace(op, [&]() {
        op.setTableOffsetAttr(rewriter.getIndexAttr(fromAPInt(a.getValue())));
      });

      return success();
    }

    LogicalResult matchAndRewrite(
        FieldReadOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
    ) const override {
      if (tableOffsetIsntSymbol(op)) {
        return failure();
      }

      return super::matchAndRewrite(op, adaptor, rewriter);
    }
  };

  static inline DenseMap<Attribute, Attribute>
  buildNameToValueMap(ArrayAttr paramNames, ArrayRef<Attribute> paramInstantiations) {
    // pre-conditions
    assert(!isNullOrEmpty(paramNames));
    assert(paramNames.size() == paramInstantiations.size());
    // Map parameter names to instantiated values
    DenseMap<Attribute, Attribute> ret;
    for (size_t i = 0, e = paramNames.size(); i < e; ++i) {
      ret[paramNames[i]] = paramInstantiations[i];
    }
    return ret;
  }

  FailureOr<StructType> genClone(StructType typeAtCaller, ArrayRef<Attribute> typeAtCallerParams) {
    // Find the StructDefOp for the original StructType
    FailureOr<SymbolLookupResult<StructDefOp>> r = typeAtCaller.getDefinition(symTables, rootMod);
    if (failed(r)) {
      return failure(); // getDefinition() already emits a sufficient error message
    }
    StructDefOp origStruct = r->get();

    // Clone the original struct, apply the new name, and remove the parameters.
    StructDefOp newStruct = origStruct.clone();
    newStruct.setSymName(
        typeAtCaller.getNameRef().getLeafReference().str() + "_" + shortString(typeAtCallerParams)
    );
    newStruct.setConstParamsAttr(ArrayAttr {});

    // Insert 'newStruct' into the parent ModuleOp of the original StructDefOp. Use the
    // `SymbolTable::insert()` function directly so that the name will be made unique.
    ModuleOp parentModule = llvm::cast<ModuleOp>(origStruct.getParentOp());
    symTables.getSymbolTable(parentModule).insert(newStruct, Block::iterator(origStruct));
    // Retrieve the new type AFTER inserting since the name may be appended to make it unique.
    StructType newRemoteType = newStruct.getType();

    // Within the new struct, replace all references to the original struct's type (i.e. the
    // locally-parameterized version) with the new flattened (i.e. no parameters) struct's type,
    // and replace all uses of the struct parameters with the concrete values.
    MLIRContext *ctx = rootMod.getContext();
    StructType typeAtDef = origStruct.getType();
    DenseMap<Attribute, Attribute> nameToValueMap =
        buildNameToValueMap(typeAtDef.getParams(), typeAtCallerParams);
    MappedTypeConverter tyConv(typeAtDef, newRemoteType, nameToValueMap);
    ConversionTarget target =
        newConverterDefinedTarget<EmitEqualityOp>(tyConv, ctx, tableOffsetIsntSymbol);

    target.addIllegalOp<ConstReadOp>();

    RewritePatternSet patterns = newGeneralRewritePatternSet<EmitEqualityOp>(tyConv, ctx, target);
    patterns.add<ClonedStructCallOpPattern>(tyConv, ctx);
    patterns.add<ClonedStructConstReadOpPattern>(
        tyConv, ctx, nameToValueMap, tracker_.delayedDiagnosticSet(newRemoteType)
    );
    patterns.add<ClonedStructFieldReadOpPattern>(tyConv, ctx, nameToValueMap);
    if (failed(applyFullConversion(newStruct, target, std::move(patterns)))) {
      return failure();
    }
    return newRemoteType;
  }

public:
  StructCloner(ConversionTracker &tracker, ModuleOp root)
      : tracker_(tracker), rootMod(root), symTables() {}

  FailureOr<StructType> createInstantiatedClone(StructType orig) {
    if (ArrayAttr params = orig.getParams()) {
      // If all parameters are concrete values (Integer or Type), then replace with a
      // no-parameter StructType referencing the de-parameterized struct.
      if (llvm::all_of(params, isConcreteAttr<>)) {
        FailureOr<StructType> res = genClone(orig, params.getValue());
        if (succeeded(res)) {
          return res.value();
        }
      }
    }
    return failure();
  }
};

class ParameterizedStructUseTypeConverter : public TypeConverter {
  ConversionTracker &tracker_;
  StructCloner cloner;

public:
  ParameterizedStructUseTypeConverter(ConversionTracker &tracker, ModuleOp root)
      : TypeConverter(), tracker_(tracker), cloner(tracker, root) {

    addConversion([](Type inputTy) { return inputTy; });

    addConversion([this](StructType inputTy) -> StructType {
      // First check for a cached entry
      if (auto opt = tracker_.getInstantiation(inputTy)) {
        return opt.value();
      }

      // Otherwise, try to create a clone of the struct with instantiated params
      FailureOr<StructType> cloneRes = cloner.createInstantiatedClone(inputTy);
      if (failed(cloneRes)) {
        return inputTy;
      }
      StructType newTy = cloneRes.value();
      LLVM_DEBUG(
          llvm::dbgs() << "[ParameterizedStructUseTypeConverter] instantiating " << inputTy
                       << " as " << newTy << '\n'
      );
      tracker_.recordInstantiation(inputTy, newTy);
      return newTy;
    });
  }
};

class CallStructFuncPattern : public OpConversionPattern<CallOp> {
  ConversionTracker &tracker_;

public:
  CallStructFuncPattern(TypeConverter &converter, MLIRContext *ctx, ConversionTracker &tracker)
      // future proof: use higher priority than GeneralTypeReplacePattern
      : OpConversionPattern<CallOp>(converter, ctx, 2), tracker_(tracker) {}

  LogicalResult matchAndRewrite(CallOp op, OpAdaptor adapter, ConversionPatternRewriter &rewriter)
      const override {
    // Convert the result types of the CallOp
    SmallVector<Type> newResultTypes;
    if (failed(getTypeConverter()->convertTypes(op.getResultTypes(), newResultTypes))) {
      return op->emitError("Could not convert Op result types.");
    }

    // Update the callee to reflect the new struct target if necessary. These checks are based on
    // `CallOp::calleeIsStructC*()` but the types must not come from the CallOp in this case.
    // Instead they must come from the converted versions.
    SymbolRefAttr calleeAttr = op.getCalleeAttr();
    if (op.calleeIsStructCompute()) {
      if (StructType newStTy = getIfSingleton<StructType>(newResultTypes)) {
        assert(isNullOrEmpty(newStTy.getParams()) && "must be fully instantiated");
        calleeAttr = appendLeaf(newStTy.getNameRef(), calleeAttr.getLeafReference());
        tracker_.reportDelayedDiagnostics(newStTy, op);
      }
    } else if (op.calleeIsStructConstrain()) {
      if (StructType newStTy = getAtIndex<StructType>(adapter.getArgOperands().getTypes(), 0)) {
        assert(isNullOrEmpty(newStTy.getParams()) && "must be fully instantiated");
        calleeAttr = appendLeaf(newStTy.getNameRef(), calleeAttr.getLeafReference());
      }
    }
    replaceOpWithNewOp<CallOp>(
        rewriter, op, newResultTypes, calleeAttr, adapter.getMapOperands(),
        op.getNumDimsPerMapAttr(), adapter.getArgOperands()
    );
    return success();
  }
};

LogicalResult run(ModuleOp modOp, ConversionTracker &tracker) {
  MLIRContext *ctx = modOp.getContext();
  ParameterizedStructUseTypeConverter tyConv(tracker, modOp);
  ConversionTarget target = newConverterDefinedTarget<>(tyConv, ctx);
  RewritePatternSet patterns = newGeneralRewritePatternSet(tyConv, ctx, target);
  patterns.add<CallStructFuncPattern>(tyConv, ctx, tracker);
  return applyPartialConversion(modOp, target, std::move(patterns));
}

} // namespace Step1_InstantiateStructs

namespace Step2_Unroll {

// OpTy can be any LoopLikeOpInterface
// TODO: not guaranteed to work with WhileOp, can try with our custom attributes though.
template <typename OpTy> class LoopUnrollPattern : public OpRewritePattern<OpTy> {
public:
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy loopOp, PatternRewriter &rewriter) const override {
    if (auto maybeConstant = getConstantTripCount(loopOp)) {
      uint64_t tripCount = *maybeConstant;
      if (tripCount == 0) {
        rewriter.eraseOp(loopOp);
        return success();
      } else if (tripCount == 1) {
        return loopOp.promoteIfSingleIteration(rewriter);
      }
      return loopUnrollByFactor(loopOp, tripCount);
    }
    return failure();
  }

private:
  /// Returns the trip count of the loop-like op if its low bound, high bound and step are
  /// constants, `nullopt` otherwise. Trip count is computed as ceilDiv(highBound - lowBound, step).
  static std::optional<int64_t> getConstantTripCount(LoopLikeOpInterface loopOp) {
    std::optional<OpFoldResult> lbVal = loopOp.getSingleLowerBound();
    std::optional<OpFoldResult> ubVal = loopOp.getSingleUpperBound();
    std::optional<OpFoldResult> stepVal = loopOp.getSingleStep();
    if (!lbVal.has_value() || !ubVal.has_value() || !stepVal.has_value()) {
      return std::nullopt;
    }
    return constantTripCount(lbVal.value(), ubVal.value(), stepVal.value());
  }
};

LogicalResult run(ModuleOp modOp, ConversionTracker &tracker) {
  MLIRContext *ctx = modOp.getContext();
  RewritePatternSet patterns(ctx);
  patterns.add<LoopUnrollPattern<scf::ForOp>>(ctx);
  patterns.add<LoopUnrollPattern<affine::AffineForOp>>(ctx);

  return applyAndFoldGreedily(modOp, tracker, std::move(patterns));
}
} // namespace Step2_Unroll

namespace Step3_InstantiateAffineMaps {

SmallVector<std::unique_ptr<Region>> moveRegions(Operation *op) {
  SmallVector<std::unique_ptr<Region>> newRegions;
  for (Region &region : op->getRegions()) {
    auto newRegion = std::make_unique<Region>();
    newRegion->takeBody(region);
    newRegions.push_back(std::move(newRegion));
  }
  return newRegions;
}

// Adapted from `mlir::getConstantIntValues()` but that one failed in CI for an unknown reason. This
// version uses a basic loop instead of llvm::map_to_vector().
std::optional<SmallVector<int64_t>> getConstantIntValues(ArrayRef<OpFoldResult> ofrs) {
  SmallVector<int64_t> res;
  for (OpFoldResult ofr : ofrs) {
    std::optional<int64_t> cv = getConstantIntValue(ofr);
    if (!cv.has_value()) {
      return std::nullopt;
    }
    res.push_back(cv.value());
  }
  return res;
}

struct AffineMapFolder {
  struct Input {
    OperandRangeRange mapOpGroups;
    DenseI32ArrayAttr dimsPerGroup;
    ArrayRef<Attribute> paramsOfStructTy;
  };

  struct Output {
    SmallVector<SmallVector<Value>> mapOpGroups;
    SmallVector<int32_t> dimsPerGroup;
    SmallVector<Attribute> paramsOfStructTy;
  };

  static inline SmallVector<ValueRange> getConvertedMapOpGroups(Output out) {
    return llvm::map_to_vector(out.mapOpGroups, [](const SmallVector<Value> &grp) {
      return ValueRange(grp);
    });
  }

  static LogicalResult
  fold(PatternRewriter &rewriter, const Input &in, Output &out, Operation *op, const char *aspect) {
    if (in.mapOpGroups.empty()) {
      // No affine map operands so nothing to do
      return failure();
    }

    assert(in.mapOpGroups.size() <= in.paramsOfStructTy.size());
    assert(std::cmp_equal(in.mapOpGroups.size(), in.dimsPerGroup.size()));

    size_t idx = 0; // index in `mapOpGroups`, i.e. the number of AffineMapAttr encountered
    for (Attribute sizeAttr : in.paramsOfStructTy) {
      if (AffineMapAttr m = dyn_cast<AffineMapAttr>(sizeAttr)) {
        ValueRange currMapOps = in.mapOpGroups[idx++];
        LLVM_DEBUG(
            llvm::dbgs() << "[AffineMapFolder] currMapOps: " << debug::toStringList(currMapOps)
                         << '\n'
        );
        SmallVector<OpFoldResult> currMapOpsCast = getAsOpFoldResult(currMapOps);
        LLVM_DEBUG(
            llvm::dbgs() << "[AffineMapFolder] currMapOps as fold results: "
                         << debug::toStringList(currMapOpsCast) << '\n'
        );
        if (auto constOps = Step3_InstantiateAffineMaps::getConstantIntValues(currMapOpsCast)) {
          SmallVector<Attribute> result;
          bool hasPoison = false; // indicates divide by 0 or mod by <1
          auto constAttrs = llvm::map_to_vector(*constOps, [&rewriter](int64_t v) -> Attribute {
            return rewriter.getIndexAttr(v);
          });
          LogicalResult foldResult = m.getAffineMap().constantFold(constAttrs, result, &hasPoison);
          if (hasPoison) {
            LLVM_DEBUG(op->emitRemark().append(
                "Cannot fold affine_map for ", aspect, " ", out.paramsOfStructTy.size(),
                " due to divide by 0 or modulus with negative divisor"
            ));
            return failure();
          }
          if (failed(foldResult)) {
            LLVM_DEBUG(op->emitRemark().append(
                "Folding affine_map for ", aspect, " ", out.paramsOfStructTy.size(), " failed"
            ));
            return failure();
          }
          if (result.size() != 1) {
            LLVM_DEBUG(op->emitRemark().append(
                "Folding affine_map for ", aspect, " ", out.paramsOfStructTy.size(), " produced ",
                result.size(), " results but expected 1"
            ));
            return failure();
          }
          assert(!llvm::isa<AffineMapAttr>(result[0]) && "not converted");
          out.paramsOfStructTy.push_back(result[0]);
          continue;
        }
        // If affine but not foldable, preserve the map ops
        out.mapOpGroups.emplace_back(currMapOps);
        out.dimsPerGroup.push_back(in.dimsPerGroup[idx - 1]); // idx was already incremented
      }
      // If not affine and foldable, preserve the original
      out.paramsOfStructTy.push_back(sizeAttr);
    }
    assert(idx == in.mapOpGroups.size() && "all affine_map not processed");
    assert(
        in.paramsOfStructTy.size() == out.paramsOfStructTy.size() &&
        "produced wrong number of dimensions"
    );

    return success();
  }
};

/// Instantiate parameterized ArrayType resulting from CreateArrayOp.
class InstantiateAtCreateArrayOp final : public OpRewritePattern<CreateArrayOp> {
  ConversionTracker &tracker_;

public:
  InstantiateAtCreateArrayOp(MLIRContext *ctx, ConversionTracker &tracker)
      : OpRewritePattern(ctx), tracker_(tracker) {}

  LogicalResult matchAndRewrite(CreateArrayOp op, PatternRewriter &rewriter) const override {
    ArrayType oldResultType = op.getType();

    AffineMapFolder::Output out;
    AffineMapFolder::Input in = {
        op.getMapOperands(),
        op.getNumDimsPerMapAttr(),
        oldResultType.getDimensionSizes(),
    };
    if (failed(AffineMapFolder::fold(rewriter, in, out, op, "array dimension"))) {
      return failure();
    }

    ArrayType newResultType = ArrayType::get(oldResultType.getElementType(), out.paramsOfStructTy);
    if (newResultType == oldResultType) {
      // nothing changed
      return failure();
    }
    // ASSERT: folding only preserves the original Attribute or converts affine to integer
    assert(tracker_.isLegalConversion(oldResultType, newResultType, "InstantiateAtCreateArrayOp"));
    LLVM_DEBUG(
        llvm::dbgs() << "[InstantiateAtCreateArrayOp] instantiating " << oldResultType << " as "
                     << newResultType << " in \"" << op << "\"\n"
    );
    replaceOpWithNewOp<CreateArrayOp>(
        rewriter, op, newResultType, AffineMapFolder::getConvertedMapOpGroups(out), out.dimsPerGroup
    );
    return success();
  }
};

/// Update the array element type by looking at the values stored into it from uses.
class UpdateArrayElemFromWrite final : public OpRewritePattern<CreateArrayOp> {
  ConversionTracker &tracker_;

public:
  UpdateArrayElemFromWrite(MLIRContext *ctx, ConversionTracker &tracker)
      : OpRewritePattern(ctx), tracker_(tracker) {}

  LogicalResult matchAndRewrite(CreateArrayOp op, PatternRewriter &rewriter) const override {
    Value createResult = op.getResult();
    ArrayType createResultType = dyn_cast<ArrayType>(createResult.getType());
    assert(createResultType && "CreateArrayOp must produce ArrayType");
    Type oldResultElemType = createResultType.getElementType();

    // Look for WriteArrayOp where the array reference is the result of the CreateArrayOp and the
    // element type is different.
    Type newResultElemType = nullptr;
    for (Operation *user : createResult.getUsers()) {
      if (WriteArrayOp writeOp = dyn_cast<WriteArrayOp>(user)) {
        if (writeOp.getArrRef() != createResult) {
          continue;
        }
        Type writeRValueType = writeOp.getRvalue().getType();
        if (writeRValueType == oldResultElemType) {
          continue;
        }
        if (newResultElemType && newResultElemType != writeRValueType) {
          LLVM_DEBUG(
              llvm::dbgs()
              << "[UpdateArrayElemFromWrite] multiple possible element types for CreateArrayOp "
              << newResultElemType << " vs " << writeRValueType << '\n'
          );
          return failure();
        }
        newResultElemType = writeRValueType;
      }
    }
    if (!newResultElemType) {
      // no replacement type found
      return failure();
    }
    if (!tracker_.isLegalConversion(
            oldResultElemType, newResultElemType, "UpdateArrayElemFromWrite"
        )) {
      return failure();
    }
    ArrayType newType = createResultType.cloneWith(newResultElemType);
    rewriter.modifyOpInPlace(op, [&createResult, &newType]() { createResult.setType(newType); });
    LLVM_DEBUG(llvm::dbgs() << "[UpdateArrayElemFromWrite] updated result type of " << op << '\n');
    return success();
  }
};

/// Update the type of FieldDefOp instances by checking the updated types from FieldWriteOp.
class UpdateFieldTypeFromWrite final : public OpRewritePattern<FieldDefOp> {
  ConversionTracker &tracker_;

public:
  UpdateFieldTypeFromWrite(MLIRContext *ctx, ConversionTracker &tracker)
      : OpRewritePattern(ctx), tracker_(tracker) {}

  LogicalResult matchAndRewrite(FieldDefOp op, PatternRewriter &rewriter) const override {
    // Find all uses of the field symbol name within its parent struct.
    FailureOr<StructDefOp> parentRes = getParentOfType<StructDefOp>(op);
    assert(succeeded(parentRes) && "FieldDefOp parent is always StructDefOp"); // per ODS def

    // If the symbol is used by a FieldWriteOp with a different result type then change
    // the type of the FieldDefOp to match the FieldWriteOp result type.
    Type newType = nullptr;
    if (auto fieldUsers = SymbolTable::getSymbolUses(op, parentRes.value())) {
      std::optional<Location> newTypeLoc = std::nullopt;
      for (SymbolTable::SymbolUse symUse : fieldUsers.value()) {
        if (FieldWriteOp writeOp = llvm::dyn_cast<FieldWriteOp>(symUse.getUser())) {
          Type writeToType = writeOp.getVal().getType();
          LLVM_DEBUG(llvm::dbgs() << "[UpdateFieldTypeFromWrite] checking " << writeOp << '\n');
          if (!newType) {
            // If a new type has not yet been discovered, store the new type.
            newType = writeToType;
            newTypeLoc = writeOp.getLoc();
          } else if (writeToType != newType) {
            // If a new type has already been discovered from another FieldWriteOp and the current
            // FieldWriteOp writes a different type, fail the conversion. There should only be one
            // write for each field of a struct but do not rely on that assumption.
            return rewriter.notifyMatchFailure(op, [&](Diagnostic &diag) {
              diag.append(
                  "Cannot update type of '", FieldDefOp::getOperationName(),
                  "' because there are multiple '", FieldWriteOp::getOperationName(),
                  "' with different value types"
              );
              if (newTypeLoc) {
                diag.attachNote(*newTypeLoc).append("type written here is ", newType);
              }
              diag.attachNote(writeOp.getLoc()).append("type written here is ", writeToType);
            });
          }
        }
      }
    }
    if (!newType || newType == op.getType()) {
      // nothing changed
      return failure();
    }
    if (!tracker_.isLegalConversion(op.getType(), newType, "UpdateFieldTypeFromWrite")) {
      return failure();
    }
    LLVM_DEBUG(llvm::dbgs() << "[UpdateFieldTypeFromWrite] replaced " << op);
    DictionaryAttr attrs = op->getDiscardableAttrDictionary();
    FieldDefOp newOp = replaceOpWithNewOp<FieldDefOp>(rewriter, op, op.getSymName(), newType);
    newOp->setDiscardableAttrs(attrs);
    LLVM_DEBUG(llvm::dbgs() << " with " << newOp << '\n');
    return success();
  }
};

/// Updates the result type in Ops with the InferTypeOpAdaptor trait including ReadArrayOp,
/// ExtractArrayOp, etc.
class UpdateInferredResultTypes final : public OpTraitRewritePattern<OpTrait::InferTypeOpAdaptor> {
  ConversionTracker &tracker_;

public:
  UpdateInferredResultTypes(MLIRContext *ctx, ConversionTracker &tracker)
      : OpTraitRewritePattern(ctx), tracker_(tracker) {}

  LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
    SmallVector<Type, 1> inferredResultTypes;
    InferTypeOpInterface retTypeFn = cast<InferTypeOpInterface>(op);
    LogicalResult result = retTypeFn.inferReturnTypes(
        op->getContext(), op->getLoc(), op->getOperands(), op->getRawDictionaryAttrs(),
        op->getPropertiesStorage(), op->getRegions(), inferredResultTypes
    );
    if (failed(result)) {
      return failure();
    }
    if (op->getResultTypes() == inferredResultTypes) {
      // nothing changed
      return failure();
    }
    if (!tracker_.areLegalConversions(
            op->getResultTypes(), inferredResultTypes, "UpdateInferredResultTypes"
        )) {
      return failure();
    }

    // Move nested region bodies and replace the original op with the updated types list.
    LLVM_DEBUG(llvm::dbgs() << "[UpdateInferredResultTypes] replaced " << *op);
    SmallVector<std::unique_ptr<Region>> newRegions = moveRegions(op);
    Operation *newOp = rewriter.create(
        op->getLoc(), op->getName().getIdentifier(), op->getOperands(), inferredResultTypes,
        op->getAttrs(), op->getSuccessors(), newRegions
    );
    rewriter.replaceOp(op, newOp);
    LLVM_DEBUG(llvm::dbgs() << " with " << *newOp << '\n');
    return success();
  }
};

/// Update FuncOp return type by checking the updated types from ReturnOp.
class UpdateFuncTypeFromReturn final : public OpRewritePattern<FuncOp> {
  ConversionTracker &tracker_;

public:
  UpdateFuncTypeFromReturn(MLIRContext *ctx, ConversionTracker &tracker)
      : OpRewritePattern(ctx), tracker_(tracker) {}

  LogicalResult matchAndRewrite(FuncOp op, PatternRewriter &rewriter) const override {
    Region &body = op.getFunctionBody();
    if (body.empty()) {
      return failure();
    }
    ReturnOp retOp = llvm::dyn_cast<ReturnOp>(body.back().getTerminator());
    assert(retOp && "final op in body region must be return");
    OperandRange::type_range tyFromReturnOp = retOp.getOperands().getTypes();

    FunctionType oldFuncTy = op.getFunctionType();
    if (oldFuncTy.getResults() == tyFromReturnOp) {
      // nothing changed
      return failure();
    }
    if (!tracker_.areLegalConversions(
            oldFuncTy.getResults(), tyFromReturnOp, "UpdateFuncTypeFromReturn"
        )) {
      return failure();
    }

    rewriter.modifyOpInPlace(op, [&]() {
      op.setFunctionType(rewriter.getFunctionType(oldFuncTy.getInputs(), tyFromReturnOp));
    });
    LLVM_DEBUG(
        llvm::dbgs() << "[UpdateFuncTypeFromReturn] changed " << op.getSymName() << " from "
                     << oldFuncTy << " to " << op.getFunctionType() << '\n'
    );
    return success();
  }
};

/// Update CallOp result type based on the updated return type from the target FuncOp.
/// This only applies to global (i.e. non-struct) functions because the functions within structs
/// only return StructType or nothing and propagating those can result in bringing un-instantiated
/// types from a templated struct into the current call which will give errors.
class UpdateGlobalCallOpTypes final : public OpRewritePattern<CallOp> {
  ConversionTracker &tracker_;

public:
  UpdateGlobalCallOpTypes(MLIRContext *ctx, ConversionTracker &tracker)
      : OpRewritePattern(ctx), tracker_(tracker) {}

  LogicalResult matchAndRewrite(CallOp op, PatternRewriter &rewriter) const override {
    SymbolTableCollection tables;
    auto lookupRes = lookupTopLevelSymbol<FuncOp>(tables, op.getCalleeAttr(), op);
    if (failed(lookupRes)) {
      return failure();
    }
    FuncOp targetFunc = lookupRes->get();
    if (targetFunc.isInStruct()) {
      // this pattern only applies when the callee is NOT in a struct
      return failure();
    }
    if (op.getResultTypes() == targetFunc.getFunctionType().getResults()) {
      // nothing changed
      return failure();
    }
    if (!tracker_.areLegalConversions(
            op.getResultTypes(), targetFunc.getFunctionType().getResults(),
            "UpdateGlobalCallOpTypes"
        )) {
      return failure();
    }

    LLVM_DEBUG(llvm::dbgs() << "[UpdateGlobalCallOpTypes] replaced " << op);
    CallOp newOp = replaceOpWithNewOp<CallOp>(rewriter, op, targetFunc, op.getArgOperands());
    LLVM_DEBUG(llvm::dbgs() << " with " << newOp << '\n');
    return success();
  }
};

/// Instantiate parameterized StructType resulting from CallOp targeting "compute()" functions.
class InstantiateAtCallOpCompute final : public OpRewritePattern<CallOp> {
  ConversionTracker &tracker_;

public:
  InstantiateAtCallOpCompute(MLIRContext *ctx, ConversionTracker &tracker)
      : OpRewritePattern(ctx), tracker_(tracker) {}

  LogicalResult matchAndRewrite(CallOp op, PatternRewriter &rewriter) const override {
    if (!op.calleeIsStructCompute()) {
      // this pattern only applies when the callee is "compute()" within a struct
      return failure();
    }
    StructType oldRetTy = op.getComputeSingleResultType();
    LLVM_DEBUG({
      llvm::dbgs() << "[InstantiateAtCallOpCompute] target: " << op.getCallee() << '\n';
      llvm::dbgs() << "[InstantiateAtCallOpCompute]   oldRetTy: " << oldRetTy << '\n';
    });
    ArrayAttr params = oldRetTy.getParams();
    if (isNullOrEmpty(params)) {
      // nothing to do if the StructType is not parameterized
      return failure();
    }

    AffineMapFolder::Output out;
    AffineMapFolder::Input in = {
        op.getMapOperands(),
        op.getNumDimsPerMapAttr(),
        params.getValue(),
    };
    if (!in.mapOpGroups.empty()) {
      // If there are affine map operands, attempt to fold them to a constant.
      if (failed(AffineMapFolder::fold(rewriter, in, out, op, "struct parameter"))) {
        return failure();
      }
      LLVM_DEBUG({
        llvm::dbgs() << "[InstantiateAtCallOpCompute]   folded affine_map in result type params\n";
      });
    } else {
      // If there are no affine map operands, attempt to refine the result type of the CallOp using
      // the function argument types and the type of the target function.
      auto callArgTypes = op.getArgOperands().getTypes();
      if (callArgTypes.empty()) {
        // no refinement posible if no function arguments
        return failure();
      }
      SymbolTableCollection tables;
      auto lookupRes = lookupTopLevelSymbol<FuncOp>(tables, op.getCalleeAttr(), op);
      if (failed(lookupRes)) {
        return failure();
      }
      if (failed(instantiateViaTargetType(in, out, callArgTypes, lookupRes->get()))) {
        return failure();
      }
      LLVM_DEBUG({
        llvm::dbgs() << "[InstantiateAtCallOpCompute]   propagated instantiations via symrefs in "
                        "result type params: "
                     << debug::toStringList(out.paramsOfStructTy) << '\n';
      });
    }

    StructType newRetTy = StructType::get(oldRetTy.getNameRef(), out.paramsOfStructTy);
    if (newRetTy == oldRetTy) {
      // nothing changed
      return failure();
    }
    // The `newRetTy` is computed via instantiateViaTargetType() which can only preserve the
    // original Attribute or convert to a concrete attribute via the unification process. Thus, if
    // the conversion here is illegal it means there is a type conflict within the LLZK code that
    // prevents instantiation of the struct with the requested type.
    if (!tracker_.isLegalConversion(oldRetTy, newRetTy, "InstantiateAtCallOpCompute")) {
      return rewriter.notifyMatchFailure(op, [&](Diagnostic &diag) {
        diag.append(
            "result type mismatch: due to struct instantiation, expected type ", newRetTy,
            ", but found ", oldRetTy
        );
      });
    }
    LLVM_DEBUG(
        llvm::dbgs() << "[InstantiateAtCallOpCompute] instantiating " << oldRetTy << " as "
                     << newRetTy << " in \"" << op << "\"\n"
    );
    replaceOpWithNewOp<CallOp>(
        rewriter, op, TypeRange {newRetTy}, op.getCallee(),
        AffineMapFolder::getConvertedMapOpGroups(out), out.dimsPerGroup, op.getArgOperands()
    );
    return success();
  }

private:
  /// Use the type of the target function to propagate instantation knowledge from the function
  /// argument types to the function return type in the CallOp.
  inline LogicalResult instantiateViaTargetType(
      const AffineMapFolder::Input &in, AffineMapFolder::Output &out,
      OperandRange::type_range callArgTypes, FuncOp targetFunc
  ) const {
    assert(targetFunc.isStructCompute()); // since `op.calleeIsStructCompute()`
    ArrayAttr targetResTyParams = targetFunc.getComputeSingleResultType().getParams();
    assert(!isNullOrEmpty(targetResTyParams)); // same cardinality as `in.paramsOfStructTy`
    assert(in.paramsOfStructTy.size() == targetResTyParams.size()); // verifier ensures this

    if (llvm::all_of(in.paramsOfStructTy, [](Attribute a) { return isConcreteAttr<>(a); })) {
      // Nothing can change if everything is already concrete
      return failure();
    }

    LLVM_DEBUG({
      llvm::dbgs() << '[' << __FUNCTION__ << ']'
                   << " call arg types: " << debug::toStringList(callArgTypes) << '\n';
      llvm::dbgs() << '[' << __FUNCTION__ << ']' << " target func arg types: "
                   << debug::toStringList(targetFunc.getArgumentTypes()) << '\n';
      llvm::dbgs() << '[' << __FUNCTION__ << ']'
                   << " struct params @ call: " << debug::toStringList(in.paramsOfStructTy) << '\n';
      llvm::dbgs() << '[' << __FUNCTION__ << ']'
                   << " target struct params: " << debug::toStringList(targetResTyParams) << '\n';
    });

    UnificationMap unifications;
    bool unifies = typeListsUnify(targetFunc.getArgumentTypes(), callArgTypes, {}, &unifications);
    assert(unifies && "should have been checked by verifiers");

    LLVM_DEBUG({
      llvm::dbgs() << '[' << __FUNCTION__ << ']'
                   << " unifications of arg types: " << debug::toStringList(unifications) << '\n';
    });

    // Check for LHS SymRef (i.e. from the target function) that have RHS concrete Attributes (i.e.
    // from the call argument types) without any struct parameters (because the type with concrete
    // struct parameters will be used to instantiate the target struct rather than the fully
    // flattened struct type resulting in type mismatch of the callee to target) and perform those
    // replacements in the `targetFunc` return type to produce the new result type for the CallOp.
    SmallVector<Attribute> newReturnStructParams = llvm::map_to_vector(
        llvm::zip_equal(targetResTyParams.getValue(), in.paramsOfStructTy),
        [&unifications](std::tuple<Attribute, Attribute> p) {
      Attribute fromCall = std::get<1>(p);
      // Preserve attributes that are already concrete at the call site. Otherwise attempt to lookup
      // non-parameterized concrete unification for the target struct parameter symbol.
      if (!isConcreteAttr<>(fromCall)) {
        Attribute fromTgt = std::get<0>(p);
        LLVM_DEBUG({
          llvm::dbgs() << "[instantiateViaTargetType]   fromCall = " << fromCall << '\n';
          llvm::dbgs() << "[instantiateViaTargetType]   fromTgt = " << fromTgt << '\n';
        });
        assert(llvm::isa<SymbolRefAttr>(fromTgt));
        auto it = unifications.find(std::make_pair(llvm::cast<SymbolRefAttr>(fromTgt), Side::LHS));
        if (it != unifications.end()) {
          Attribute unifiedAttr = it->second;
          LLVM_DEBUG({
            llvm::dbgs() << "[instantiateViaTargetType]   unifiedAttr = " << unifiedAttr << '\n';
          });
          if (unifiedAttr && isConcreteAttr<false>(unifiedAttr)) {
            return unifiedAttr;
          }
        }
      }
      return fromCall;
    }
    );

    out.paramsOfStructTy = newReturnStructParams;
    assert(out.paramsOfStructTy.size() == in.paramsOfStructTy.size() && "post-condition");
    assert(out.mapOpGroups.empty() && "post-condition");
    assert(out.dimsPerGroup.empty() && "post-condition");
    return success();
  }
};

LogicalResult run(ModuleOp modOp, ConversionTracker &tracker) {
  MLIRContext *ctx = modOp.getContext();
  RewritePatternSet patterns(ctx);
  patterns.add<
      // clang-format off
      InstantiateAtCreateArrayOp,
      UpdateFieldTypeFromWrite,
      UpdateInferredResultTypes,
      UpdateFuncTypeFromReturn,
      UpdateGlobalCallOpTypes,
      InstantiateAtCallOpCompute,
      UpdateArrayElemFromWrite
      // clang-format on
      >(ctx, tracker);

  return applyAndFoldGreedily(modOp, tracker, std::move(patterns));
}

} // namespace Step3_InstantiateAffineMaps

namespace Step4_Cleanup {

template <typename OpTy> class EraseOpPattern : public OpConversionPattern<OpTy> {
public:
  EraseOpPattern(MLIRContext *ctx) : OpConversionPattern<OpTy>(ctx) {}

  LogicalResult
  matchAndRewrite(OpTy op, OpTy::Adaptor, ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

LogicalResult run(ModuleOp modOp, const ConversionTracker &tracker) {
  FailureOr<ModuleOp> topRoot = getTopRootModule(modOp);
  if (failed(topRoot)) {
    return failure();
  }

  // Use a conversion to erase instantiated structs if they have no other references.
  //
  // TODO: There's a chance the "no other references" criteria will leave some behind when running
  // only a single pass of this because they may reference each other. Maybe I can check if the
  // references are only located within another struct in the list, but would have to do a deep
  // deep lookup to ensure no references and avoid infinite loop back on self. The CallGraphAnalysis
  // is not sufficient because it looks only at calls but there could be (although unlikely) a
  // FieldDefOp referencing a struct type despite having no calls to that struct's functions.
  //
  // TODO: There's another scenario that leaves some behind. Once a StructDefOp is visited and
  // considered legal, that decision cannot be reversed. Hence, StructDefOp that become illegal only
  // after removing another one that uses it will not be removed. See
  // test/Dialect/LLZK/instantiate_structs_affine_pass.llzk
  // One idea is to use one of the `SymbolTable::getSymbolUses` functions starting from a struct
  // listed in `instantiatedNames` to determine if it is reachable from some other struct that is
  // NOT listed there and remove it if not. For efficiency, this reachability information can be
  // pre-computed and or cached.
  //
  DenseSet<SymbolRefAttr> instantiatedNames = tracker.getInstantiatedStructNames();
  auto isLegalStruct = [&](bool emitWarning, StructDefOp op) {
    if (instantiatedNames.contains(op.getType().getNameRef())) {
      if (!hasUsesWithin(op, *topRoot)) {
        // Parameterized struct with no uses is illegal, i.e. should be removed.
        return false;
      }
      if (emitWarning) {
        op.emitWarning("Parameterized struct still has uses!").report();
      }
    }
    return true;
  };

  // Perform the conversion, i.e. remove StructDefOp that were instantiated and are unused.
  MLIRContext *ctx = modOp.getContext();
  RewritePatternSet patterns(ctx);
  patterns.add<EraseOpPattern<StructDefOp>>(ctx);
  ConversionTarget target = newBaseTarget(ctx);
  target.addDynamicallyLegalOp<StructDefOp>(std::bind_front(isLegalStruct, false));
  if (failed(applyFullConversion(modOp, target, std::move(patterns)))) {
    return failure();
  }

  // Warn about any structs that were instantiated but still have uses elsewhere.
  modOp->walk([&](StructDefOp op) {
    isLegalStruct(true, op);
    return WalkResult::skip(); // StructDefOp cannot be nested
  });

  return success();
}

} // namespace Step4_Cleanup

class FlatteningPass : public llzk::impl::FlatteningPassBase<FlatteningPass> {

  static constexpr unsigned LIMIT = 1000;

  void runOnOperation() override {
    ModuleOp modOp = getOperation();

    ConversionTracker tracker;
    unsigned loopCount = 0;
    do {
      ++loopCount;
      if (loopCount > LIMIT) {
        llvm::errs() << DEBUG_TYPE << " exceeded the limit of " << LIMIT << " iterations!\n";
        signalPassFailure();
        break;
      }
      tracker.resetModifiedFlag();

      // Find calls to "compute()" that return a parameterized struct and replace it to call a
      // flattened version of the struct that has parameters replaced with the constant values.
      // Create the necessary instantiated/flattened struct in the same location as the original.
      if (failed(Step1_InstantiateStructs::run(modOp, tracker))) {
        llvm::errs() << DEBUG_TYPE << " failed while replacing concrete-parameter struct types\n";
        signalPassFailure();
        break;
      }

      // Unroll loops with known iterations.
      if (failed(Step2_Unroll::run(modOp, tracker))) {
        llvm::errs() << DEBUG_TYPE << " failed while unrolling loops\n";
        signalPassFailure();
        break;
      }

      // Instantiate affine_map parameters of StructType and ArrayType.
      if (failed(Step3_InstantiateAffineMaps::run(modOp, tracker))) {
        llvm::errs() << DEBUG_TYPE << " failed while instantiating `affine_map` parameters\n";
        signalPassFailure();
        break;
      }

      LLVM_DEBUG(if (tracker.isModified()) {
        llvm::dbgs() << "=====================================================================\n";
        llvm::dbgs() << " Dumping module between iterations of " << DEBUG_TYPE << " \n";
        modOp.print(llvm::dbgs(), OpPrintingFlags().assumeVerified());
        llvm::dbgs() << "=====================================================================\n";
      });
    } while (tracker.isModified());

    // Remove the parameterized StructDefOp that were instantiated.
    if (failed(Step4_Cleanup::run(modOp, tracker))) {
      llvm::errs() << DEBUG_TYPE
                   << " failed while removing parameterized structs that were replaced with "
                      "instantiated versions\n";
      signalPassFailure();
    }

    // Dump the current IR if the pass failed
    LLVM_DEBUG(if (this->getPassState().irAndPassFailed.getInt()) {
      llvm::dbgs() << "=====================================================================\n";
      llvm::dbgs() << " Dumping module after failure of pass " << DEBUG_TYPE << " \n";
      modOp.print(llvm::dbgs(), OpPrintingFlags().assumeVerified());
      llvm::dbgs() << "=====================================================================\n";
    });
  }
};

} // namespace

std::unique_ptr<Pass> llzk::createFlatteningPass() { return std::make_unique<FlatteningPass>(); };
