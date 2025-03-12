#include "llzk/Dialect/LLZK/IR/Dialect.h"
#include "llzk/Dialect/LLZK/IR/Ops.h"
#include "llzk/Dialect/LLZK/Transforms/LLZKTransformationPasses.h"
#include "llzk/Dialect/LLZK/Util/AttributeHelper.h"
#include "llzk/Dialect/LLZK/Util/Debug.h"
#include "llzk/Dialect/LLZK/Util/SymbolHelper.h"

#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Affine/LoopUtils.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/SCF/Transforms/Patterns.h>
#include <mlir/Dialect/SCF/Utils/Utils.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Interfaces/InferTypeOpInterface.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DepthFirstIterator.h>
#include <llvm/Support/Debug.h>

/// Include the generated base pass class definitions.
namespace llzk {
#define GEN_PASS_DEF_INSTANTIATESTRUCTSPASS
#include "llzk/Dialect/LLZK/Transforms/LLZKTransformationPasses.h.inc"
} // namespace llzk

using namespace llzk;
using namespace mlir;

#define DEBUG_TYPE "llzk-flatten"

namespace {

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
    rewriter.replaceOpWithNewOp<OpTy>(
        op, TypeRange(newResultTypes), adaptor.getOperands(), ArrayRef(newAttrs)
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
      rewriter.replaceOpWithNewOp<CreateArrayOp>(
          op, llvm::cast<ArrayType>(newType), adapter.getElements()
      );
    } else {
      rewriter.replaceOpWithNewOp<CreateArrayOp>(
          op, llvm::cast<ArrayType>(newType), adapter.getMapOperands(), numDimsPerMap
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

/// Return a new `ConversionTarget` allowing all LLZK-required dialects and defining Op legality
/// based on the given `TypeConverter` for Ops listed in both fields of `OpClassesWithStructTypes`
/// and in `AdditionalOpTypes`.
template <typename... AdditionalOpTypes>
ConversionTarget newConverterDefinedTarget(TypeConverter &tyConv, MLIRContext *ctx) {
  ConversionTarget target = newBaseTarget(ctx);
  auto inserter = [&](auto... opClasses) {
    target.addDynamicallyLegalOp<decltype(opClasses)...>([&tyConv](Operation *op) {
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

class ConversionTracker {
  /// Tracks if some step performed a modification of the code such that another pass should be run.
  bool modified;
  /// Maps original remote (i.e. use site) type to new remote type.
  /// Note: The keys are always parameterized StructType and the values are no-parameter StructType.
  DenseMap<StructType, StructType> structInstantiations;
  /// Contains the reverse of mappings in `structInstantiations` for use in legal conversion check.
  DenseMap<StructType, StructType> reverseInstantiations;
  /// Maps new remote type (i.e. the values in 'structInstantiations') to location of the compute()
  /// calls that cause instantiation
  DenseMap<StructType, DenseSet<Location>> newTyComputeLocs;

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

  const DenseMap<StructType, StructType> &getAllInstantiations() const {
    return structInstantiations;
  }

  /// Collect the fully-qualified names of all structs that were instantiated.
  DenseSet<SymbolRefAttr> getInstantiatedStructNames() const {
    DenseSet<SymbolRefAttr> instantiatedNames;
    for (const auto &[origRemoteTy, _] : structInstantiations) {
      instantiatedNames.insert(origRemoteTy.getNameRef());
    }
    return instantiatedNames;
  }

  /// Record the location of a "compute" function that produces the given instantiated `StructType`.
  void recordLocation(StructType newType, Location instantiationLocation) {
    newTyComputeLocs[newType].insert(instantiationLocation);
  }

  /// Get the locations of all "compute" functions that produce the given instantiated `StructType`.
  const DenseSet<Location> *getLocations(StructType newType) const {
    auto res = newTyComputeLocs.find(newType);
    return (res == newTyComputeLocs.end()) ? nullptr : &res->getSecond();
  }

  /// Check if the type conversion  is legal, i.e. the new type unifies with and is more concrete
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

namespace Step1_FindComputeTypes {

class ParameterizedStructUseTypeConverter : public TypeConverter {
  ConversionTracker &tracker_;

public:
  ParameterizedStructUseTypeConverter(ConversionTracker &tracker)
      : TypeConverter(), tracker_(tracker) {

    addConversion([](Type inputTy) { return inputTy; });

    addConversion([this](StructType inputTy) -> StructType {
      // First check for a cached entry
      if (auto opt = tracker_.getInstantiation(inputTy)) {
        return opt.value();
      }
      // Otherwise, try to perform a conversion
      if (ArrayAttr params = inputTy.getParams()) {
        // If all prameters are concrete values (Integer or Type), then replace with a
        // no-parameter StructType referencing the de-parameterized struct.
        if (llvm::all_of(params, isConcreteAttr<>)) {
          StructType result =
              StructType::get(appendLeafName(inputTy.getNameRef(), "_" + shortString(params)));
          LLVM_DEBUG(
              llvm::dbgs() << "[ParameterizedStructUseTypeConverter] instantiating " << inputTy
                           << " as " << result << "\n"
          );
          tracker_.recordInstantiation(inputTy, result);
          return result;
        }
      }
      return inputTy;
    });
  }
};

class CallComputePattern : public OpConversionPattern<CallOp> {
  ConversionTracker &tracker_;

public:
  CallComputePattern(TypeConverter &converter, MLIRContext *ctx, ConversionTracker &tracker)
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
        assert(isNullOrEmpty(newStTy.getParams()) && "must be fully instantiated"); // I think
        calleeAttr = appendLeaf(newStTy.getNameRef(), calleeAttr.getLeafReference());
        tracker_.recordLocation(newStTy, op.getLoc());
      }
    } else if (op.calleeIsStructConstrain()) {
      if (StructType newStTy = getAtIndex<StructType>(adapter.getArgOperands().getTypes(), 0)) {
        assert(isNullOrEmpty(newStTy.getParams()) && "must be fully instantiated"); // I think
        calleeAttr = appendLeaf(newStTy.getNameRef(), calleeAttr.getLeafReference());
      }
    }
    rewriter.replaceOpWithNewOp<CallOp>(
        op, newResultTypes, calleeAttr, adapter.getMapOperands(), op.getNumDimsPerMapAttr(),
        adapter.getArgOperands()
    );
    return success();
  }
};

LogicalResult run(ModuleOp modOp, ConversionTracker &tracker) {
  MLIRContext *ctx = modOp.getContext();
  ParameterizedStructUseTypeConverter tyConv(tracker);
  ConversionTarget target = newConverterDefinedTarget<>(tyConv, ctx);
  RewritePatternSet patterns = newGeneralRewritePatternSet(tyConv, ctx, target);
  patterns.add<CallComputePattern>(tyConv, ctx, tracker);
  return applyPartialConversion(modOp, target, std::move(patterns));
}

} // namespace Step1_FindComputeTypes

namespace Step2_CreateStructs {

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
        return StructType::get(inputTy.getNameRef(), ArrayAttr::get(inputTy.getContext(), updated));
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

class CallOpPattern : public OpConversionPattern<CallOp> {
public:
  CallOpPattern(TypeConverter &converter, MLIRContext *ctx)
      // future proof: use higher priority than GeneralTypeReplacePattern
      : OpConversionPattern<CallOp>(converter, ctx, 2) {}

  LogicalResult matchAndRewrite(CallOp op, OpAdaptor adapter, ConversionPatternRewriter &rewriter)
      const override {
    // Convert the result types of the CallOp
    SmallVector<Type> newResultTypes;
    if (failed(getTypeConverter()->convertTypes(op.getResultTypes(), newResultTypes))) {
      return op->emitError("Could not convert Op result types.");
    }
    rewriter.replaceOpWithNewOp<CallOp>(
        op, newResultTypes, op.getCalleeAttr(), adapter.getMapOperands(), op.getNumDimsPerMapAttr(),
        adapter.getArgOperands()
    );
    return success();
  }
};

class ConstReadOpPattern : public OpConversionPattern<ConstReadOp> {
  const DenseMap<Attribute, Attribute> &paramNameToValue;
  const DenseSet<Location> *locations;

public:
  ConstReadOpPattern(
      TypeConverter &converter, MLIRContext *ctx,
      const DenseMap<Attribute, Attribute> &paramNameToInstantiatedValue,
      const DenseSet<Location> *instantiationLocations
  )
      // future proof: use higher priority than GeneralTypeReplacePattern
      : OpConversionPattern<ConstReadOp>(converter, ctx, 2),
        paramNameToValue(paramNameToInstantiatedValue), locations(instantiationLocations) {}

  LogicalResult matchAndRewrite(
      ConstReadOp op, OpAdaptor adapter, ConversionPatternRewriter &rewriter
  ) const override {
    auto res = this->paramNameToValue.find(op.getConstNameAttr());
    if (res == this->paramNameToValue.end()) {
      return op->emitOpError("missing instantiation");
    }
    Attribute resAttr = res->second;
    if (IntegerAttr iAttr = llvm::dyn_cast<IntegerAttr>(resAttr)) {
      APInt attrValue = iAttr.getValue();
      Type origResTy = op.getType();
      if (llvm::isa<FeltType>(origResTy)) {
        rewriter.replaceOpWithNewOp<FeltConstantOp>(
            op, FeltConstAttr::get(rewriter.getContext(), attrValue)
        );
      } else if (llvm::isa<IndexType>(origResTy)) {
        rewriter.replaceOpWithNewOp<arith::ConstantIndexOp>(op, fromAPInt(attrValue));
      } else if (origResTy.isSignlessInteger(1)) {
        // Treat 0 as false and any other value as true (but give a warning if it's not 1)
        if (attrValue.isZero()) {
          rewriter.replaceOpWithNewOp<arith::ConstantIntOp>(op, false, origResTy);
        } else {
          if (!attrValue.isOne()) {
            InFlightDiagnostic warning = op.emitWarning().append(
                "Interpretting non-zero value ", stringWithoutType(iAttr), " as true"
            );
            if (locations) {
              for (Location loc : *locations) {
                warning.attachNote(loc).append(
                    "when instantiating ", StructDefOp::getOperationName(), " parameter \"",
                    res->first, "\" for this call"
                );
              }
            }
            warning.report();
          }
          rewriter.replaceOpWithNewOp<arith::ConstantIntOp>(op, true, origResTy);
        }
      } else {
        return op->emitOpError().append("unexpected result type ", origResTy);
      }
      return success();
    } else if (FeltConstAttr fcAttr = llvm::dyn_cast<FeltConstAttr>(resAttr)) {
      rewriter.replaceOpWithNewOp<FeltConstantOp>(op, fcAttr);
      return success();
    }
    return op->emitOpError().append(
        "expected value with type ", op.getType(), " but found ", resAttr
    );
  }
};

DenseMap<Attribute, Attribute>
buildNameToValueMap(ArrayAttr paramNames, ArrayAttr paramInstantiations) {
  // pre-conditions
  assert(!isNullOrEmpty(paramNames));
  assert(!isNullOrEmpty(paramInstantiations));
  assert(paramNames.size() == paramInstantiations.size());
  // Map parameter names to instantiated values
  DenseMap<Attribute, Attribute> ret;
  for (size_t i = 0, e = paramNames.size(); i < e; ++i) {
    ret[paramNames[i]] = paramInstantiations[i];
  }
  return ret;
}

LogicalResult run(ModuleOp modOp, ConversionTracker &tracker) {
  SymbolTableCollection symTables;
  MLIRContext *ctx = modOp.getContext();
  for (auto &[origRemoteTy, newRemoteTy] : tracker.getAllInstantiations()) {
    // Find the StructDefOp for the original StructType
    FailureOr<SymbolLookupResult<StructDefOp>> lookupRes =
        origRemoteTy.getDefinition(symTables, modOp);
    if (failed(lookupRes)) {
      return failure();
    }
    StructDefOp origStruct = lookupRes->get();

    // Only add new StructDefOp if it does not already exist
    // Note: parent is ModuleOp per ODS for StructDefOp.
    ModuleOp parentModule = llvm::cast<ModuleOp>(origStruct.getParentOp());
    StringAttr newStructName = newRemoteTy.getNameRef().getLeafReference();
    if (parentModule.lookupSymbol(newStructName) == nullptr) {
      StructType origStructTy = origStruct.getType();

      // Clone the original struct, apply the new name, and remove the parameters.
      StructDefOp newStruct = origStruct.clone();
      newStruct.setSymNameAttr(newStructName);
      newStruct.setConstParamsAttr(ArrayAttr {});

      // Within the new struct, replace all references to the original struct's type (i.e. the
      // locally-parameterized version) with the new flattened (i.e. no parameters) struct's type,
      // and replace all uses of the struct parameters with the concrete values.
      DenseMap<Attribute, Attribute> nameToValueMap =
          buildNameToValueMap(origStructTy.getParams(), origRemoteTy.getParams());
      MappedTypeConverter tyConv(origStructTy, newRemoteTy, nameToValueMap);
      ConversionTarget target = newConverterDefinedTarget<EmitEqualityOp>(tyConv, ctx);
      target.addIllegalOp<ConstReadOp>();
      RewritePatternSet patterns = newGeneralRewritePatternSet<EmitEqualityOp>(tyConv, ctx, target);
      patterns.add<CallOpPattern>(tyConv, ctx);
      patterns.add<ConstReadOpPattern>(
          tyConv, ctx, nameToValueMap, tracker.getLocations(newRemoteTy)
      );

      if (failed(applyFullConversion(newStruct, target, std::move(patterns)))) {
        return failure();
      }

      // Insert 'newStruct' into the parent ModuleOp of the original StructDefOp.
      parentModule.insert(origStruct, newStruct);
    }
  }

  return success();
}

} // namespace Step2_CreateStructs

namespace Step3_Unroll {

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
  MLIRContext *ctx = modOp->getContext();
  RewritePatternSet patterns(ctx);
  patterns.add<LoopUnrollPattern<scf::ForOp>>(ctx);
  patterns.add<LoopUnrollPattern<affine::AffineForOp>>(ctx);

  bool currStepModified = false;
  auto result = applyPatternsAndFoldGreedily(
      modOp->getRegion(0), std::move(patterns), GreedyRewriteConfig(), &currStepModified
  );
  tracker.updateModifiedFlag(currStepModified);
  return result;
}
} // namespace Step3_Unroll

namespace Step4_InstantiateAffineMaps {

SmallVector<std::unique_ptr<Region>> moveRegions(Operation *op) {
  SmallVector<std::unique_ptr<Region>> newRegions;
  for (Region &region : op->getRegions()) {
    auto newRegion = std::make_unique<Region>();
    newRegion->takeBody(region);
    newRegions.push_back(std::move(newRegion));
  }
  return newRegions;
}

// Adapted from `mlir::getConstantIntValues()` but that one failed in CI.
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
                         << "\n"
        );
        SmallVector<OpFoldResult> currMapOpsCast = getAsOpFoldResult(currMapOps);
        LLVM_DEBUG(
            llvm::dbgs() << "[AffineMapFolder] currMapOps as fold results: "
                         << debug::toStringList(currMapOpsCast) << "\n"
        );
        if (auto constOps = Step4_InstantiateAffineMaps::getConstantIntValues(currMapOpsCast)) {
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
    rewriter.replaceOpWithNewOp<CreateArrayOp>(
        op, newResultType, AffineMapFolder::getConvertedMapOpGroups(out), out.dimsPerGroup
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
        if (writeOp.getArrRef() == createResult) {
          Type writeRValueType = writeOp.getRvalue().getType();
          if (writeRValueType != oldResultElemType) {
            if (newResultElemType && newResultElemType != writeRValueType) {
              LLVM_DEBUG(
                  llvm::dbgs()
                  << "[UpdateArrayElemFromWrite] multiple possible element types for CreateArrayOp "
                  << newResultElemType << " vs " << writeRValueType << "\n"
              );
              return failure();
            }
            newResultElemType = writeRValueType;
          }
        }
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
    LLVM_DEBUG(llvm::dbgs() << "[UpdateArrayElemFromWrite] updated result type of " << op << "\n");
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
      Type fieldDefType = op.getType();
      for (SymbolTable::SymbolUse symUse : fieldUsers.value()) {
        if (FieldWriteOp writeOp = llvm::dyn_cast<FieldWriteOp>(symUse.getUser())) {
          Type writeToType = writeOp.getVal().getType();
          if (newType) {
            // If a new type has already been discovered from another FieldWriteOp, check if they
            // match and fail the conversion if they do not. There should only be one write for each
            // field of a struct but do not rely on that assumption for correctness here.f
            if (writeToType != newType) {
              LLVM_DEBUG(op.emitRemark()
                             .append("Cannot update type of FieldDefOp because there are "
                                     "multiple FieldWriteOp with different value types")
                             .attachNote(writeOp.getLoc())
                             .append("one write is located here"));
              return failure();
            }
          } else if (writeToType != fieldDefType) {
            // If a new type has not been discovered yet and the current FieldWriteOp has a
            // different type from the FieldDefOp, then store the new type to use in the end.
            newType = writeToType;
            LLVM_DEBUG(
                llvm::dbgs() << "[UpdateFieldTypeFromWrite] found new type in " << writeOp << "\n"
            );
          }
        }
      }
    }
    if (!newType) {
      // nothing changed
      return failure();
    }
    if (!tracker_.isLegalConversion(op.getType(), newType, "UpdateFieldTypeFromWrite")) {
      return failure();
    }
    LLVM_DEBUG(llvm::dbgs() << "[UpdateFieldTypeFromWrite] replaced " << op);
    FieldDefOp newOp = rewriter.replaceOpWithNewOp<FieldDefOp>(op, op.getSymName(), newType);
    LLVM_DEBUG(llvm::dbgs() << " with " << newOp << "\n");
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
    LLVM_DEBUG(llvm::dbgs() << " with " << *newOp << "\n");
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
    // Collect unique return type lists
    std::optional<OperandRange::type_range> tyFromReturnOp;
    op.walk([&tyFromReturnOp](Operation *p) {
      if (ReturnOp retOp = dyn_cast<ReturnOp>(p)) {
        auto currReturnType = retOp.getOperands().getTypes();
        // If a type was found from another ReturnOp, make sure it matches the current or give up.
        if (tyFromReturnOp && currReturnType != *tyFromReturnOp) {
          tyFromReturnOp = std::nullopt;
          return WalkResult::interrupt();
        }
        // Otherwise, keep track of the current one and continue.
        tyFromReturnOp = currReturnType;
      }
      return WalkResult::advance();
    });

    if (!tyFromReturnOp) {
      return failure();
    }
    FunctionType oldFuncTy = op.getFunctionType();
    if (oldFuncTy.getResults() == *tyFromReturnOp) {
      // nothing changed
      return failure();
    }
    if (!tracker_.areLegalConversions(
            oldFuncTy.getResults(), *tyFromReturnOp, "UpdateFuncTypeFromReturn"
        )) {
      return failure();
    }

    rewriter.modifyOpInPlace(op, [&]() {
      op.setFunctionType(rewriter.getFunctionType(oldFuncTy.getInputs(), *tyFromReturnOp));
    });
    LLVM_DEBUG(
        llvm::dbgs() << "[UpdateFuncTypeFromReturn] changed " << op.getSymName() << " from "
                     << oldFuncTy << " to " << op.getFunctionType() << "\n"
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
    if (succeeded(getParentOfType<StructDefOp>(targetFunc))) {
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
    CallOp newOp = rewriter.replaceOpWithNewOp<CallOp>(op, targetFunc, op.getArgOperands());
    LLVM_DEBUG(llvm::dbgs() << " with " << newOp << "\n");
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
    StructType oldComputeRetTy = op.getComputeSingleResultType();
    LLVM_DEBUG({
      llvm::dbgs() << "[InstantiateAtCallOpCompute] target: " << op.getCallee() << "\n";
      llvm::dbgs() << "[InstantiateAtCallOpCompute]   oldComputeRetTy: " << oldComputeRetTy << "\n";
    });
    ArrayAttr params = oldComputeRetTy.getParams();
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
                     << debug::toStringList(out.paramsOfStructTy) << "\n";
      });
    }

    StructType newComputeRetTy =
        StructType::get(oldComputeRetTy.getNameRef(), out.paramsOfStructTy);
    if (newComputeRetTy == oldComputeRetTy) {
      // nothing changed
      return failure();
    }
    // ASSERT: instantiateViaTargetType() only preserves the original Attribute or converts to a
    // concrete attribute via the unification process.
    assert(
        tracker_.isLegalConversion(oldComputeRetTy, newComputeRetTy, "InstantiateAtCallOpCompute")
    );
    LLVM_DEBUG(
        llvm::dbgs() << "[InstantiateAtCallOpCompute] instantiating " << oldComputeRetTy << " as "
                     << newComputeRetTy << " in \"" << op << "\"\n"
    );
    rewriter.replaceOpWithNewOp<CallOp>(
        op, TypeRange {newComputeRetTy}, op.getCallee(),
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
      llvm::dbgs() << "[instantiateViaTargetType] in.paramsOfStructTy = "
                   << debug::toStringList(in.paramsOfStructTy) << "\n";
      llvm::dbgs() << "[instantiateViaTargetType] targetResTyParams = "
                   << debug::toStringList(targetResTyParams) << "\n";
      llvm::dbgs() << "[instantiateViaTargetType] target.argtypes = "
                   << debug::toStringList(targetFunc.getArgumentTypes()) << "\n";
      llvm::dbgs() << "[instantiateViaTargetType] callArgTypes = "
                   << debug::toStringList(callArgTypes) << "\n";
    });

    UnificationMap unifications;
    bool unifies = typeListsUnify(targetFunc.getArgumentTypes(), callArgTypes, {}, &unifications);
    assert(unifies && "should have been checked by verifiers");

    LLVM_DEBUG({
      llvm::dbgs() << "[instantiateViaTargetType] unifications = "
                   << debug::toStringList(unifications) << "\n";
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
        assert(llvm::isa<SymbolRefAttr>(fromTgt));
        auto it = unifications.find(std::make_pair(llvm::cast<SymbolRefAttr>(fromTgt), Side::LHS));
        if (it != unifications.end()) {
          Attribute unifiedAttr = it->second;
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
  MLIRContext *ctx = modOp->getContext();
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

  bool currStepModified = false;
  auto result = applyPatternsAndFoldGreedily(
      modOp->getRegion(0), std::move(patterns), GreedyRewriteConfig(), &currStepModified
  );
  tracker.updateModifiedFlag(currStepModified);
  return result;
}

} // namespace Step4_InstantiateAffineMaps

namespace Step5_Cleanup {

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

  // Peform the conversion, i.e. remove StructDefOp that were instantiated and are unused.
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

} // namespace Step5_Cleanup

class InstantiateStructsPass
    : public llzk::impl::InstantiateStructsPassBase<InstantiateStructsPass> {

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
      if (failed(Step1_FindComputeTypes::run(modOp, tracker))) {
        llvm::errs() << DEBUG_TYPE << " failed while replacing concrete-parameter struct types\n";
        signalPassFailure();
        break;
      }

      // Create the necessary instantiated/flattened struct(s) in their parent module(s).
      if (failed(Step2_CreateStructs::run(modOp, tracker))) {
        llvm::errs() << DEBUG_TYPE << " failed while generating required flattened structs\n";
        signalPassFailure();
        break;
      }

      // Unroll loops with known iterations.
      if (failed(Step3_Unroll::run(modOp, tracker))) {
        llvm::errs() << DEBUG_TYPE << " failed while unrolling loops\n";
        signalPassFailure();
        break;
      }

      // Instantiate affine_map parameters of StructType and ArrayType.
      if (failed(Step4_InstantiateAffineMaps::run(modOp, tracker))) {
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
    if (failed(Step5_Cleanup::run(modOp, tracker))) {
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

std::unique_ptr<Pass> llzk::createFlatteningPass() {
  return std::make_unique<InstantiateStructsPass>();
};
