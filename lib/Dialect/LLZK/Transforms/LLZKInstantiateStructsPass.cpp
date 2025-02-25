#include "llzk/Dialect/LLZK/IR/Dialect.h"
#include "llzk/Dialect/LLZK/IR/Ops.h"
#include "llzk/Dialect/LLZK/Transforms/LLZKTransformationPasses.h"
#include "llzk/Dialect/LLZK/Util/AttributeHelper.h"
#include "llzk/Dialect/LLZK/Util/SymbolHelper.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/SCF/Transforms/Patterns.h>
#include <mlir/Dialect/SCF/Utils/Utils.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>

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

#define DEBUG_TYPE "llzk-instantiate-structs"

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
  /// `GeneralTypeReplacePattern` and must have a `OpConversionPattern` defined if they need to be
  /// converted.
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
  // Add builtin FunctionType converter
  populateFunctionOpInterfaceTypeConversionPattern(FuncOp::getOperationName(), patterns, tyConv);
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

namespace Step1 {

bool isConcreteAttr(Attribute a) {
  if (TypeAttr tyAttr = dyn_cast<TypeAttr>(a)) {
    return isConcreteType(tyAttr.getValue());
  }
  return llvm::isa<IntegerAttr>(a);
}

class ParameterizedStructUseTypeConverter : public TypeConverter {
  DenseMap<StructType, StructType> &instantiations;

public:
  ParameterizedStructUseTypeConverter(DenseMap<StructType, StructType> &structInstantiations)
      : TypeConverter(), instantiations(structInstantiations) {

    addConversion([](Type inputTy) { return inputTy; });

    addConversion([this](StructType inputTy) -> StructType {
      // First check for a cached entry
      auto cachedResult = this->instantiations.find(inputTy);
      if (cachedResult != this->instantiations.end()) {
        return cachedResult->second;
      }
      // Otherwise, try to perform a conversion
      if (ArrayAttr params = inputTy.getParams()) {
        // If all prameters are concrete values (Integer or Type), then replace with a no-parameter
        // StructType referencing the de-parameterized struct.
        if (llvm::all_of(params, isConcreteAttr)) {
          StructType result =
              StructType::get(appendLeafName(inputTy.getNameRef(), "_" + shortString(params)));
          LLVM_DEBUG(llvm::dbgs() << "instantiating " << inputTy << " as " << result << "\n");
          this->instantiations[inputTy] = result;
          return result;
        }
      }
      return inputTy;
    });
  }
};

class CallComputePattern : public OpConversionPattern<CallOp> {
  DenseMap<StructType, DenseSet<Location>> &newTyComputeLocs;

public:
  CallComputePattern(
      TypeConverter &converter, MLIRContext *ctx,
      DenseMap<StructType, DenseSet<Location>> &newTyComputeLocations
  )
      // future proof: use higher priority than GeneralTypeReplacePattern
      : OpConversionPattern<CallOp>(converter, ctx, 2), newTyComputeLocs(newTyComputeLocations) {}

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
        newTyComputeLocs[newStTy].insert(op.getLoc());
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

LogicalResult
run(ModuleOp modOp, DenseMap<StructType, StructType> &structInstantiations,
    DenseMap<StructType, DenseSet<Location>> &newTyComputeLocations) {

  MLIRContext *ctx = modOp.getContext();
  ParameterizedStructUseTypeConverter tyConv(structInstantiations);
  ConversionTarget target = newConverterDefinedTarget<>(tyConv, ctx);
  RewritePatternSet patterns = newGeneralRewritePatternSet(tyConv, ctx, target);
  patterns.add<CallComputePattern>(tyConv, ctx, newTyComputeLocations);

  if (failed(applyPartialConversion(modOp, target, std::move(patterns)))) {
    return modOp.emitError("failed to convert all `compute()` calls");
  }
  return success();
}

} // namespace Step1

namespace Step2 {

class MappedTypeConverter : public TypeConverter {
  StructType origTy;
  StructType newTy;
  const DenseMap<Attribute, Attribute> &instantiationMap;

public:
  MappedTypeConverter(
      StructType originalType, StructType newType,
      /// Instantiated values for the parameter names in `origTy`
      const DenseMap<Attribute, Attribute> &nameToValueMap
  )
      : TypeConverter(), origTy(originalType), newTy(newType), instantiationMap(nameToValueMap) {

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
          auto res = this->instantiationMap.find(a);
          updated.push_back((res != this->instantiationMap.end()) ? res->second : a);
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
          auto res = this->instantiationMap.find(a);
          updated.push_back((res != this->instantiationMap.end()) ? res->second : a);
        }
        return ArrayType::get(this->convertType(inputTy.getElementType()), updated);
      }
      // Otherwise, return the type unchanged
      return inputTy;
    });

    addConversion([this](TypeVarType inputTy) -> Type {
      // Check for replacement of parameter symbol name with a concrete type
      auto res = this->instantiationMap.find(inputTy.getNameRef());
      if (res != this->instantiationMap.end()) {
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

class CreateArrayOpPattern : public OpConversionPattern<CreateArrayOp> {
public:
  CreateArrayOpPattern(TypeConverter &converter, MLIRContext *ctx)
      // future proof: use higher priority than GeneralTypeReplacePattern
      : OpConversionPattern<CreateArrayOp>(converter, ctx, 2) {}

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
    assert(llvm::isa<ArrayType>(newType) && "impl out of sync with converter");
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

class ConstReadOpPattern : public OpConversionPattern<ConstReadOp> {
  const DenseMap<Attribute, Attribute> &instantiationMap;
  const DenseSet<Location> &locations;

public:
  ConstReadOpPattern(
      TypeConverter &converter, MLIRContext *ctx,
      const DenseMap<Attribute, Attribute> &nameToValueMap,
      const DenseSet<Location> &instantiationLocations
  )
      // future proof: use higher priority than GeneralTypeReplacePattern
      : OpConversionPattern<ConstReadOp>(converter, ctx, 2), instantiationMap(nameToValueMap),
        locations(instantiationLocations) {}

  LogicalResult matchAndRewrite(
      ConstReadOp op, OpAdaptor adapter, ConversionPatternRewriter &rewriter
  ) const override {
    auto res = this->instantiationMap.find(op.getConstNameAttr());
    if (res == this->instantiationMap.end()) {
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
            for (Location loc : locations) {
              warning.attachNote(loc).append(
                  "when instantiating ", StructDefOp::getOperationName(), " parameter \"",
                  res->first, "\" for this call"
              );
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

LogicalResult
run(ModuleOp modOp, const DenseMap<StructType, StructType> &structInstantiations,
    DenseMap<StructType, DenseSet<Location>> &newTyComputeLocations) {

  SymbolTableCollection symTables;
  MLIRContext *ctx = modOp.getContext();
  for (auto &[origRemoteTy, newRemoteTy] : structInstantiations) {
    // Find the StructDefOp for the original StructType
    FailureOr<SymbolLookupResult<StructDefOp>> lookupRes =
        origRemoteTy.getDefinition(symTables, modOp);
    if (failed(lookupRes)) {
      return failure(); // getDefinition() already emits a sufficient error message
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
      patterns.add<CallOpPattern, CreateArrayOpPattern>(tyConv, ctx);
      patterns.add<ConstReadOpPattern>(
          tyConv, ctx, nameToValueMap, newTyComputeLocations[newRemoteTy]
      );

      if (failed(applyFullConversion(newStruct, target, std::move(patterns)))) {
        return modOp.emitError("failed to generate all required flattened structs");
      }

      // Insert 'newStruct' into the parent ModuleOp of the original StructDefOp.
      parentModule.insert(origStruct, newStruct);
    }
  }

  return success();
}

} // namespace Step2

namespace Step3 {

template <typename OpTy> class EraseOpPattern : public OpConversionPattern<OpTy> {
public:
  EraseOpPattern(MLIRContext *ctx) : OpConversionPattern<OpTy>(ctx) {}

  LogicalResult
  matchAndRewrite(OpTy op, OpTy::Adaptor, ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

LogicalResult run(ModuleOp modOp, const DenseMap<StructType, StructType> &structInstantiations) {
  FailureOr<ModuleOp> topRoot = getTopRootModule(modOp);
  if (failed(topRoot)) {
    return failure();
  }

  // Collect the fully-qualified names of all structs that were instantiated.
  DenseSet<SymbolRefAttr> instantiatedNames;
  for (const auto &[origRemoteTy, _] : structInstantiations) {
    instantiatedNames.insert(origRemoteTy.getNameRef());
  }

  // Use a conversion to erase those structs if they have no other references.
  //
  // TODO: there's a chance the "no other references" criteria will leave some behind when running
  // only a single pass of this because they may reference each other. Maybe I can check if the
  // references are only located within another struct in the list, but would have to do a deep
  // deep lookup to ensure no references and avoid infinite loop back on self.
  //
  MLIRContext *ctx = modOp.getContext();
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
  RewritePatternSet patterns(ctx);
  patterns.add<EraseOpPattern<StructDefOp>>(ctx);
  ConversionTarget target = newBaseTarget(ctx);
  target.addDynamicallyLegalOp<StructDefOp>(std::bind_front(isLegalStruct, false));
  if (failed(applyFullConversion(modOp, target, std::move(patterns)))) {
    return modOp.emitError("failed to remove parameterized structs that were instantiated");
  }

  // Warn about any structs that were instantiated but still have uses elsewhere.
  modOp->walk([&](StructDefOp op) {
    isLegalStruct(true, op);
    return WalkResult::skip(); // StructDefOp cannot be nested
  });

  return success();
}

} // namespace Step3

class InstantiateStructsPass
    : public llzk::impl::InstantiateStructsPassBase<InstantiateStructsPass> {
  void runOnOperation() override {
    ModuleOp modOp = getOperation();

    unsigned prevMapSize;
    // Maps original remote (i.e. use site) type to new remote type
    DenseMap<StructType, StructType> structInstantiations;
    // Maps new remote type to location of the compute() calls that cause instantiation
    DenseMap<StructType, DenseSet<Location>> newTyComputeLocations;
    do {
      prevMapSize = structInstantiations.size();

      // Find calls to "compute()" that return a parameterized struct and replace it to call a
      // flattened version of the struct that has parameters replaced with the constant values.
      if (failed(Step1::run(modOp, structInstantiations, newTyComputeLocations))) {
        signalPassFailure();
        return;
      }

      // Create the necessary instantiated/flattened struct(s) in their parent module(s).
      if (failed(Step2::run(modOp, structInstantiations, newTyComputeLocations))) {
        signalPassFailure();
        return;
      }

      // Repeat as long as a new type was requested
    } while (structInstantiations.size() > prevMapSize);

    // Remove the parameterized StructDefOp that were instantiated.
    if (failed(Step3::run(modOp, structInstantiations))) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace

std::unique_ptr<Pass> llzk::createInstantiateStructsPass() {
  return std::make_unique<InstantiateStructsPass>();
};
