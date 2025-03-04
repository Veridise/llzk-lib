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

namespace Step1_Unroll {

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

LogicalResult run(ModuleOp modOp, bool &modified) {
  MLIRContext *ctx = modOp->getContext();
  RewritePatternSet patterns(ctx);
  patterns.add<LoopUnrollPattern<scf::ForOp>>(ctx);
  patterns.add<LoopUnrollPattern<affine::AffineForOp>>(ctx);
  return applyPatternsAndFoldGreedily(
      modOp->getRegion(0), std::move(patterns), GreedyRewriteConfig(), &modified
  );
}
} // namespace Step1_Unroll

namespace Step2_InstantiateAffineMaps {

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
        LLVM_DEBUG(llvm::dbgs() << "currMapOps: " << debug::toStringList(currMapOps) << "\n");
        SmallVector<OpFoldResult> currMapOpsCast = getAsOpFoldResult(currMapOps);
        LLVM_DEBUG(
            llvm::dbgs() << "  currMapOps as fold results: " << debug::toStringList(currMapOpsCast)
                         << "\n"
        );
        if (auto constOps = Step2_InstantiateAffineMaps::getConstantIntValues(currMapOpsCast)) {
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
struct InstantiateAtCreateArrayOp final : public OpRewritePattern<CreateArrayOp> {
  using OpRewritePattern<CreateArrayOp>::OpRewritePattern;

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
    LLVM_DEBUG(
        llvm::dbgs() << "[InstantiateAtCreateArrayOp] instantiated " << oldResultType << " as "
                     << newResultType << " at " << op << "\n"
    );
    rewriter.replaceOpWithNewOp<CreateArrayOp>(
        op, newResultType, AffineMapFolder::getConvertedMapOpGroups(out), out.dimsPerGroup
    );
    return success();
  }
};

/// Update the array element type by looking at the values stored into it from uses.
struct UpdateArrayElemFromWrite final : public OpRewritePattern<CreateArrayOp> {
  using OpRewritePattern<CreateArrayOp>::OpRewritePattern;

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

    ArrayType newType = createResultType.cloneWith(newResultElemType);
    rewriter.modifyOpInPlace(op, [&createResult, &newType]() { createResult.setType(newType); });
    LLVM_DEBUG(llvm::dbgs() << "[UpdateArrayElemFromWrite] updated result type of " << op << "\n");
    return success();
  }
};

/// Update the type of FieldDefOp instances by checking the updated types from FieldWriteOp.
struct UpdateFieldTypeFromWrite final : public OpRewritePattern<FieldDefOp> {
  using OpRewritePattern<FieldDefOp>::OpRewritePattern;

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
    LLVM_DEBUG(llvm::dbgs() << "[UpdateFieldTypeFromWrite] replaced " << op);
    FieldDefOp newOp = rewriter.replaceOpWithNewOp<FieldDefOp>(op, op.getSymName(), newType);
    LLVM_DEBUG(llvm::dbgs() << " with " << newOp << "\n");
    return success();
  }
};

/// Updates the result type in Ops with the InferTypeOpAdaptor trait including ReadArrayOp,
/// ExtractArrayOp, etc.
struct UpdateInferredResultTypes final : public OpTraitRewritePattern<OpTrait::InferTypeOpAdaptor> {
  using OpTraitRewritePattern::OpTraitRewritePattern;
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
struct UpdateFuncTypeFromReturn final : public OpRewritePattern<FuncOp> {
  using OpRewritePattern<FuncOp>::OpRewritePattern;

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
struct UpdateGlobalCallOpTypes final : public OpRewritePattern<CallOp> {
  using OpRewritePattern<CallOp>::OpRewritePattern;

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

    LLVM_DEBUG(llvm::dbgs() << "[UpdateGlobalCallOpTypes] replaced " << op);
    CallOp newOp = rewriter.replaceOpWithNewOp<CallOp>(op, targetFunc, op.getArgOperands());
    LLVM_DEBUG(llvm::dbgs() << " with " << newOp << "\n");
    return success();
  }
};

/// Instantiate parameterized StructType resulting from CallOp targeting "compute()" functions.
struct InstantiateAtCallOpCompute final : public OpRewritePattern<CallOp> {
  using OpRewritePattern<CallOp>::OpRewritePattern;

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
        llvm::dbgs() << "[InstantiateAtCallOpCompute]   propagated symrefs in result type params\n";
      });
    }

    StructType newComputeRetTy =
        StructType::get(oldComputeRetTy.getNameRef(), out.paramsOfStructTy);
    if (newComputeRetTy == oldComputeRetTy) {
      // nothing changed
      return failure();
    }
    LLVM_DEBUG(
        llvm::dbgs() << "[InstantiateAtCallOpCompute] instantiated " << oldComputeRetTy << " as "
                     << newComputeRetTy << " at " << op << "\n"
    );
    rewriter.replaceOpWithNewOp<CallOp>(
        op, TypeRange {newComputeRetTy}, op.getCallee(),
        AffineMapFolder::getConvertedMapOpGroups(out), out.dimsPerGroup, op.getArgOperands()
    );
    return success();
  }

private:
  inline LogicalResult instantiateViaTargetType(
      const AffineMapFolder::Input &in, AffineMapFolder::Output &out,
      OperandRange::type_range callArgTypes, FuncOp targetFunc
  ) const {
    assert(targetFunc.isStructCompute()); // since `op.calleeIsStructCompute()`
    ArrayAttr targetResTyParams = targetFunc.getComputeSingleResultType().getParams();
    assert(!isNullOrEmpty(targetResTyParams)); // same cardinality as `in.paramsOfStructTy`
    assert(in.paramsOfStructTy.size() == targetResTyParams.size()); // verifier ensures this

    // Initialize the updated return StructType parameter list where each index uses the CallOp
    // StructType parameter if concrete, otherwise using the target struct type parameter.
    bool hasTargetStructParams = false;
    SmallVector<Attribute> newReturnStructParams = llvm::map_to_vector(
        llvm::zip_equal(in.paramsOfStructTy, targetResTyParams.getValue()),
        [&hasTargetStructParams](std::tuple<Attribute, Attribute> p) {
      Attribute fromCall = std::get<0>(p);
      if (isConcreteAttr<>(fromCall)) {
        return fromCall;
      } else {
        hasTargetStructParams = true;
        return std::get<1>(p);
      }
    }
    );
    if (!hasTargetStructParams) {
      // Nothing will change if no target parameters are used
      return failure();
    }

    LLVM_DEBUG({
      llvm::dbgs() << "in.paramsOfStructTy = " << debug::toStringList(in.paramsOfStructTy) << "\n";
      llvm::dbgs() << "targetResTyParams = " << debug::toStringList(targetResTyParams) << "\n";
      llvm::dbgs() << "newReturnStructParams (init) = "
                   << debug::toStringList(newReturnStructParams) << "\n";
    });

    UnificationMap unifications;
    bool unifies = typeListsUnify(targetFunc.getArgumentTypes(), callArgTypes, {}, &unifications);
    assert(unifies && "should have been checked by verifiers");

    LLVM_DEBUG(llvm::dbgs() << "unifications = " << debug::toStringList(unifications) << "\n");

    // Check for LHS SymRef that have RHS concrete Attributes without any struct parameters (because
    // a call with concrete struct parameters will be replaced elsewhere and doing it here would
    // interfere and result in a type mismatch) and perform those replacements in the `targetFunc`
    // return type to produce the new result type for the CallOp.
    for (Attribute &a : newReturnStructParams) {
      if (SymbolRefAttr symRef = llvm::dyn_cast<SymbolRefAttr>(a)) {
        auto it = unifications.find(std::make_pair(symRef, Side::LHS));
        if (it != unifications.end()) {
          Attribute unifiedAttr = it->second;
          if (unifiedAttr && isConcreteAttr<false>(unifiedAttr)) {
            a = unifiedAttr;
          }
        }
      }
    }

    out.paramsOfStructTy = newReturnStructParams;
    assert(out.paramsOfStructTy.size() == in.paramsOfStructTy.size() && "post-condition");
    assert(out.mapOpGroups.empty() && "post-condition");
    assert(out.dimsPerGroup.empty() && "post-condition");
    return success();
  }
};

LogicalResult run(ModuleOp modOp, bool &modified) {
  MLIRContext *ctx = modOp->getContext();
  RewritePatternSet patterns(ctx);
  patterns.add<InstantiateAtCreateArrayOp>(ctx);
  patterns.add<UpdateFieldTypeFromWrite>(ctx);
  patterns.add<UpdateInferredResultTypes>(ctx);
  patterns.add<UpdateFuncTypeFromReturn>(ctx);
  patterns.add<UpdateGlobalCallOpTypes>(ctx);
  patterns.add<InstantiateAtCallOpCompute>(ctx);
  patterns.add<UpdateArrayElemFromWrite>(ctx);
  return applyPatternsAndFoldGreedily(
      modOp->getRegion(0), std::move(patterns), GreedyRewriteConfig(), &modified
  );
}

} // namespace Step2_InstantiateAffineMaps

namespace Step3_FindComputeTypes {

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
        // If all prameters are concrete values (Integer or Type), then replace with a
        // no-parameter StructType referencing the de-parameterized struct.
        if (llvm::all_of(params, isConcreteAttr<>)) {
          StructType result =
              StructType::get(appendLeafName(inputTy.getNameRef(), "_" + shortString(params)));
          LLVM_DEBUG(
              llvm::dbgs() << "[ParameterizedStructUseTypeConverter] instantiating " << inputTy
                           << " as " << result << "\n"
          );
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

} // namespace Step3_FindComputeTypes

namespace Step4_CreateStructs {

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

} // namespace Step4_CreateStructs

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
  // TODO: There's a chance the "no other references" criteria will leave some behind when running
  // only a single pass of this because they may reference each other. Maybe I can check if the
  // references are only located within another struct in the list, but would have to do a deep
  // deep lookup to ensure no references and avoid infinite loop back on self.
  // TODO: There's another scenario that leaves some behind. Once a StructDefOp is visited and
  // considered legal, that decision cannot be reversed. Hence, StructDefOp that become illegal only
  // after removing another one that uses it will not be removed. See
  // test/Dialect/LLZK/instantiate_structs_affine_pass.llzk
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

} // namespace Step5_Cleanup

class InstantiateStructsPass
    : public llzk::impl::InstantiateStructsPassBase<InstantiateStructsPass> {
  void runOnOperation() override {
    ModuleOp modOp = getOperation();

    bool modified;
    // Maps original remote (i.e. use site) type to new remote type
    DenseMap<StructType, StructType> structInstantiations;
    // Maps new remote type to location of the compute() calls that cause instantiation
    DenseMap<StructType, DenseSet<Location>> newTyComputeLocations;
    do {
      modified = false;

      // Unroll loops with known iterations
      if (failed(Step1_Unroll::run(modOp, modified))) {
        signalPassFailure();
        return;
      }

      // Instantiate affine_map parameters of StructType and ArrayType
      if (failed(Step2_InstantiateAffineMaps::run(modOp, modified))) {
        signalPassFailure();
        return;
      }

      unsigned prevMapSize = structInstantiations.size();

      // Find calls to "compute()" that return a parameterized struct and replace it to call a
      // flattened version of the struct that has parameters replaced with the constant values.
      if (failed(Step3_FindComputeTypes::run(modOp, structInstantiations, newTyComputeLocations))) {
        signalPassFailure();
        return;
      }

      // Create the necessary instantiated/flattened struct(s) in their parent module(s).
      if (failed(Step4_CreateStructs::run(modOp, structInstantiations, newTyComputeLocations))) {
        signalPassFailure();
        return;
      }

      // Check if a new StructType was required by an instantiation
      modified |= structInstantiations.size() > prevMapSize;

      LLVM_DEBUG(if (modified) {
        llvm::dbgs() << "=====================================================================\n";
        llvm::dbgs() << " Dumping module between iterations of " << DEBUG_TYPE << " \n";
        modOp.print(llvm::dbgs());
        llvm::dbgs() << "=====================================================================\n";
      });
    } while (modified);

    // Remove the parameterized StructDefOp that were instantiated.
    if (failed(Step5_Cleanup::run(modOp, structInstantiations))) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace

std::unique_ptr<Pass> llzk::createFlatteningPass() {
  return std::make_unique<InstantiateStructsPass>();
};
