//===-- LLZKInlineStructsPass.cpp -------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the `-llzk-inline-structs` pass.
///
//===----------------------------------------------------------------------===//

#include "llzk/Analysis/GraphUtil.h"
#include "llzk/Analysis/SymbolDefTree.h"
#include "llzk/Analysis/SymbolUseGraph.h"
#include "llzk/Dialect/Constrain/IR/Ops.h"
#include "llzk/Dialect/Felt/IR/Ops.h"
#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Dialect/Struct/IR/Ops.h"
#include "llzk/Transforms/LLZKTransformationPasses.h"
#include "llzk/Transforms/WalkPatternRewriteDriver.h"
#include "llzk/Util/Debug.h"
#include "llzk/Util/SymbolHelper.h"
#include "llzk/Util/SymbolLookup.h"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Transforms/InliningUtils.h>

#include <llvm/ADT/PostOrderIterator.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Debug.h>

// Include the generated base pass class definitions.
namespace llzk {
// the *DECL* macro is required when a pass has options to declare the option struct
#define GEN_PASS_DECL_INLINESTRUCTSPASS
#define GEN_PASS_DEF_INLINESTRUCTSPASS
#include "llzk/Transforms/LLZKTransformationPasses.h.inc"
} // namespace llzk

using namespace mlir;
using namespace llzk;
using namespace llzk::component;
using namespace llzk::function;

#define DEBUG_TYPE "llzk-inline-structs"

namespace {

// TODO: this will be available in `llzk/Transforms/LLZKLoweringUtils.h` from R1CS PR.
// TODO: I added the assert at the start though.
// TODO: also changed `body.back()` but made that same comment in Shankara's open PR.
Value getSelfValueFromComputeDef(FuncDefOp computeFunc) {
  assert(computeFunc.nameIsCompute()); // skip inStruct check to allow dangling functions
  // Get the single block of the function body
  Region &body = computeFunc.getBody();
  assert(!body.empty() && "compute() function body is empty");

  Block &block = body.back();

  // The terminator should be the return op
  Operation *terminator = block.getTerminator();
  assert(terminator && "compute() function has no terminator");

  // The return op should be of type ReturnOp
  auto retOp = dyn_cast<ReturnOp>(terminator);
  if (!retOp) {
    llvm::errs() << "Expected ReturnOp as terminator in compute() but found: "
                 << terminator->getName() << "\n";
    llvm_unreachable("compute() function terminator is not a ReturnOp");
  }

  return retOp.getOperands().front();
}

// TODO: this should move to `llzk/Transforms/LLZKLoweringUtils.h` after R1CS PR.
Value getSelfValueFromConstrainDef(FuncDefOp constrainFunc) {
  assert(constrainFunc.nameIsConstrain()); // skip inStruct check to allow dangling functions
  return constrainFunc.getArguments().front();
}

Value getSelfValue(FuncDefOp f) {
  if (f.nameIsCompute()) {
    return getSelfValueFromComputeDef(f);
  } else if (f.nameIsConstrain()) {
    return getSelfValueFromConstrainDef(f);
  } else {
    llvm_unreachable("expected \"compute\" or \"constrain\" function");
  }
}

// TODO: this should move to `llzk/Transforms/LLZKLoweringUtils.h` after R1CS PR.
Value getSelfValueFromComputeCall(CallOp callToCompute) {
  assert(callToCompute.calleeIsStructCompute());
  return callToCompute.getResults().front();
}

// TODO: this should move to `llzk/Transforms/LLZKLoweringUtils.h` after R1CS PR.
Value getSelfValueFromConstrainCall(CallOp callToConstrain) {
  assert(callToConstrain.calleeIsStructConstrain());
  return callToConstrain.getArgOperands().front();
}

class StructInlinerBase {
public:
  /// Mapping of `destStruct` field that has `srcStruct` type to each FieldDefOp from `srcStruct` to
  /// the cloned version of that `srcStruct` field within `destStruct`.
  using DestFieldWithSrcStructType = FieldDefOp;
  using DestCloneOfSrcStructField = FieldDefOp;
  using SrcStructFieldToCloneInDest = DenseMap<StringAttr, DestCloneOfSrcStructField>;
  using DestToSrcToClonedSrcInDest =
      DenseMap<DestFieldWithSrcStructType, SrcStructFieldToCloneInDest>;

protected:
  SymbolTableCollection &tables;

  StructInlinerBase(SymbolTableCollection &symTables) : tables(symTables) {}

  inline FieldDefOp getDef(FieldRefOpInterface fRef) const {
    auto r = fRef.getFieldDefOp(tables);
    assert(succeeded(r));
    return r->get();
  }

  LogicalResult handleCall(
      const DestToSrcToClonedSrcInDest &destToSrcToClone, FuncDefOp srcFunc, FuncDefOp destFunc,
      llvm::function_ref<FieldRefOpInterface(CallOp)> getSelfRef
  ) {
    InlinerInterface inliner(destFunc.getContext());

    // Find any CallOp within `destFunc` that target `srcFunc` and inline them.
    auto walkRes = destFunc.getBody().walk<WalkOrder::PreOrder>([&](CallOp callOp) {
      // Ensure the CallOp targets `srcFunc`
      auto callOpTarget = callOp.getCalleeTarget(tables);
      assert(succeeded(callOpTarget));
      if (callOpTarget->get() != srcFunc) {
        return WalkResult::skip(); // Don't go in. CallOp are not nested.
      }

      // Get the "self" struct value from the CallOp and determine which struct field is used.
      FieldRefOpInterface selfFieldRefOp = getSelfRef(callOp);
      if (!selfFieldRefOp) {
        return WalkResult::interrupt(); // use interrupt to signal failure
      }

      // Create a clone of the source function (must do the whole function not just the body region
      // because `inlineCall()` expects the Region to have a parent op) and update field references
      // to the old struct fields to instead use the new struct fields.
      FuncDefOp srcFuncClone =
          FieldRefRewriter::cloneWithFieldRefUpdate(std::make_unique<FieldRefRewriter>(
              srcFunc, selfFieldRefOp.getComponent(), destToSrcToClone.at(getDef(selfFieldRefOp))
          ));

      // Inline the cloned function in place of `callOp`
      LogicalResult inlineCallRes =
          inlineCall(inliner, callOp, srcFuncClone, &srcFuncClone.getBody(), false);
      if (failed(inlineCallRes)) {
        return WalkResult::interrupt(); // use interrupt to signal failure
      }
      srcFuncClone.erase();      // delete what's left after transferring the body elsewhere
      callOp.erase();            // delete the original CallOp
      return WalkResult::skip(); // Must skip because the CallOp was erased.
    });

    return failure(walkRes.wasInterrupted());
  }

private:
  // Update field read/write ops that target the "self" value of `withinFunction` and some key in
  // `oldToNewFieldDef` to instead target `newBaseVal` and the mapped value from `oldToNewFieldDef`.
  // Example:
  //  old:  %1 = struct.readf %0[@f1] : <@Component1A>, !felt.type
  //  new:  %1 = struct.readf %self[@"f2:!s<@Component1A>+f1"] : <@Component1B>, !felt.type
  class FieldRefRewriter final : public OpInterfaceRewritePattern<FieldRefOpInterface> {
    /// This is initially the `originalFunc` parameter from the constructor but after the clone is
    /// created within `cloneWithFieldRefUpdate()`, it is reassigned to the cloned function.
    FuncDefOp funcRef;
    /// The "self" value in the cloned function.
    Value oldBaseVal;
    /// The new base value for updated field references.
    Value newBaseVal;
    const SrcStructFieldToCloneInDest &oldToNewFields;

  public:
    FieldRefRewriter(
        FuncDefOp originalFunc, Value newRefBase,
        const SrcStructFieldToCloneInDest &oldToNewFieldDef
    )
        : OpInterfaceRewritePattern(originalFunc.getContext()), funcRef(originalFunc),
          oldBaseVal(nullptr), newBaseVal(newRefBase), oldToNewFields(oldToNewFieldDef) {}

    LogicalResult match(FieldRefOpInterface op) const final {
      assert(oldBaseVal); // ensure it's used via `cloneWithFieldRefUpdate()` only
      // Check if the FieldRef accesses a field of "self" within the `oldToNewFields` map.
      // Per `cloneWithFieldRefUpdate()`, `oldBaseVal` is the "self" value of `funcRef` so
      // check for a match there and then check that the referenced field name is in the map.
      return success(
          op.getComponent() == oldBaseVal &&
          oldToNewFields.contains(op.getFieldNameAttr().getAttr())
      );
    }

    void rewrite(FieldRefOpInterface op, PatternRewriter &rewriter) const final {
      rewriter.modifyOpInPlace(op, [this, &op]() {
        FieldDefOp newF = oldToNewFields.at(op.getFieldNameAttr().getAttr());
        op.setFieldName(newF.getSymName());
        op.getComponentMutable().set(this->newBaseVal);
      });
    }

    /// Create a clone of the`FuncDefOp` and update field references according to the
    /// `SrcStructFieldToCloneInDest` map (both are within the given `FieldRefRewriter`).
    static FuncDefOp cloneWithFieldRefUpdate(std::unique_ptr<FieldRefRewriter> thisPat) {
      IRMapping mapper;
      FuncDefOp srcFuncClone = thisPat->funcRef.clone(mapper);
      // Update some data in the `FieldRefRewriter` instance before moving it.
      thisPat->funcRef = srcFuncClone;
      thisPat->oldBaseVal = getSelfValue(srcFuncClone);
      // Run the rewriter to replace read/write ops
      MLIRContext *ctx = thisPat->getContext();
      RewritePatternSet patterns(ctx, std::move(thisPat));
      walkAndApplyPatterns(srcFuncClone, std::move(patterns));

      return srcFuncClone;
    }
  };
};

class StructInliner : public StructInlinerBase {
  /// The struct that will be inlined (and maybe removed).
  StructDefOp srcStruct;
  /// The Struct whose body will be augmented with the inlined content.
  StructDefOp destStruct;

  // Find any field(s) in `destStruct` whose type matches `srcStruct` (allowing any parameters, if
  // applicable). For each such field, clone all fields from `srcStruct` into `destStruct` and cache
  // the mapping of `destStruct` to `srcStruct` to cloned fields in the return value.
  DestToSrcToClonedSrcInDest cloneFields() {
    DestToSrcToClonedSrcInDest destToSrcToClone;

    SymbolTable &destStructSymTable = tables.getSymbolTable(destStruct);
    StructType srcStructType = srcStruct.getType();
    for (FieldDefOp destField : destStruct.getFieldDefs()) {
      if (StructType destFieldType = llvm::dyn_cast<StructType>(destField.getType())) {
        UnificationMap unifications;
        if (!structTypesUnify(srcStructType, destFieldType, {}, &unifications)) {
          continue;
        }
        assert(unifications.empty() && "TODO: handle this!");
        SrcStructFieldToCloneInDest &srcToClone = destToSrcToClone.getOrInsertDefault(destField);
        // Clone each field from 'srcStruct' into 'destStruct'
        auto srcFields = srcStruct.getFieldDefs();
        if (srcFields.empty()) {
          continue;
        }
        OpBuilder builder(destField);
        std::string newNameBase =
            destField.getName().str() + ':' + BuildShortTypeString::from(destFieldType);
        for (FieldDefOp srcField : srcFields) {
          FieldDefOp srcFieldClone = llvm::cast<FieldDefOp>(builder.clone(*srcField));
          srcFieldClone.setName(builder.getStringAttr(newNameBase + '+' + srcFieldClone.getName()));
          srcToClone[srcField.getSymNameAttr()] = srcFieldClone;
          // Also update the cached SymbolTable
          destStructSymTable.insert(srcFieldClone);
        }
      }
    }
    return destToSrcToClone;
  }

  /// Inline the "constrain" function from `srcStruct` into `destStruct`.
  LogicalResult handleConstrainCall(const DestToSrcToClonedSrcInDest &destToSrcToClone) {
    return handleCall(
        destToSrcToClone, srcStruct.getConstrainFuncOp(), destStruct.getConstrainFuncOp(),
        [destStruct = &this->destStruct](CallOp callOp) {
      // The typical pattern is to read a struct instance from a field and then call "constrain()"
      // on it. Get the Value passed as the "self" struct to the CallOp and determine which field it
      // was read from in the current struct (i.e., `destStruct`).
      Value selfArgFromCall = getSelfValueFromConstrainCall(callOp);
      FieldRefOpInterface selfFieldRefOp =
          llvm::dyn_cast_if_present<FieldReadOp>(selfArgFromCall.getDefiningOp());
      if (selfFieldRefOp && selfFieldRefOp.getComponent().getType() == destStruct->getType()) {
        return selfFieldRefOp;
      }
      // TODO: There is a possibility that this value is not from a field read (ex: it could be a
      // parameter to the `destStruct` function). That will not have a mapping in `destToSrcToClone`
      // and new fields will still need to be added, they can be prefixed with parameter index since
      // there is no current field name to use as the unique prefix.
      //
      // TODO: also, in that case, the signature of this lambda is insufficient. If there's no
      // field involved we need to instead return the two relevant pieces, the "self" Value and
      // the FieldDefOp created within `destStruct`.
      //
      llvm::errs() << "[TODO] callOp = " << callOp << '\n';
      llvm::errs() << "[TODO] selfArgFromCall = " << selfArgFromCall << '\n';
      assert(false && "TODO: handle this!");
    }
    );
  }

  /// Inline the "compute" function from `srcStruct` into `destStruct`.
  LogicalResult handleComputeCall(const DestToSrcToClonedSrcInDest &destToSrcToClone) {
    return handleCall(
        destToSrcToClone, srcStruct.getComputeFuncOp(), destStruct.getComputeFuncOp(),
        [destStruct = &this->destStruct](CallOp callOp) {
      // The typical pattern is to write the return value of "compute()" to a field in
      // the current struct (i.e., `destStruct`).
      FieldRefOpInterface selfFieldRefOp = nullptr;
      Value selfArgFromCall = getSelfValueFromComputeCall(callOp);
      for (OpOperand &use : selfArgFromCall.getUses()) {
        if (auto writeOp = llvm::dyn_cast<FieldWriteOp>(use.getOwner())) {
          // ASSERT: FieldWriteOp are only allowed to write to the current struct.
          assert(writeOp.getComponent().getType() == destStruct->getType());
          if (selfFieldRefOp) {
            selfFieldRefOp = nullptr;
            break;
          } else {
            selfFieldRefOp = writeOp;
          }
        }
      }
      if (selfFieldRefOp) {
        return selfFieldRefOp;
      }
      // TODO: Possible weird cases: no write or more than one write. See `handleConstrainCall()`.
      llvm::errs() << "[TODO] callOp = " << callOp << '\n';
      llvm::errs() << "[TODO] selfArgFromCall = " << selfArgFromCall << '\n';
      assert(false && "TODO: handle this!");
    }
    );
  }

  /// Remove read/write ops from within `destStruct` functions that target replaced `destStruct`
  /// fields and delete the replaced fields from `destStruct`.
  LogicalResult deleteFields(DestToSrcToClonedSrcInDest &destToSrcToClone) {
    if (!destToSrcToClone.empty()) {
      // Remove read/write ops targeting the fields
      auto eraser = [this, &destToSrcToClone](FieldRefOpInterface fRef) {
        if (destToSrcToClone.contains(this->getDef(fRef))) {
          fRef.erase();
        }
      };
      destStruct.getComputeFuncOp().getBody().walk(eraser);
      destStruct.getConstrainFuncOp().getBody().walk(eraser);
      // Delete the fields themselves
      for (auto &[f, _] : destToSrcToClone) {
        // erase via SymbolTable so table itself is updated too
        auto parentStruct = f.getParentOp<StructDefOp>(); // parent is StructDefOp per ODS
        tables.getSymbolTable(parentStruct).erase(f);
      }
    }
    return success();
  }

  // Implementation note: This pattern is split out from `cloneWithFieldRefUpdate()` because the
  // result value from the `CreateStructOp` is used by the `ReturnOp` that is not removed until the
  // actual inlining takes place.
  class EraseExcessCreateStructOp final : public OpRewritePattern<CreateStructOp> {
    StructType toDelete;

  public:
    EraseExcessCreateStructOp(MLIRContext *ctx, StructType newStructTypeToDelete)
        : OpRewritePattern(ctx), toDelete(newStructTypeToDelete) {}

    LogicalResult match(CreateStructOp op) const final { return success(op.getType() == toDelete); }
    void rewrite(CreateStructOp op, PatternRewriter &rewriter) const final {
      // TODO: Although unlikely, there can be uses of the `CreateStructOp` result besides the
      // `FieldWriteOp` that were deleted by `deleteFields()`. In that case, the `eraseOp()` would
      // fail with the assertion "expected 'op' to have no uses". One example that would cause this
      // assertion failure is passing the struct instance returned by a "compute()" call into some
      // other function that does not get inlined. In that case, additional transformations would
      // be needed to modify the CallOp and the target FuncDefOp to accept all fields of `srcStruct`
      // as individual parameters.
      rewriter.eraseOp(op);
    }
  };

  LogicalResult handleRemainingStructValues() {
    if (false) { // TODO:TEMP
      llvm::dbgs() << "[FINDME] After inlining " << srcStruct.getSymName() << ":\n";
      destStruct.dump();
    }

    MLIRContext *ctx = destStruct.getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<EraseExcessCreateStructOp>(ctx, srcStruct.getType());
    // Just run on "compute()" function because `CreateStructOp` cannot be used in "constrain()"
    walkAndApplyPatterns(destStruct.getComputeFuncOp(), std::move(patterns));

    return success();
  }

public:
  StructInliner(SymbolTableCollection &tables, StructDefOp from, StructDefOp into)
      : StructInlinerBase(tables), srcStruct(from), destStruct(into) {}

  LogicalResult doInline() {
    LLVM_DEBUG(
        llvm::dbgs() << "[StructInliner] merge " << srcStruct.getSymNameAttr() << " into "
                     << destStruct.getSymNameAttr() << '\n'
    );

    DestToSrcToClonedSrcInDest destToSrcToClone = cloneFields();
    // clang-format off
    if (failed(handleConstrainCall(destToSrcToClone))
        || failed(handleComputeCall(destToSrcToClone))
        || failed(deleteFields(destToSrcToClone))) {
      return failure();
    }
    // clang-format on
    return handleRemainingStructValues();
  }
};

class InlineStructsPass : public llzk::impl::InlineStructsPassBase<InlineStructsPass> {
  /// Maps caller struct to callees that should be inlined. The outer SmallVector preserves the
  /// ordering from the bottom-up traversal that builds the InliningPlan so performing inlining
  /// in the order given will not lose any or require doing any more than once.
  // TODO: however, applying in the opposite direction would reduce making clones of clones of ...
  // for the ops within the inlined struct functions, as they are inlined further up the tree. But
  // would have to make sure it's done properly, and update some mappings in this plan as we go.
  using InliningPlan = SmallVector<std::pair<StructDefOp, SmallVector<StructDefOp>>>;

  static unsigned complexity(FuncDefOp f) {
    unsigned complexity = 0;
    f.getBody().walk([&complexity](Operation *op) {
      // TODO: `EmitEqualityOp` and `EmitContainmentOp` should probably increment based on
      // dimension sizes in the operands rather than just +1 in all cases.
      if (llvm::isa<constrain::EmitEqualityOp, constrain::EmitContainmentOp, felt::MulFeltOp>(op)) {
        ++complexity;
      }
    });
    return complexity;
  }

  static FailureOr<FuncDefOp>
  getIfStructConstrain(const SymbolUseGraphNode *node, SymbolTableCollection &tables) {
    auto lookupRes = node->lookupSymbol(tables, false);
    assert(succeeded(lookupRes) && "graph contains node with invalid path");
    if (auto f = llvm::dyn_cast<FuncDefOp>(lookupRes->get())) {
      if (f.isStructConstrain()) {
        return f;
      }
    }
    return failure();
  }

  /// Return the parent StructDefOp for the given Function (which is known to be a struct
  /// "constrain" function so it must have a StructDefOp parent).
  static inline StructDefOp getParentStruct(FuncDefOp func) {
    assert(func.isStructConstrain()); // pre-condition
    auto currentNodeParentStruct = getParentOfType<StructDefOp>(func);
    assert(succeeded(currentNodeParentStruct)); // follows from ODS definition
    return currentNodeParentStruct.value();
  }

  /// Return 'true' iff the `maxComplexity` option is set and the given value exceeds it.
  inline bool exceedsMaxComplexity(unsigned check) {
    return maxComplexity > 0 && check > maxComplexity;
  }

  /// Perform a bottom-up traversal of the "constrain" function nodes in the SymbolUseGraph to
  /// determine which ones can be inlined to their callers while respecting the `maxComplexity`
  /// option. Using a bottom-up traversal may give a better result than top-down because the latter
  /// could result in a chain of structs being inlined differently from different use sites. The
  /// resulting `InliningPlan` also preserves this bottom-up ordering to
  inline InliningPlan makePlan(const SymbolUseGraph &useGraph, SymbolTableCollection &tables) {
    LLVM_DEBUG(
        llvm::dbgs() << "Running InlineStructsPass with max complexity " << maxComplexity << '\n'
    );
    InliningPlan retVal;
    DenseMap<const SymbolUseGraphNode *, unsigned> complexityMemo;

    // NOTE: The assumption that the use graph has no cycles allows `complexityMemo` to only
    // store the result for relevant nodes and assume nodes without a mapped value are `0`. This
    // must be true of the "compute"/"constrain" function uses and field defs because circuits
    // must be acyclic. This is likely true to for the symbol use graph is general but if a
    // counterexample is ever found, the algorithm below must be re-evaluated.
    assert(!hasCycle(&useGraph));

    // Traverse "constrain" function nodes to compute their complexity and an inlining plan. Use
    // post-order traversal so the complexity of all successor nodes is computed before computing
    // the current node's complexity.
    for (const SymbolUseGraphNode *currentNode : llvm::post_order(&useGraph)) {
      if (!currentNode->isRealNode()) {
        continue;
      }
      FailureOr<FuncDefOp> currentFunc = getIfStructConstrain(currentNode, tables);
      if (failed(currentFunc)) {
        continue;
      }
      unsigned currentComplexity = complexity(currentFunc.value());
      // If the current complexity is already too high, store it and continue.
      if (exceedsMaxComplexity(currentComplexity)) {
        complexityMemo[currentNode] = currentComplexity;
        continue;
      }
      // Otherwise, make a plan that adds successor "constrain" functions unless the
      // complexity becomes too high by adding that successor.
      SmallVector<StructDefOp> successorsToMerge;
      for (const SymbolUseGraphNode *successor : currentNode->successorIter()) {
        // Note: all "constrain" function nodes will have a value, and all other nodes will not.
        auto memoResult = complexityMemo.find(successor);
        if (memoResult == complexityMemo.end()) {
          continue; // inner loop
        }
        unsigned sComplexity = memoResult->second;
        unsigned potentialComplexity = currentComplexity + sComplexity;
        assert(potentialComplexity >= currentComplexity && "overflow");
        if (!exceedsMaxComplexity(potentialComplexity)) {
          currentComplexity = potentialComplexity;
          FailureOr<FuncDefOp> successorFunc = getIfStructConstrain(successor, tables);
          assert(succeeded(successorFunc)); // follows from the Note above
          successorsToMerge.push_back(getParentStruct(successorFunc.value()));
        }
      }
      complexityMemo[currentNode] = currentComplexity;
      if (!successorsToMerge.empty()) {
        retVal.emplace_back(getParentStruct(currentFunc.value()), std::move(successorsToMerge));
      }
    }

    return retVal;
  }

public:
  void runOnOperation() override {
    const SymbolUseGraph &useGraph = getAnalysis<SymbolUseGraph>();
    LLVM_DEBUG(useGraph.dumpToDotFile());

    SymbolTableCollection tables;
    InliningPlan plan = makePlan(useGraph, tables);
    for (auto &[caller, callees] : plan) {
      for (StructDefOp toInline : callees) {
        LogicalResult res = StructInliner(tables, toInline, caller).doInline();
        if (failed(res)) {
          llvm::errs() << "failure!" << '\n'; // TODO: message?
          signalPassFailure();
          return;
        }
      }
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> llzk::createInlineStructsPass() {
  return std::make_unique<InlineStructsPass>();
};
