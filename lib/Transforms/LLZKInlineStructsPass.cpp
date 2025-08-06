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
#include <llvm/ADT/TypeSwitch.h>
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

/// Mapping of `destStruct` field that has `srcStruct` type to each FieldDefOp from `srcStruct` to
/// the cloned version of that `srcStruct` field within `destStruct`.
using DestFieldWithSrcStructType = FieldDefOp;
using DestCloneOfSrcStructField = FieldDefOp;
using SrcStructFieldToCloneInDest = DenseMap<StringAttr, DestCloneOfSrcStructField>;
using DestToSrcToClonedSrcInDest =
    DenseMap<DestFieldWithSrcStructType, SrcStructFieldToCloneInDest>;

Value getSelfValue(FuncDefOp f) {
  if (f.nameIsCompute()) {
    return f.getSelfValueFromCompute();
  } else if (f.nameIsConstrain()) {
    return f.getSelfValueFromConstrain();
  } else {
    llvm_unreachable("expected \"compute\" or \"constrain\" function");
  }
}

inline FieldDefOp getDef(SymbolTableCollection &tables, FieldRefOpInterface fRef) {
  auto r = fRef.getFieldDefOp(tables);
  assert(succeeded(r));
  return r->get();
}

/// If there exists a field ref chain in `destToSrcToClone` for the given `FieldReadOp` (as
/// described in `combineReadChain()` or `combineNewThenReadChain()`), replace it with a
/// new `FieldReadOp` that directly reads from the cloned field.
bool combineHelper(
    FieldReadOp readOp, SymbolTableCollection &tables,
    const DestToSrcToClonedSrcInDest &destToSrcToClone, FieldRefOpInterface destFieldRefOp
) {
  auto srcToClone = destToSrcToClone.find(getDef(tables, destFieldRefOp));
  if (srcToClone == destToSrcToClone.end()) {
    return false;
  }
  SrcStructFieldToCloneInDest oldToNewFields = srcToClone->second;
  auto resNewField = oldToNewFields.find(readOp.getFieldNameAttr().getAttr());
  if (resNewField == oldToNewFields.end()) {
    return false;
  }

  // Replace this FieldReadOp with a new one that targets the cloned field.
  OpBuilder builder(readOp);
  auto newRead = builder.create<FieldReadOp>(
      readOp.getLoc(), readOp.getType(), destFieldRefOp.getComponent(),
      resNewField->second.getNameAttr()
  );
  readOp.replaceAllUsesWith(newRead.getOperation());
  readOp.erase(); // delete the original FieldReadOp
  return true;
}

/// If the base component Value of the given FieldReadOp is the result of reading from a field in
/// `destToSrcToClone` and the field referenced by this FieldReadOp has a cloned field mapping in
/// `destToSrcToClone`, replace this read with a new FieldReadOp referencing the cloned field.
///
/// Example:
///   Given the mapping (@fa, !struct.type<@Component10A>) -> @f -> \@"fa:!s<@Component10A>+f"
///   And the input:
///     %0 = struct.readf %arg0[@fa] : !struct.type<@Main>, !struct.type<@Component10A>
///     %3 = struct.readf %0[@f] : !struct.type<@Component10A>, !felt.type
///   Replace the final read with:
///     %3 = struct.readf %arg0[@"fa:!s<@Component10A>+f"] : !struct.type<@Main>, !felt.type
///
/// Return true if replaced, false if not.
bool combineReadChain(
    FieldReadOp readOp, SymbolTableCollection &tables,
    const DestToSrcToClonedSrcInDest &destToSrcToClone
) {
  auto readThatDefinesBaseComponent =
      llvm::dyn_cast_if_present<FieldReadOp>(readOp.getComponent().getDefiningOp());
  if (!readThatDefinesBaseComponent) {
    return false;
  }
  return combineHelper(readOp, tables, destToSrcToClone, readThatDefinesBaseComponent);
}

/// If the base component Value of the given FieldReadOp is the result of `struct.new` which is
/// written to a field in `destToSrcToClone` and the field referenced by this FieldReadOp has a
/// cloned field mapping in `destToSrcToClone`, replace this read with a new FieldReadOp referencing
/// the cloned field.
///
/// Example:
///   Given the mapping (@fa, !struct.type<@Component10A>) -> @f -> \@"fa:!s<@Component10A>+f"
///   And the input:
///     %0 = struct.new : !struct.type<@Main>
///     %2 = struct.new : !struct.type<@Component10A>
///     struct.writef %0[@fa] = %2 : !struct.type<@Main>, !struct.type<@Component10A>
///     %4 = struct.readf %2[@f] : !struct.type<@Component10A>, !felt.type
///   Replace the final read with:
///     %4 = struct.readf %0[@"fa:!s<@Component10A>+f"] : !struct.type<@Main>, !felt.type
///
/// Return true if replaced, false if not.
LogicalResult combineNewThenReadChain(
    FieldReadOp readOp, SymbolTableCollection &tables,
    const DestToSrcToClonedSrcInDest &destToSrcToClone
) {
  auto createThatDefinesBaseComponent =
      llvm::dyn_cast_if_present<CreateStructOp>(readOp.getComponent().getDefiningOp());
  if (!createThatDefinesBaseComponent) {
    return success(); // No error. The pattern simply doesn't match.
  }

  FieldWriteOp foundWrite = nullptr;
  Value createdValue = createThatDefinesBaseComponent.getResult();
  for (OpOperand &use : createdValue.getUses()) {
    if (auto writeOp = llvm::dyn_cast<FieldWriteOp>(use.getOwner())) {
      // Find the write op that stores the created value
      if (writeOp.getVal() == createdValue) {
        if (foundWrite) {
          // Note: There is no reason for a subcomponent to be stored to more than one field.
          auto diag = createThatDefinesBaseComponent.emitOpError(
              "result should not be written to more than one field."
          );
          diag.attachNote(foundWrite.getLoc()).append("written here");
          diag.attachNote(writeOp.getLoc()).append("written here");
          return diag;
        } else {
          foundWrite = writeOp;
        }
      }
    }
  }

  if (!foundWrite) {
    // Note: There is no reason to construct a subcomponent and not store it to a field.
    return createThatDefinesBaseComponent.emitOpError("result should be written to a field.");
  }

  return success(combineHelper(readOp, tables, destToSrcToClone, foundWrite));
}

/// Cache various ops from the caller struct that should be erased but only after all callees are
/// fully handled (to avoid "still has uses" errors).
struct PendingErasure {
  SmallVector<FieldRefOpInterface> fieldRefOps;
  SmallVector<CreateStructOp> newStructOps;
  SmallVector<FieldDefOp> fieldDefs;
};

class StructInlinerBase {

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

protected:
  SymbolTableCollection &tables;
  PendingErasure &toDelete;

  StructInlinerBase(SymbolTableCollection &symTables, PendingErasure &opsToDelete)
      : tables(symTables), toDelete(opsToDelete) {}

  inline FieldDefOp getDef(FieldRefOpInterface fRef) const { return ::getDef(tables, fRef); }

  LogicalResult doInlining(
      const DestToSrcToClonedSrcInDest &destToSrcToClone, FuncDefOp srcFunc, FuncDefOp destFunc,
      llvm::function_ref<FieldRefOpInterface(CallOp)> getSelfRef,
      llvm::function_ref<void(FuncDefOp)> processCloneBeforeInlining = nullptr
  ) {
    InlinerInterface inliner(destFunc.getContext());

    /// Replaces CallOp that target `srcFunc` with an inlined version of `srcFunc`.
    auto callHandler = [&](CallOp callOp) {
      // Ensure the CallOp targets `srcFunc`
      auto callOpTarget = callOp.getCalleeTarget(tables);
      assert(succeeded(callOpTarget));
      if (callOpTarget->get() != srcFunc) {
        return WalkResult::advance();
      }

      // Get the "self" struct value from the CallOp and determine which struct field is used.
      FieldRefOpInterface selfFieldRefOp = getSelfRef(callOp);
      if (!selfFieldRefOp) {
        // Note: error message was already printed within `getSelfRef` callback.
        return WalkResult::interrupt(); // use interrupt to signal failure
      }

      // Create a clone of the source function (must do the whole function not just the body region
      // because `inlineCall()` expects the Region to have a parent op) and update field references
      // to the old struct fields to instead use the new struct fields.
      FuncDefOp srcFuncClone =
          FieldRefRewriter::cloneWithFieldRefUpdate(std::make_unique<FieldRefRewriter>(
              srcFunc, selfFieldRefOp.getComponent(), destToSrcToClone.at(getDef(selfFieldRefOp))
          ));
      if (processCloneBeforeInlining) {
        processCloneBeforeInlining(srcFuncClone);
      }

      // Inline the cloned function in place of `callOp`
      LogicalResult inlineCallRes =
          inlineCall(inliner, callOp, srcFuncClone, &srcFuncClone.getBody(), false);
      if (failed(inlineCallRes)) {
        callOp.emitError().append("Failed to inline ", srcFunc.getFullyQualifiedName()).report();
        return WalkResult::interrupt(); // use interrupt to signal failure
      }
      srcFuncClone.erase();      // delete what's left after transferring the body elsewhere
      callOp.erase();            // delete the original CallOp
      return WalkResult::skip(); // Must skip because the CallOp was erased.
    };

    auto fieldWriteHandler = [&](FieldWriteOp writeOp) {
      // Check if the field ref op should be deleted in the end
      if (destToSrcToClone.contains(getDef(writeOp))) {
        toDelete.fieldRefOps.push_back(writeOp);
      }
      return WalkResult::advance();
    };

    /// Combine chained FieldReadOp according to replacements in `destToSrcToClone`.
    /// See `combineReadChain()`
    auto fieldReadHandler = [&](FieldReadOp readOp) {
      // Check if the field ref op should be deleted in the end
      if (destToSrcToClone.contains(getDef(readOp))) {
        toDelete.fieldRefOps.push_back(readOp);
      }
      // If the FieldReadOp was replaced/erased, must skip.
      return combineReadChain(readOp, tables, destToSrcToClone) ? WalkResult::skip()
                                                                : WalkResult::advance();
    };

    auto walkRes = destFunc.getBody().walk<WalkOrder::PreOrder>([&](Operation *op) {
      return TypeSwitch<Operation *, WalkResult>(op)
          .Case<CallOp>(callHandler)
          .Case<FieldWriteOp>(fieldWriteHandler)
          .Case<FieldReadOp>(fieldReadHandler)
          .Default([](Operation *) { return WalkResult::advance(); });
    });

    return failure(walkRes.wasInterrupted());
  }
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
        // Mark the original `destField` for deletion
        toDelete.fieldDefs.push_back(destField);
        // Clone each field from 'srcStruct' into 'destStruct'. Add an entry to `destToSrcToClone`
        // even if there are no fields in `srcStruct` so its presence can be used as a marker.
        SrcStructFieldToCloneInDest &srcToClone = destToSrcToClone.getOrInsertDefault(destField);
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
  LogicalResult inlineConstrainCall(const DestToSrcToClonedSrcInDest &destToSrcToClone) {
    return doInlining(
        destToSrcToClone, srcStruct.getConstrainFuncOp(), destStruct.getConstrainFuncOp(),
        [destStruct = &this->destStruct](CallOp callOp) {
      // The typical pattern is to read a struct instance from a field and then call "constrain()"
      // on it. Get the Value passed as the "self" struct to the CallOp and determine which field it
      // was read from in the current struct (i.e., `destStruct`).
      Value selfArgFromCall = callOp.getSelfValueFromConstrain();
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
  LogicalResult inlineComputeCall(const DestToSrcToClonedSrcInDest &destToSrcToClone) {
    return doInlining(
        destToSrcToClone, srcStruct.getComputeFuncOp(), destStruct.getComputeFuncOp(),
        [destStruct = &this->destStruct](CallOp callOp) {
      // The typical pattern is to write the return value of "compute()" to a field in
      // the current struct (i.e., `destStruct`).
      FieldRefOpInterface selfFieldRefOp = nullptr;
      Value selfArgFromCall = callOp.getSelfValueFromCompute();
      for (OpOperand &use : selfArgFromCall.getUses()) {
        if (auto writeOp = llvm::dyn_cast<FieldWriteOp>(use.getOwner())) {
          // ASSERT: FieldWriteOp are only allowed to write to the current struct.
          assert(writeOp.getComponent().getType() == destStruct->getType());
          // ASSERT: The only other Value use in a write op is the Value being written.
          assert(writeOp.getVal() == selfArgFromCall);
          if (selfFieldRefOp) {
            // Note: There is no reason for a subcomponent to be stored to more than one field.
            auto diag = callOp.emitOpError().append(
                "\"@", FUNC_NAME_COMPUTE, "\" result should not be written to more than one field."
            );
            diag.attachNote(selfFieldRefOp.getLoc()).append("written here");
            diag.attachNote(writeOp.getLoc()).append("written here");
            diag.report();
            return static_cast<FieldRefOpInterface>(nullptr);
          } else {
            selfFieldRefOp = writeOp;
          }
        }
      }
      if (!selfFieldRefOp) {
        // Note: There is no reason to construct a subcomponent and not store it to a field.
        callOp.emitOpError()
            .append("\"@", FUNC_NAME_COMPUTE, "\" result should be written to a field.")
            .report();
      }
      return selfFieldRefOp;
    },
        // Within the compute function, find `CreateStructOp` with `srcStruct` type and mark them
        // for later deletion. The deletion must occur later because these values may still have
        // uses until ALL callees of a function have been inlined.
        [this](FuncDefOp func) {
      StructType srcStructType = this->srcStruct.getType();
      func.getBody().walk([&](CreateStructOp newStructOp) {
        if (newStructOp.getType() == srcStructType) {
          toDelete.newStructOps.push_back(newStructOp);
        }
      });
    }
    );
  }

public:
  StructInliner(
      SymbolTableCollection &tables, PendingErasure &toDelete, StructDefOp from, StructDefOp into
  )
      : StructInlinerBase(tables, toDelete), srcStruct(from), destStruct(into) {}

  FailureOr<DestToSrcToClonedSrcInDest> doInline() {
    LLVM_DEBUG(
        llvm::dbgs() << "[StructInliner] merge " << srcStruct.getSymNameAttr() << " into "
                     << destStruct.getSymNameAttr() << '\n'
    );

    DestToSrcToClonedSrcInDest destToSrcToClone = cloneFields();
    if (failed(inlineConstrainCall(destToSrcToClone)) ||
        failed(inlineComputeCall(destToSrcToClone))) {
      return failure(); // error already printed within doInlining()
    }
    return destToSrcToClone;
  }
};

class InlineStructsPass : public llzk::impl::InlineStructsPassBase<InlineStructsPass> {
  /// Maps caller struct to callees that should be inlined. The outer SmallVector preserves the
  /// ordering from the bottom-up traversal that builds the InliningPlan so performing inlining
  /// in the order given will not lose any or require doing any more than once.
  /// Note: Applying in the opposite direction would reduce making repeated clones of the ops within
  /// the inlined struct functions (as they are inlined further and further up the tree) but that
  /// would require updating some mapping in the plan along the way to ensure it's done properly.
  using InliningPlan = SmallVector<std::pair<StructDefOp, SmallVector<StructDefOp>>>;

  static int64_t complexity(FuncDefOp f) {
    int64_t complexity = 0;
    f.getBody().walk([&complexity](Operation *op) {
      if (llvm::isa<felt::MulFeltOp>(op)) {
        ++complexity;
      } else if (auto ee = llvm::dyn_cast<constrain::EmitEqualityOp>(op)) {
        complexity += computeEmitEqCardinality(ee.getLhs().getType());
      } else if (auto ec = llvm::dyn_cast<constrain::EmitContainmentOp>(op)) {
        // TODO: increment based on dimension sizes in the operands
        // Pending update to implementation/semantics of EmitContainmentOp.
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
  inline bool exceedsMaxComplexity(int64_t check) {
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
    DenseMap<const SymbolUseGraphNode *, int64_t> complexityMemo;

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
      int64_t currentComplexity = complexity(currentFunc.value());
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
        int64_t sComplexity = memoResult->second;
        int64_t potentialComplexity = currentComplexity + sComplexity;
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

  inline static LogicalResult finalizeStruct(
      SymbolTableCollection &tables, StructDefOp caller, PendingErasure &&toDelete,
      DestToSrcToClonedSrcInDest &&destToSrcToClone
  ) {
    // Compress chains of reads that result after inlining multiple callees.
    caller.getConstrainFuncOp().walk([&tables, &destToSrcToClone](FieldReadOp readOp) {
      combineReadChain(readOp, tables, destToSrcToClone);
    });
    auto res = caller.getComputeFuncOp().walk([&tables, &destToSrcToClone](FieldReadOp readOp) {
      combineReadChain(readOp, tables, destToSrcToClone);
      auto res = combineNewThenReadChain(readOp, tables, destToSrcToClone);
      return failed(res) ? WalkResult::interrupt() : WalkResult::advance();
    });
    if (res.wasInterrupted()) {
      return failure();
    }

    // To avoid "still has uses" errors, must erase FieldRefOpInterface before erasing
    // the CreateStructOp or FieldDefOp.
    for (auto op : toDelete.fieldRefOps) {
      op.erase();
    }
    for (auto op : toDelete.newStructOps) {
      // Before erasing the op, check if there are any uses of the struct Value as a free function
      // parameter. This is (currently) unsupported. Free functions with a body could either be
      // inlined also or have struct parameter(s) split to pass each field value separately. Those
      // without a body (i.e. external implementation) present a problem because LLZK does not
      // define a memory layout for structs that the external implementation could reference.
      if (!op.use_empty()) {
        for (OpOperand &use : op->getUses()) {
          if (auto c = llvm::dyn_cast<CallOp>(use.getOwner())) {
            if (!c.calleeIsStructCompute() && !c.calleeIsStructConstrain()) {
              return op
                  .emitOpError(
                      "passed as parameter to a free function is not supported by this pass."
                  )
                  .attachNote(c.getLoc())
                  .append("used by this call");
            }
          }
        }
      }
      op.erase();
    }
    // Erase FieldDefOp via SymbolTable so table itself is updated too.
    auto callerSymTab = tables.getSymbolTable(caller);
    for (auto op : toDelete.fieldDefs) {
      assert(op.getParentOp() == caller); // using correct SymbolTable
      callerSymTab.erase(op);
    }

    return success();
  }

public:
  void runOnOperation() override {
    const SymbolUseGraph &useGraph = getAnalysis<SymbolUseGraph>();
    LLVM_DEBUG(useGraph.dumpToDotFile());

    SymbolTableCollection tables;
    InliningPlan plan = makePlan(useGraph, tables);
    for (auto &[caller, callees] : plan) {
      // Cache operations that should be deleted but must wait until all callees are processed
      // to ensure that all uses of the values defined by these operations are replaced.
      PendingErasure toDelete;
      // Cache old-to-new field mappings across all calleeds inlined for the current struct.
      DestToSrcToClonedSrcInDest aggregateReplacements;
      // Inline callees/subcomponents of the current struct
      for (StructDefOp toInline : callees) {
        FailureOr<DestToSrcToClonedSrcInDest> res =
            StructInliner(tables, toDelete, toInline, caller).doInline();
        if (failed(res)) {
          signalPassFailure(); // error already printed w/in doInline()
          return;
        }
        // Add current field replacements to the aggregate
        for (auto &[k, v] : res.value()) {
          assert(!aggregateReplacements.contains(k) && "duplicate not possible");
          aggregateReplacements[k] = std::move(v);
        }
      }
      // Complete steps to finalize/cleanup the caller
      auto finalizeResult =
          finalizeStruct(tables, caller, std::move(toDelete), std::move(aggregateReplacements));
      if (failed(finalizeResult)) {
        signalPassFailure(); // error already printed w/in combineNewThenReadChain()
        return;
      }
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> llzk::createInlineStructsPass() {
  return std::make_unique<InlineStructsPass>();
};
