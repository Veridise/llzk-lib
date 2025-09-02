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
/// This pass should be run after `llzk-flatten` to ensure structs do not have template parameters
/// (this restriction may be removed in the future).
///
/// This pass also assumes that all subcomponents that are created by calling a struct "@compute"
/// function are ultimately written to exactly one field within the current struct.
///
//===----------------------------------------------------------------------===//

#include "llzk/Analysis/GraphUtil.h"
#include "llzk/Analysis/SymbolDefTree.h"
#include "llzk/Analysis/SymbolUseGraph.h"
#include "llzk/Dialect/Constrain/IR/Ops.h"
#include "llzk/Dialect/Felt/IR/Ops.h"
#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Dialect/Struct/IR/Ops.h"
#include "llzk/Transforms/LLZKConversionUtils.h"
#include "llzk/Transforms/LLZKTransformationPasses.h"
#include "llzk/Transforms/WalkPatternRewriteDriver.h"
#include "llzk/Util/Debug.h"
#include "llzk/Util/SymbolHelper.h"
#include "llzk/Util/SymbolLookup.h"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Transforms/InliningUtils.h>

#include <llvm/ADT/PostOrderIterator.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringMap.h>
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

using DestFieldWithSrcStructType = FieldDefOp;
using DestCloneOfSrcStructField = FieldDefOp;
/// Mapping of the name of each field in the inlining source struct to the new cloned version of the
/// source field in the destination struct. Uses `std::map` for consistent ordering between multiple
/// compilations of the same LLZK IR input.
using SrcStructFieldToCloneInDest = std::map<StringRef, DestCloneOfSrcStructField>;
/// Mapping of `FieldDefOp` in the inlining destination struct to each `FieldDefOp` from the
/// inlining source struct to the new cloned version of the source field in the destination struct.
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
  auto resNewField = oldToNewFields.find(readOp.getFieldName());
  if (resNewField == oldToNewFields.end()) {
    return false;
  }

  // Replace this FieldReadOp with a new one that targets the cloned field.
  OpBuilder builder(readOp);
  FieldReadOp newRead = builder.create<FieldReadOp>(
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
  FieldReadOp readThatDefinesBaseComponent =
      llvm::dyn_cast_if_present<FieldReadOp>(readOp.getComponent().getDefiningOp());
  if (!readThatDefinesBaseComponent) {
    return false;
  }
  return combineHelper(readOp, tables, destToSrcToClone, readThatDefinesBaseComponent);
}

/// Find the `FieldWriteOp` that writes the given subcomponent struct `Value`. Produce an error
/// (using the given callback) if there is not exactly once such `FieldWriteOp`.
FailureOr<FieldWriteOp>
findOpThatStoresSubcmp(Value writtenValue, function_ref<InFlightDiagnostic()> emitError) {
  FieldWriteOp foundWrite = nullptr;
  for (Operation *user : writtenValue.getUsers()) {
    if (FieldWriteOp writeOp = llvm::dyn_cast<FieldWriteOp>(user)) {
      // Find the write op that stores the created value
      if (writeOp.getVal() == writtenValue) {
        if (foundWrite) {
          // Note: There is no reason for a subcomponent to be stored to more than one field.
          auto diag = emitError().append("result should not be written to more than one field.");
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
    return emitError().append("result should be written to a field.");
  }
  return foundWrite;
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
  CreateStructOp createThatDefinesBaseComponent =
      llvm::dyn_cast_if_present<CreateStructOp>(readOp.getComponent().getDefiningOp());
  if (!createThatDefinesBaseComponent) {
    return success(); // No error. The pattern simply doesn't match.
  }
  FailureOr<FieldWriteOp> foundWrite =
      findOpThatStoresSubcmp(createThatDefinesBaseComponent, [&createThatDefinesBaseComponent]() {
    return createThatDefinesBaseComponent.emitOpError();
  });
  if (failed(foundWrite)) {
    return failure(); // error already printed within findOpThatStoresSubcmp()
  }
  return success(combineHelper(readOp, tables, destToSrcToClone, foundWrite.value()));
}

inline FieldReadOp getFieldReadThatDefinesSelfValuePassedToConstrain(CallOp callOp) {
  Value selfArgFromCall = callOp.getSelfValueFromConstrain();
  return llvm::dyn_cast_if_present<FieldReadOp>(selfArgFromCall.getDefiningOp());
}

/// Cache various ops from the caller struct that should be erased but only after all callees are
/// fully handled (to avoid "still has uses" errors).
struct PendingErasure {
  SmallVector<FieldRefOpInterface> fieldRefOps;
  SmallVector<CreateStructOp> newStructOps;
  SmallVector<DestFieldWithSrcStructType> fieldDefs;
};

class StructInliner {
  SymbolTableCollection &tables;
  PendingErasure &toDelete;
  /// The struct that will be inlined (and maybe removed).
  StructDefOp srcStruct;
  /// The struct whose body will be augmented with the inlined content.
  StructDefOp destStruct;

  inline FieldDefOp getDef(FieldRefOpInterface fRef) const { return ::getDef(tables, fRef); }

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
      return success(op.getComponent() == oldBaseVal && oldToNewFields.contains(op.getFieldName()));
    }

    void rewrite(FieldRefOpInterface op, PatternRewriter &rewriter) const final {
      rewriter.modifyOpInPlace(op, [this, &op]() {
        DestCloneOfSrcStructField newF = oldToNewFields.at(op.getFieldName());
        op.setFieldName(newF.getSymName());
        op.getComponentMutable().set(this->newBaseVal);
      });
    }

    /// Create a clone of the `FuncDefOp` and update field references according to the
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

  /// Common implementation for inlining both "constrain" and "compute" functions.
  class ImplBase {
  protected:
    const StructInliner &data;
    const DestToSrcToClonedSrcInDest &destToSrcToClone;

    /// Get the "self" struct parameter from the CallOp and determine which field that struct was
    /// stored in within the caller.
    virtual FieldRefOpInterface getSelfRefField(CallOp callOp) = 0;
    virtual void processCloneBeforeInlining(FuncDefOp func) {}
    virtual ~ImplBase() = default;

  public:
    ImplBase(const StructInliner &inliner, const DestToSrcToClonedSrcInDest &destToSrcToCloneRef)
        : data(inliner), destToSrcToClone(destToSrcToCloneRef) {}

    LogicalResult doInlining(FuncDefOp srcFunc, FuncDefOp destFunc) {
      LLVM_DEBUG({
        llvm::dbgs() << "[doInlining] SOURCE FUNCTION:\n";
        srcFunc.dump();
        llvm::dbgs() << "[doInlining] DESTINATION FUNCTION:\n";
        destFunc.dump();
      });

      InlinerInterface inliner(destFunc.getContext());

      /// Replaces CallOp that target `srcFunc` with an inlined version of `srcFunc`.
      auto callHandler = [this, &inliner, &srcFunc](CallOp callOp) {
        // Ensure the CallOp targets `srcFunc`
        auto callOpTarget = callOp.getCalleeTarget(this->data.tables);
        assert(succeeded(callOpTarget));
        if (callOpTarget->get() != srcFunc) {
          return WalkResult::advance();
        }

        // Get the "self" struct parameter from the CallOp and determine which field that struct
        // was stored in within the caller (i.e. `destFunc`).
        FieldRefOpInterface selfFieldRefOp = this->getSelfRefField(callOp);
        if (!selfFieldRefOp) {
          // Note: error message was already printed within `getSelfRefField()`
          return WalkResult::interrupt(); // use interrupt to signal failure
        }

        // Create a clone of the source function (must do the whole function not just the body
        // region because `inlineCall()` expects the Region to have a parent op) and update field
        // references to the old struct fields to instead use the new struct fields.
        FuncDefOp srcFuncClone =
            FieldRefRewriter::cloneWithFieldRefUpdate(std::make_unique<FieldRefRewriter>(
                srcFunc, selfFieldRefOp.getComponent(),
                this->destToSrcToClone.at(this->data.getDef(selfFieldRefOp))
            ));
        this->processCloneBeforeInlining(srcFuncClone);

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

      auto fieldWriteHandler = [this](FieldWriteOp writeOp) {
        // Check if the field ref op should be deleted in the end
        if (this->destToSrcToClone.contains(this->data.getDef(writeOp))) {
          this->data.toDelete.fieldRefOps.push_back(writeOp);
        }
        return WalkResult::advance();
      };

      /// Combine chained FieldReadOp according to replacements in `destToSrcToClone`.
      /// See `combineReadChain()`
      auto fieldReadHandler = [this](FieldReadOp readOp) {
        // Check if the field ref op should be deleted in the end
        if (this->destToSrcToClone.contains(this->data.getDef(readOp))) {
          this->data.toDelete.fieldRefOps.push_back(readOp);
        }
        // If the FieldReadOp was replaced/erased, must skip.
        return combineReadChain(readOp, this->data.tables, destToSrcToClone)
                   ? WalkResult::skip()
                   : WalkResult::advance();
      };

      WalkResult walkRes = destFunc.getBody().walk<WalkOrder::PreOrder>([&](Operation *op) {
        return TypeSwitch<Operation *, WalkResult>(op)
            .Case<CallOp>(callHandler)
            .Case<FieldWriteOp>(fieldWriteHandler)
            .Case<FieldReadOp>(fieldReadHandler)
            .Default([](Operation *) { return WalkResult::advance(); });
      });

      return failure(walkRes.wasInterrupted());
    }
  };

  class ConstrainImpl : public ImplBase {
    using ImplBase::ImplBase;

    FieldRefOpInterface getSelfRefField(CallOp callOp) override {
      // The typical pattern is to read a struct instance from a field and then call "constrain()"
      // on it. Get the Value passed as the "self" struct to the CallOp and determine which field it
      // was read from in the current struct (i.e., `destStruct`).
      FieldRefOpInterface selfFieldRef = getFieldReadThatDefinesSelfValuePassedToConstrain(callOp);
      if (selfFieldRef &&
          selfFieldRef.getComponent().getType() == this->data.destStruct.getType()) {
        return selfFieldRef;
      }
      callOp.emitError()
          .append(
              "expected \"self\" parameter to \"@", FUNC_NAME_CONSTRAIN,
              "\" to be passed a value read from a field in the current stuct."
          )
          .report();
      return nullptr;
    }
  };

  class ComputeImpl : public ImplBase {
    using ImplBase::ImplBase;

    FieldRefOpInterface getSelfRefField(CallOp callOp) override {
      // The typical pattern is to write the return value of "compute()" to a field in
      // the current struct (i.e., `destStruct`).
      // It doesn't really make sense (although there is no semantic restriction against it) to just
      // pass the "compute()" result into another function and never write it to a field since that
      // leaves no way for the "constrain()" function to call "constrain()" on that result struct.
      FailureOr<FieldWriteOp> foundWrite =
          findOpThatStoresSubcmp(callOp.getSelfValueFromCompute(), [&callOp]() {
        return callOp.emitOpError().append("\"@", FUNC_NAME_COMPUTE, "\" ");
      });
      return static_cast<FieldRefOpInterface>(foundWrite.value_or(nullptr));
    }

    void processCloneBeforeInlining(FuncDefOp func) override {
      // Within the compute function, find `CreateStructOp` with `srcStruct` type and mark them
      // for later deletion. The deletion must occur later because these values may still have
      // uses until ALL callees of a function have been inlined.
      func.getBody().walk([this](CreateStructOp newStructOp) {
        if (newStructOp.getType() == this->data.srcStruct.getType()) {
          this->data.toDelete.newStructOps.push_back(newStructOp);
        }
      });
    }
  };

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
        assert(unifications.empty()); // `makePlan()` reports failure earlier
        // Mark the original `destField` for deletion
        toDelete.fieldDefs.push_back(destField);
        // Clone each field from 'srcStruct' into 'destStruct'. Add an entry to `destToSrcToClone`
        // even if there are no fields in `srcStruct` so its presence can be used as a marker.
        SrcStructFieldToCloneInDest &srcToClone = destToSrcToClone.getOrInsertDefault(destField);
        std::vector<FieldDefOp> srcFields = srcStruct.getFieldDefs();
        if (srcFields.empty()) {
          continue;
        }
        OpBuilder builder(destField);
        std::string newNameBase =
            destField.getName().str() + ':' + BuildShortTypeString::from(destFieldType);
        for (FieldDefOp srcField : srcFields) {
          DestCloneOfSrcStructField newF = llvm::cast<FieldDefOp>(builder.clone(*srcField));
          newF.setName(builder.getStringAttr(newNameBase + '+' + newF.getName()));
          srcToClone[srcField.getSymNameAttr()] = newF;
          // Also update the cached SymbolTable
          destStructSymTable.insert(newF);
        }
      }
    }
    return destToSrcToClone;
  }

  /// Inline the "constrain" function from `srcStruct` into `destStruct`.
  inline LogicalResult inlineConstrainCall(const DestToSrcToClonedSrcInDest &destToSrcToClone) {
    return ConstrainImpl(*this, destToSrcToClone)
        .doInlining(srcStruct.getConstrainFuncOp(), destStruct.getConstrainFuncOp());
  }

  /// Inline the "compute" function from `srcStruct` into `destStruct`.
  inline LogicalResult inlineComputeCall(const DestToSrcToClonedSrcInDest &destToSrcToClone) {
    return ComputeImpl(*this, destToSrcToClone)
        .doInlining(srcStruct.getComputeFuncOp(), destStruct.getComputeFuncOp());
  }

public:
  StructInliner(
      SymbolTableCollection &tbls, PendingErasure &opsToDelete, StructDefOp from, StructDefOp into
  )
      : tables(tbls), toDelete(opsToDelete), srcStruct(from), destStruct(into) {}

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

/// Replace the function parameter at `paramIdx` with multiple parameters according to the types of
/// the values in the `nameToNewField` map. Within the body, replace reads from the original
/// parameter with direct uses of the new block argument Values per the field name keys in the map.
inline void splitFunctionParam(
    FuncDefOp func, unsigned paramIdx, const SrcStructFieldToCloneInDest &nameToNewField
) {
  class Impl : public FunctionTypeConverter {
    unsigned inputIdx;
    const SrcStructFieldToCloneInDest &newFields;

  public:
    Impl(unsigned paramIdx, const SrcStructFieldToCloneInDest &nameToNewField)
        : inputIdx(paramIdx), newFields(nameToNewField) {}

  protected:
    SmallVector<Type> convertInputs(ArrayRef<Type> origTypes) override {
      SmallVector<Type> newTypes(origTypes);
      auto it = newTypes.erase(newTypes.begin() + inputIdx);
      for (auto [_, newField] : newFields) {
        newTypes.insert(it, newField.getType());
        ++it;
      }
      return newTypes;
    }
    SmallVector<Type> convertResults(ArrayRef<Type> origTypes) override {
      return SmallVector<Type>(origTypes);
    }
    ArrayAttr convertInputAttrs(ArrayAttr origAttrs, SmallVector<Type>) override {
      if (origAttrs) {
        // Replicate the value at `origAttrs[inputIdx]` to have `newFields.size()`
        SmallVector<Attribute> newAttrs(origAttrs.getValue());
        newAttrs.insert(newAttrs.begin() + inputIdx, newFields.size() - 1, origAttrs[inputIdx]);
        return ArrayAttr::get(origAttrs.getContext(), newAttrs);
      }
      return nullptr;
    }
    ArrayAttr convertResultAttrs(ArrayAttr origAttrs, SmallVector<Type>) override {
      return origAttrs;
    }

    void processBlockArgs(Block &entryBlock, RewriterBase &rewriter) override {
      Value oldStructRef = entryBlock.getArgument(inputIdx);

      // Insert new Block arguments, one per field, following the original one. Keep a map
      // of field name to the associated block argument for replacing FieldReadOp.
      llvm::StringMap<BlockArgument> fieldNameToNewArg;
      Location loc = oldStructRef.getLoc();
      unsigned idx = inputIdx;
      for (auto [fieldName, newField] : newFields) {
        // note: pre-increment so the original to be erased is still at `inputIdx`
        BlockArgument newArg = entryBlock.insertArgument(++idx, newField.getType(), loc);
        fieldNameToNewArg[fieldName] = newArg;
      }

      // Find all field reads from the original Block argument and replace uses of those
      // reads with the appropriate new Block argument.
      for (OpOperand &oldBlockArgUse : llvm::make_early_inc_range(oldStructRef.getUses())) {
        if (FieldReadOp readOp = llvm::dyn_cast<FieldReadOp>(oldBlockArgUse.getOwner())) {
          if (readOp.getComponent() == oldStructRef) {
            BlockArgument newArg = fieldNameToNewArg.at(readOp.getFieldName());
            rewriter.replaceAllUsesWith(readOp, newArg);
            rewriter.eraseOp(readOp);
            continue;
          }
        }
        // Currently, there's no other way in which a StructType parameter can be used.
        llvm::errs() << "Unexpected use of " << oldBlockArgUse.get() << " in "
                     << *oldBlockArgUse.getOwner() << '\n';
        llvm_unreachable("Not yet implemented");
      }

      // Delete the original Block argument
      entryBlock.eraseArgument(inputIdx);
    }
  };
  IRRewriter rewriter(func.getContext());
  Impl(paramIdx, nameToNewField).convert(func, rewriter);
}

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
    if (FuncDefOp f = llvm::dyn_cast<FuncDefOp>(lookupRes->get())) {
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
    FailureOr<StructDefOp> currentNodeParentStruct = getParentOfType<StructDefOp>(func);
    assert(succeeded(currentNodeParentStruct)); // follows from ODS definition
    return currentNodeParentStruct.value();
  }

  /// Return 'true' iff the `maxComplexity` option is set and the given value exceeds it.
  inline bool exceedsMaxComplexity(int64_t check) {
    return maxComplexity > 0 && check > maxComplexity;
  }

  /// Check for additional conditions that make inlining impossible (at least in the current
  /// implementation).
  static inline bool canInline(FuncDefOp currentFunc, FuncDefOp successorFunc) {
    // Find CallOp for `successorFunc` within `currentFunc` and check the condition used by
    // `ConstrainImpl::getSelfRefField()`.
    //
    // Implementation Note: There is a possibility that the "self" value is not from a field read.
    // It could be a parameter to the current/destination function or a global read. Inlining a
    // struct stored to a global would probably require splitting up the global into multiple, one
    // for each field in the successor/source struct. That may not be a good idea. The parameter
    // case could be handled but it will not have a mapping in `destToSrcToClone` in
    // `getSelfRefField()` and new fields will still need to be added. They can be prefixed with
    // parameter index since there is no current field name to use as the unique prefix. Handling
    // that would require refactoring the inlining process a bit.
    WalkResult res = currentFunc.walk([](CallOp c) {
      return getFieldReadThatDefinesSelfValuePassedToConstrain(c)
                 ? WalkResult::interrupt() // use interrupt to indicate success
                 : WalkResult::advance();
    });
    LLVM_DEBUG({
      llvm::dbgs() << "[canInline] " << successorFunc.getFullyQualifiedName() << " into "
                   << currentFunc.getFullyQualifiedName() << "? " << res.wasInterrupted() << '\n';
    });
    return res.wasInterrupted();
  }

  /// Perform a bottom-up traversal of the "constrain" function nodes in the SymbolUseGraph to
  /// determine which ones can be inlined to their callers while respecting the `maxComplexity`
  /// option. Using a bottom-up traversal may give a better result than top-down because the latter
  /// could result in a chain of structs being inlined differently from different use sites.
  inline FailureOr<InliningPlan>
  makePlan(const SymbolUseGraph &useGraph, SymbolTableCollection &tables) {
    LLVM_DEBUG({
      llvm::dbgs() << "Running InlineStructsPass with max complexity ";
      if (maxComplexity == 0) {
        llvm::dbgs() << "unlimited";
      } else {
        llvm::dbgs() << maxComplexity;
      }
      llvm::dbgs() << '\n';
    });
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
      LLVM_DEBUG(llvm::dbgs() << "\ncurrentNode = " << currentNode->toString());
      if (!currentNode->isRealNode()) {
        continue;
      }
      if (currentNode->isStructParam()) {
        // Try to get the location of the StructDefOp to report an error.
        Operation *lookupFrom = currentNode->getSymbolPathRoot().getOperation();
        SymbolRefAttr prefix = getPrefixAsSymbolRefAttr(currentNode->getSymbolPath());
        auto res = lookupSymbolIn<StructDefOp>(tables, prefix, lookupFrom, lookupFrom, false);
        // If that lookup didn't work for some reason, report at the path root location.
        Operation *reportLoc = succeeded(res) ? res->get() : lookupFrom;
        return reportLoc->emitError("Cannot inline structs with parameters.");
      }
      FailureOr<FuncDefOp> currentFuncOpt = getIfStructConstrain(currentNode, tables);
      if (failed(currentFuncOpt)) {
        continue;
      }
      FuncDefOp currentFunc = currentFuncOpt.value();
      int64_t currentComplexity = complexity(currentFunc);
      // If the current complexity is already too high, store it and continue.
      if (exceedsMaxComplexity(currentComplexity)) {
        complexityMemo[currentNode] = currentComplexity;
        continue;
      }
      // Otherwise, make a plan that adds successor "constrain" functions unless the
      // complexity becomes too high by adding that successor.
      SmallVector<StructDefOp> successorsToMerge;
      for (const SymbolUseGraphNode *successor : currentNode->successorIter()) {
        LLVM_DEBUG(llvm::dbgs().indent(2) << "successor: " << successor->toString() << '\n');
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
          FailureOr<FuncDefOp> successorFuncOpt = getIfStructConstrain(successor, tables);
          assert(succeeded(successorFuncOpt)); // follows from the Note above
          FuncDefOp successorFunc = successorFuncOpt.value();
          if (canInline(currentFunc, successorFunc)) {
            successorsToMerge.push_back(getParentStruct(successorFunc));
          }
        }
      }
      complexityMemo[currentNode] = currentComplexity;
      if (!successorsToMerge.empty()) {
        retVal.emplace_back(getParentStruct(currentFunc), std::move(successorsToMerge));
      }
    }
    LLVM_DEBUG({
      llvm::dbgs() << "-----------------------------------------------------------------\n";
      llvm::dbgs() << "InlineStructsPass plan:\n";
      for (auto &[caller, callees] : retVal) {
        llvm::dbgs().indent(2) << "inlining the following into \"" << caller.getSymName() << "\"\n";
        for (StructDefOp c : callees) {
          llvm::dbgs().indent(4) << "\"" << c.getSymName() << "\"\n";
        }
      }
      llvm::dbgs() << "-----------------------------------------------------------------\n";
    });
    return retVal;
  }

  /// Called before erasing an Operation to ensure that any remaining uses of the Operation's result
  /// are removed if possible, else report an error (the subsequent call to erase() would fail
  /// anyway if the result Value still has uses). Handles the following cases:
  /// - If the op is used as argument to a function with a body, convert to take fields separately.
  /// - If the op is used as argument to a function without a body, report an error.
  static LogicalResult handleRemainingUses(
      Operation *op, SymbolTableCollection &tables,
      const DestToSrcToClonedSrcInDest &destToSrcToClone,
      ArrayRef<FieldRefOpInterface> otherRefsToBeDeleted = {}
  ) {
    if (op->use_empty()) {
      return success(); // safe to erase
    }

    // Helper function to determine if an Operation is contained in 'otherRefsToBeDeleted'
    auto opWillBeDeleted = [&otherRefsToBeDeleted](Operation *op) -> bool {
      return std::find(otherRefsToBeDeleted.begin(), otherRefsToBeDeleted.end(), op) !=
             otherRefsToBeDeleted.end();
    };

    LLVM_DEBUG({
      llvm::dbgs() << "[handleRemainingUses] op: " << *op << '\n';
      llvm::dbgs() << "[handleRemainingUses]   in function: " << op->getParentOfType<FuncDefOp>()
                   << '\n';
    });
    for (OpOperand &use : llvm::make_early_inc_range(op->getUses())) {
      if (CallOp c = llvm::dyn_cast<CallOp>(use.getOwner())) {
        LLVM_DEBUG(llvm::dbgs() << "[handleRemainingUses]   use in call: " << c << '\n');
        unsigned argIdx = use.getOperandNumber() - c.getArgOperands().getBeginOperandIndex();
        LLVM_DEBUG(llvm::dbgs() << "[handleRemainingUses]     at index: " << argIdx << '\n');

        auto tgtFuncRes = c.getCalleeTarget(tables);
        if (failed(tgtFuncRes)) {
          return op
              ->emitOpError("as argument to an unknown function is not supported by this pass.")
              .attachNote(c.getLoc())
              .append("used by this call");
        }
        FuncDefOp tgtFunc = tgtFuncRes->get();
        LLVM_DEBUG(llvm::dbgs() << "[handleRemainingUses]   call target: " << tgtFunc << '\n');
        if (tgtFunc.isExternal()) {
          // Those without a body (i.e. external implementation) present a problem because LLZK does
          // not define a memory layout for the external implementation to interpret the struct.
          return op
              ->emitOpError("as argument to a no-body free function is not supported by this pass.")
              .attachNote(c.getLoc())
              .append("used by this call");
        }

        FieldRefOpInterface paramFromField = TypeSwitch<Operation *, FieldRefOpInterface>(op)
                                                 .Case<FieldReadOp>([](auto p) { return p; })
                                                 .Case<CreateStructOp>([](auto p) {
          return findOpThatStoresSubcmp(p, [&p]() { return p.emitOpError(); }).value_or(nullptr);
        }).Default([](Operation *p) {
          llvm::errs() << "Encountered unexpected op: "
                       << (p ? p->getName().getStringRef() : "<<null>>") << '\n';
          llvm_unreachable("Unexpected op kind");
          return nullptr;
        });
        LLVM_DEBUG({
          llvm::dbgs() << "[handleRemainingUses]   field ref op for param: "
                       << (paramFromField ? debug::toStringOne(paramFromField) : "<<null>>")
                       << '\n';
        });
        if (!paramFromField) {
          return failure(); // error already printed within findOpThatStoresSubcmp()
        }
        const SrcStructFieldToCloneInDest &newFields =
            destToSrcToClone.at(getDef(tables, paramFromField));
        LLVM_DEBUG({
          llvm::dbgs() << "[handleRemainingUses]   fields to split: "
                       << debug::toStringList(newFields) << '\n';
        });

        // Convert the FuncDefOp side first (to use the easier builder for the new CallOp).
        splitFunctionParam(tgtFunc, argIdx, newFields);
        LLVM_DEBUG({
          llvm::dbgs() << "[handleRemainingUses]   UPDATED call target: " << tgtFunc << '\n';
          llvm::dbgs() << "[handleRemainingUses]   UPDATED call target type: "
                       << tgtFunc.getFunctionType() << '\n';
        });

        // Convert the CallOp side. Add a FieldReadOp for each value from the struct and pass them
        // individually in place of the struct parameter.
        {
          OpBuilder builder(c);
          SmallVector<Value> splitArgs;
          // Before the CallOp, insert a read from every new field. These Values will replace the
          // original argument in the CallOp.
          Value originalBaseVal = paramFromField.getComponent();
          for (auto [origName, newFieldRef] : newFields) {
            splitArgs.push_back(builder.create<FieldReadOp>(
                c.getLoc(), newFieldRef.getType(), originalBaseVal, newFieldRef.getNameAttr()
            ));
          }
          // Generate the new argument list from the original but replace 'argIdx'
          SmallVector<Value> newOpArgs(c.getArgOperands());
          newOpArgs.insert(
              newOpArgs.erase(newOpArgs.begin() + argIdx), splitArgs.begin(), splitArgs.end()
          );
          // Create the new CallOp, replace uses of the old with the new, delete the old
          c.replaceAllUsesWith(builder.create<CallOp>(
              c.getLoc(), tgtFunc, CallOp::toVectorOfValueRange(c.getMapOperands()),
              c.getNumDimsPerMapAttr(), newOpArgs
          ));
          c.erase();
        }
        LLVM_DEBUG({
          llvm::dbgs() << "[handleRemainingUses]   UPDATED function: "
                       << op->getParentOfType<FuncDefOp>() << '\n';
        });
      } else {
        Operation *user = use.getOwner();
        // Report an error for any user other than some field ref that will be deleted anyway.
        if (!opWillBeDeleted(user)) {
          return op->emitOpError()
              .append(
                  "with use in '", user->getName().getStringRef(),
                  "' is not (currently) supported by this pass."
              )
              .attachNote(user->getLoc())
              .append("used by this call");
        }
      }
    }
    // Ensure that all users of the 'op' were deleted above, or will be per 'otherRefsToBeDeleted'.
    if (!op->use_empty()) {
      for (Operation *user : op->getUsers()) {
        if (!opWillBeDeleted(user)) {
          llvm::errs() << "Op has remaining use(s) that could not be removed: " << *op << '\n';
          llvm_unreachable("Expected all uses to be removed");
        }
      }
    }
    return success();
  }

  inline static LogicalResult finalizeStruct(
      SymbolTableCollection &tables, StructDefOp caller, PendingErasure &&toDelete,
      DestToSrcToClonedSrcInDest &&destToSrcToClone
  ) {
    LLVM_DEBUG({
      llvm::dbgs() << "[finalizeStruct] dumping 'caller' struct before compressing chains:\n";
      llvm::dbgs() << caller << '\n';
    });

    // Compress chains of reads that result after inlining multiple callees.
    caller.getConstrainFuncOp().walk([&tables, &destToSrcToClone](FieldReadOp readOp) {
      combineReadChain(readOp, tables, destToSrcToClone);
    });
    auto res = caller.getComputeFuncOp().walk([&tables, &destToSrcToClone](FieldReadOp readOp) {
      combineReadChain(readOp, tables, destToSrcToClone);
      LogicalResult res = combineNewThenReadChain(readOp, tables, destToSrcToClone);
      return failed(res) ? WalkResult::interrupt() : WalkResult::advance();
    });
    if (res.wasInterrupted()) {
      return failure(); // error already printed within combineNewThenReadChain()
    }

    LLVM_DEBUG({
      llvm::dbgs() << "[finalizeStruct] dumping 'caller' struct before deleting ops:\n";
      llvm::dbgs() << caller << '\n';
      llvm::dbgs() << "[finalizeStruct] ops marked for deletion:\n";
      for (FieldRefOpInterface op : toDelete.fieldRefOps) {
        llvm::dbgs().indent(2) << op << '\n';
      }
      for (CreateStructOp op : toDelete.newStructOps) {
        llvm::dbgs().indent(2) << op << '\n';
      }
      for (DestFieldWithSrcStructType op : toDelete.fieldDefs) {
        llvm::dbgs().indent(2) << op << '\n';
      }
    });

    // Handle remaining uses of CreateStructOp before deleting anything because this process
    // needs to be able to find the writes that stores the result of these ops.
    for (CreateStructOp op : toDelete.newStructOps) {
      if (failed(handleRemainingUses(op, tables, destToSrcToClone, toDelete.fieldRefOps))) {
        return failure(); // error already printed within handleRemainingUses()
      }
    }
    // Next, to avoid "still has uses" errors, must erase FieldRefOpInterface before erasing
    // the CreateStructOp or FieldDefOp.
    for (FieldRefOpInterface op : toDelete.fieldRefOps) {
      if (failed(handleRemainingUses(op, tables, destToSrcToClone))) {
        return failure(); // error already printed within handleRemainingUses()
      }
      op.erase();
    }
    for (CreateStructOp op : toDelete.newStructOps) {
      op.erase();
    }
    // Finally, erase FieldDefOp via SymbolTable so table itself is updated too.
    SymbolTable &callerSymTab = tables.getSymbolTable(caller);
    for (DestFieldWithSrcStructType op : toDelete.fieldDefs) {
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
    FailureOr<InliningPlan> plan = makePlan(useGraph, tables);
    if (failed(plan)) {
      signalPassFailure(); // error already printed w/in makePlan()
      return;
    }

    for (auto &[caller, callees] : plan.value()) {
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
      LogicalResult finalizeResult =
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
