//===-- LLZKPolyLoweringPass.cpp --------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the `-llzk-poly-lowering` pass.
///
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/Array/IR/Ops.h"
#include "llzk/Dialect/Constrain/IR/Ops.h"
#include "llzk/Dialect/Felt/IR/Ops.h"
#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Transforms/LLZKTransformationPasses.h"

#include <mlir/IR/BuiltinOps.h>

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseMapInfo.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Debug.h>

#include <deque>
#include <memory>

// Include the generated base pass class definitions.
namespace llzk {
#define GEN_PASS_DECL_POLYLOWERINGPASS
#define GEN_PASS_DEF_POLYLOWERINGPASS
#include "llzk/Transforms/LLZKTransformationPasses.h.inc"
} // namespace llzk

using namespace mlir;
using namespace llzk;
using namespace llzk::felt;
using namespace llzk::function;
using namespace llzk::component;
using namespace llzk::constrain;

#define DEBUG_TYPE "llzk-poly-lowering-pass"
#define AUXILIARY_FIELD_PREFIX "__llzk_aux_field_"

namespace {

struct AuxAssignment {
  std::string auxFieldName;
  Value computedValue;
};

class PolyLoweringPass : public llzk::impl::PolyLoweringPassBase<PolyLoweringPass> {
public:
  void setMaxDegree(unsigned degree) { this->maxDegree = degree; }

private:
  unsigned auxCounter = 0;

  void collectStructDefs(ModuleOp modOp, SmallVectorImpl<StructDefOp> &structDefs) {
    modOp.walk([&structDefs](StructDefOp structDef) {
      structDefs.push_back(structDef);
      return WalkResult::skip();
    });
  }

  void addAuxField(StructDefOp structDef, StringRef name) {
    OpBuilder builder(structDef);
    builder.setInsertionPointToEnd(&structDef.getBody().front());
    builder.create<FieldDefOp>(
        structDef.getLoc(), builder.getStringAttr(name), builder.getType<FeltType>()
    );
  }

  // Recursively compute degree of FeltOps SSA values
  unsigned getDegree(Value val, DenseMap<Value, unsigned> &memo) {
    if (memo.count(val)) {
      return memo[val];
    }
    // Handle function parameters (BlockArguments)
    if (val.isa<BlockArgument>()) {
      memo[val] = 1;
      return 1;
    }
    if (val.getDefiningOp<FeltConstantOp>()) {
      return memo[val] = 0;
    }
    if (val.getDefiningOp<FeltNonDetOp>()) {
      return memo[val] = 1;
    }
    if (val.getDefiningOp<FieldReadOp>()) {
      return memo[val] = 1;
    }
    if (auto addOp = val.getDefiningOp<AddFeltOp>()) {
      return memo[val] = std::max(getDegree(addOp.getLhs(), memo), getDegree(addOp.getRhs(), memo));
    }
    if (auto subOp = val.getDefiningOp<SubFeltOp>()) {
      return memo[val] = std::max(getDegree(subOp.getLhs(), memo), getDegree(subOp.getRhs(), memo));
    }
    if (auto mulOp = val.getDefiningOp<MulFeltOp>()) {
      return memo[val] = getDegree(mulOp.getLhs(), memo) + getDegree(mulOp.getRhs(), memo);
    }
    if (auto divOp = val.getDefiningOp<DivFeltOp>()) {
      return memo[val] = getDegree(divOp.getLhs(), memo) + getDegree(divOp.getRhs(), memo);
    }
    if (auto negOp = val.getDefiningOp<NegFeltOp>()) {
      return memo[val] = getDegree(negOp.getOperand(), memo);
    }

    llvm_unreachable("Unhandled Felt SSA value in degree computation");
  }

  /// Replaces all *subsequent uses* of `oldVal` with `newVal`, starting *after* `afterOp`.
  ///
  /// Specifically:
  /// - Uses of `oldVal` in operations that come **after** `afterOp` in the same block are replaced.
  /// - Uses in `afterOp` itself are **not replaced** (to avoid self-trivializing rewrites).
  /// - Uses in other blocks are replaced (if applicable).
  ///
  /// Typical use case:
  /// - You introduce an auxiliary value (e.g., via EmitEqualityOp) and want to replace
  ///   all *later* uses of the original value while preserving the constraint itself.
  ///
  /// \param oldVal  The original value whose uses should be redirected.
  /// \param newVal  The new value to replace subsequent uses with.
  /// \param afterOp The operation after which uses of `oldVal` will be replaced.
  void replaceSubsequentUsesWith(Value oldVal, Value newVal, Operation *afterOp) {
    assert(afterOp && "afterOp must be a valid Operation*");

    for (auto &use : llvm::make_early_inc_range(oldVal.getUses())) {
      Operation *user = use.getOwner();

      // Skip uses that are:
      // - Before afterOp in the same block.
      // - Inside afterOp itself.
      if ((user->getBlock() == afterOp->getBlock()) &&
          (user->isBeforeInBlock(afterOp) || user == afterOp)) {
        continue;
      }

      // Replace this use of oldVal with newVal.
      use.set(newVal);
    }
  }

  Value lowerExpression(
      Value val, unsigned maxDegree, StructDefOp structDef, FuncDefOp constrainFunc,
      DenseMap<Value, unsigned> &degreeMemo, DenseMap<Value, Value> &rewrites,
      SmallVector<AuxAssignment> &auxAssignments
  ) {
    if (rewrites.count(val)) {
      return rewrites[val];
    }

    unsigned degree = getDegree(val, degreeMemo);
    if (degree <= maxDegree) {
      rewrites[val] = val;
      return val;
    }

    if (auto mulOp = val.getDefiningOp<MulFeltOp>()) {
      // Recursively lower operands first
      Value lhs = lowerExpression(
          mulOp.getLhs(), maxDegree, structDef, constrainFunc, degreeMemo, rewrites, auxAssignments
      );
      Value rhs = lowerExpression(
          mulOp.getRhs(), maxDegree, structDef, constrainFunc, degreeMemo, rewrites, auxAssignments
      );

      unsigned lhsDeg = getDegree(lhs, degreeMemo);
      unsigned rhsDeg = getDegree(rhs, degreeMemo);

      OpBuilder builder(mulOp.getOperation()->getBlock(), ++Block::iterator(mulOp));
      Value selfVal = constrainFunc.getArgument(0); // %self argument
      bool eraseMul = lhsDeg + rhsDeg > maxDegree;
      // Optimization: If lhs == rhs, factor it only once
      if (lhs == rhs && eraseMul) {
        std::string auxName = AUXILIARY_FIELD_PREFIX + std::to_string(this->auxCounter++);
        addAuxField(structDef, auxName);

        auto auxVal = builder.create<FieldReadOp>(
            lhs.getLoc(), lhs.getType(), selfVal, builder.getStringAttr(auxName)
        );
        auxAssignments.push_back({auxName, lhs});
        Location loc = builder.getFusedLoc({auxVal.getLoc(), lhs.getLoc()});
        auto eqOp = builder.create<EmitEqualityOp>(loc, auxVal, lhs);

        // Memoize auxVal as degree 1
        degreeMemo[auxVal] = 1;
        rewrites[lhs] = auxVal;
        rewrites[rhs] = auxVal;
        // Now selectively replace subsequent uses of lhs with auxVal
        replaceSubsequentUsesWith(lhs, auxVal, eqOp);

        // Update lhs and rhs to use auxVal
        lhs = auxVal;
        rhs = auxVal;

        lhsDeg = rhsDeg = 1;
      }
      // While their product exceeds maxDegree, factor out one side
      while (lhsDeg + rhsDeg > maxDegree) {
        Value &toFactor = (lhsDeg >= rhsDeg) ? lhs : rhs;

        // Create auxiliary field for toFactor
        std::string auxName = AUXILIARY_FIELD_PREFIX + std::to_string(this->auxCounter++);
        addAuxField(structDef, auxName);

        // Read back as FieldReadOp (new SSA value)
        auto auxVal = builder.create<FieldReadOp>(
            toFactor.getLoc(), toFactor.getType(), selfVal, builder.getStringAttr(auxName)
        );

        // Emit constraint: auxVal == toFactor
        Location loc = builder.getFusedLoc({auxVal.getLoc(), toFactor.getLoc()});
        auto eqOp = builder.create<EmitEqualityOp>(loc, auxVal, toFactor);
        auxAssignments.push_back({auxName, toFactor});
        // Update memoization
        rewrites[toFactor] = auxVal;
        degreeMemo[auxVal] = 1; // stays same
        // replace the term with auxVal.
        replaceSubsequentUsesWith(toFactor, auxVal, eqOp);

        // Remap toFactor to auxVal for next iterations
        toFactor = auxVal;

        // Recompute degrees
        lhsDeg = getDegree(lhs, degreeMemo);
        rhsDeg = getDegree(rhs, degreeMemo);
      }

      // Now lhs * rhs fits within degree bound
      auto mulVal = builder.create<MulFeltOp>(lhs.getLoc(), lhs.getType(), lhs, rhs);
      if (eraseMul) {
        mulOp->replaceAllUsesWith(mulVal);
        mulOp->erase();
      }

      // Result of this multiply has degree lhsDeg + rhsDeg
      degreeMemo[mulVal] = lhsDeg + rhsDeg;
      rewrites[val] = mulVal;

      return mulVal;
    }

    // For non-mul ops, leave untouched (they're degree-1 safe)
    rewrites[val] = val;
    return val;
  }

  Value getSelfValueFromCompute(FuncDefOp computeFunc) {
    // Get the single block of the function body
    Region &body = computeFunc.getBody();
    assert(!body.empty() && "compute() function body is empty");

    Block &block = body.front();

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

    // Return its operands as SmallVector<Value>
    return retOp.getOperands().front();
  }

  Value rebuildExprInCompute(
      Value val, FuncDefOp computeFunc, OpBuilder &builder, DenseMap<Value, Value> &rebuildMemo
  ) {
    // Memoized already?
    if (auto it = rebuildMemo.find(val); it != rebuildMemo.end()) {
      return it->second;
    }

    // Case 1: BlockArgument from constrain() -> map to compute()
    if (auto barg = val.dyn_cast<BlockArgument>()) {
      unsigned index = barg.getArgNumber();                  // Argument index in constrain()
      Value computeArg = computeFunc.getArgument(index - 1); // Corresponding compute() arg
      rebuildMemo[val] = computeArg;
      return computeArg;
    }

    // Case 2: FieldReadOp in constrain() -> replicate FieldReadOp in compute()
    if (auto readOp = val.getDefiningOp<FieldReadOp>()) {
      Value selfVal = getSelfValueFromCompute(computeFunc); // %self is always the return value
      auto rebuiltRead = builder.create<FieldReadOp>(
          readOp.getLoc(), readOp.getType(), selfVal, readOp.getFieldNameAttr().getAttr()
      );
      rebuildMemo[val] = rebuiltRead.getResult();
      return rebuiltRead.getResult();
    }

    // Case 3: AddFeltOp
    if (auto addOp = val.getDefiningOp<AddFeltOp>()) {
      Value lhs = rebuildExprInCompute(addOp.getLhs(), computeFunc, builder, rebuildMemo);
      Value rhs = rebuildExprInCompute(addOp.getRhs(), computeFunc, builder, rebuildMemo);
      auto rebuiltAdd = builder.create<AddFeltOp>(addOp.getLoc(), addOp.getType(), lhs, rhs);
      rebuildMemo[val] = rebuiltAdd.getResult();
      return rebuiltAdd.getResult();
    }

    // Case 4: SubFeltOp
    if (auto subOp = val.getDefiningOp<SubFeltOp>()) {
      Value lhs = rebuildExprInCompute(subOp.getLhs(), computeFunc, builder, rebuildMemo);
      Value rhs = rebuildExprInCompute(subOp.getRhs(), computeFunc, builder, rebuildMemo);
      auto rebuiltSub = builder.create<SubFeltOp>(subOp.getLoc(), subOp.getType(), lhs, rhs);
      rebuildMemo[val] = rebuiltSub.getResult();
      return rebuiltSub.getResult();
    }

    // Case 5: MulFeltOp
    if (auto mulOp = val.getDefiningOp<MulFeltOp>()) {
      Value lhs = rebuildExprInCompute(mulOp.getLhs(), computeFunc, builder, rebuildMemo);
      Value rhs = rebuildExprInCompute(mulOp.getRhs(), computeFunc, builder, rebuildMemo);
      auto rebuiltMul = builder.create<MulFeltOp>(mulOp.getLoc(), mulOp.getType(), lhs, rhs);
      rebuildMemo[val] = rebuiltMul.getResult();
      return rebuiltMul.getResult();
    }

    // Case 6: NegFeltOp
    if (auto negOp = val.getDefiningOp<NegFeltOp>()) {
      Value inner = rebuildExprInCompute(negOp.getOperand(), computeFunc, builder, rebuildMemo);
      auto rebuiltNeg = builder.create<NegFeltOp>(negOp.getLoc(), negOp.getType(), inner);
      rebuildMemo[val] = rebuiltNeg.getResult();
      return rebuiltNeg.getResult();
    }

    // Case 7: DivFeltOp
    if (auto divOp = val.getDefiningOp<DivFeltOp>()) {
      Value lhs = rebuildExprInCompute(divOp.getLhs(), computeFunc, builder, rebuildMemo);
      Value rhs = rebuildExprInCompute(divOp.getRhs(), computeFunc, builder, rebuildMemo);
      auto rebuiltDiv = builder.create<DivFeltOp>(divOp.getLoc(), divOp.getType(), lhs, rhs);
      rebuildMemo[val] = rebuiltDiv.getResult();
      return rebuiltDiv.getResult();
    }

    // Case 8: ConstFeltOp
    if (auto constOp = val.getDefiningOp<FeltConstantOp>()) {
      auto newConst = builder.create<FeltConstantOp>(constOp.getLoc(), constOp.getValue());
      rebuildMemo[val] = newConst.getResult();
      return newConst.getResult();
    }

    llvm::errs() << "Unhandled expression kind in rebuildExprInCompute: " << val << "\n";
    llvm_unreachable("Unsupported op in rebuildExprInCompute");
  }

  // Throw an error if the struct has a field that matches the prefix of the auxiliary fields
  // we use in the pass. There **shouldn't** be a conflict but just in case let's throw the check.
  void checkForAuxFieldConflicts(StructDefOp structDef) {
    structDef.walk([&](FieldDefOp fieldDefOp) {
      if (fieldDefOp.getName().starts_with(AUXILIARY_FIELD_PREFIX)) {
        fieldDefOp.emitError() << "Field name: \"" << fieldDefOp.getName()
                               << "\" starts with prefix: " << AUXILIARY_FIELD_PREFIX
                               << " which is reserved for lowering pass";
        signalPassFailure();
        return;
      }
    });
  }

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();

    // Validate degree parameter
    if (maxDegree < 2) {
      moduleOp.emitError() << "Invalid max degree: " << maxDegree.getValue() << ". Must be >= 2.";
      signalPassFailure();
      return;
    }

    moduleOp.walk([&](StructDefOp structDef) {
      FuncDefOp constrainFunc = structDef.getConstrainFuncOp();
      FuncDefOp computeFunc = structDef.getComputeFuncOp();
      if (!constrainFunc) {
        structDef.emitOpError() << "\"" << structDef.getName() << "\" doesn't have a '"
                                << FUNC_NAME_CONSTRAIN << "' function";
        signalPassFailure();
        return;
      }

      if (!computeFunc) {
        structDef.emitOpError() << "\"" << structDef.getName() << "\" doesn't have a '"
                                << FUNC_NAME_COMPUTE << "' function";
        signalPassFailure();
        return;
      }

      checkForAuxFieldConflicts(structDef);

      DenseMap<Value, unsigned> degreeMemo;
      DenseMap<Value, Value> rewrites;
      SmallVector<AuxAssignment> auxAssignments;

      // Lower equality constraints
      constrainFunc.walk([&](EmitEqualityOp constraintOp) {
        auto &lhsOperand = constraintOp.getLhsMutable();
        auto &rhsOperand = constraintOp.getRhsMutable();
        unsigned degreeLhs = getDegree(lhsOperand.get(), degreeMemo);
        unsigned degreeRhs = getDegree(rhsOperand.get(), degreeMemo);

        if (degreeLhs > maxDegree) {
          Value loweredExpr = lowerExpression(
              lhsOperand.get(), maxDegree, structDef, constrainFunc, degreeMemo, rewrites,
              auxAssignments
          );
          lhsOperand.set(loweredExpr);
        }
        if (degreeRhs > maxDegree) {
          Value loweredExpr = lowerExpression(
              rhsOperand.get(), maxDegree, structDef, constrainFunc, degreeMemo, rewrites,
              auxAssignments
          );
          rhsOperand.set(loweredExpr);
        }
      });

      // The pass doesn't currently support EmitContainmentOp as it depends on
      // https://veridise.atlassian.net/browse/LLZK-245 being fixed Once this is fixed, the op
      // should lower all the elements in the row being looked up
      constrainFunc.walk([&](EmitContainmentOp containOp) {
        moduleOp.emitError() << "EmitContainmentOp is unsupported for now in the lowering pass";
        signalPassFailure();
        return;
      });

      // Lower function call arguments
      constrainFunc.walk([&](CallOp callOp) {
        if (callOp.calleeIsStructConstrain()) {
          SmallVector<Value> newOperands = llvm::to_vector(callOp.getArgOperands());
          bool modified = false;

          for (Value &arg : newOperands) {
            unsigned deg = getDegree(arg, degreeMemo);

            if (deg > 1) {
              Value loweredArg = lowerExpression(
                  arg, maxDegree, structDef, constrainFunc, degreeMemo, rewrites, auxAssignments
              );
              arg = loweredArg;
              modified = true;
            }
          }

          if (modified) {
            SmallVector<ValueRange> mapOperands;
            OpBuilder builder(callOp);
            for (auto group : callOp.getMapOperands()) {
              mapOperands.push_back(group);
            }

            builder.create<CallOp>(
                callOp.getLoc(), callOp.getResultTypes(), callOp.getCallee(), mapOperands,
                callOp.getNumDimsPerMap(), newOperands
            );
            callOp->erase();
          }
        }
      });

      DenseMap<Value, Value> rebuildMemo;
      Block &computeBlock = computeFunc.getBody().front();
      OpBuilder builder(&computeBlock, computeBlock.getTerminator()->getIterator());
      Value selfVal = getSelfValueFromCompute(computeFunc);

      for (const auto &assign : auxAssignments) {
        Value rebuiltExpr =
            rebuildExprInCompute(assign.computedValue, computeFunc, builder, rebuildMemo);
        builder.create<FieldWriteOp>(
            assign.computedValue.getLoc(), selfVal, builder.getStringAttr(assign.auxFieldName),
            rebuiltExpr
        );
      }
    });
  }
};
} // namespace

std::unique_ptr<mlir::Pass> llzk::createPolyLoweringPass() {
  return std::make_unique<PolyLoweringPass>();
};

std::unique_ptr<mlir::Pass> llzk::createPolyLoweringPass(unsigned maxDegree) {
  auto pass = std::make_unique<PolyLoweringPass>();
  static_cast<PolyLoweringPass *>(pass.get())->setMaxDegree(maxDegree);
  return pass;
}
