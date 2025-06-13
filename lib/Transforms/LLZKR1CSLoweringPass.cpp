//===-- LLZKR1CSLoweringPass.cpp --------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the `-llzk-r1cs-lowering` pass.
///
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/Array/IR/Ops.h"
#include "llzk/Dialect/Constrain/IR/Ops.h"
#include "llzk/Dialect/Felt/IR/Ops.h"
#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Transforms/LLZKLoweringUtils.h"
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
#define GEN_PASS_DEF_R1CSLOWERINGPASS
#include "llzk/Transforms/LLZKTransformationPasses.h.inc"
} // namespace llzk

using namespace mlir;
using namespace llzk;
using namespace llzk::felt;
using namespace llzk::function;
using namespace llzk::component;
using namespace llzk::constrain;

#define DEBUG_TYPE "llzk-r1cs-lowering-pass"
#define R1CS_AUXILIARY_FIELD_PREFIX "__llzk_r1cs_lowering_pass_aux_field_"

namespace {

class R1CSLoweringPass : public llzk::impl::R1CSLoweringPassBase<R1CSLoweringPass> {
private:
  unsigned auxCounter = 0;

  // Normalize a felt-valued expression into R1CS-compatible form.
  // This performs *minimal* rewriting:
  // - Only rewrites Add/Sub of two degree-2 terms
  // - Operates bottom-up using post-order traversal
  //
  // Resulting expression is R1CS-compatible (i.e., one multiplication per constraint)
  // and can be directly used in EmitEqualityOp or as operands of other expressions.

  void getPostOrder(Value root, SmallVectorImpl<Value> &postOrder) {
    SmallVector<Value, 16> worklist;
    DenseSet<Value> visited;

    worklist.push_back(root);

    while (!worklist.empty()) {
      Value val = worklist.back();

      if (visited.contains(val)) {
        worklist.pop_back();
        postOrder.push_back(val);
        continue;
      }

      visited.insert(val);
      if (Operation *op = val.getDefiningOp()) {
        for (Value operand : op->getOperands()) {
          worklist.push_back(operand);
        }
      }
    }
  }

  /// Normalize a felt-valued expression into R1CS-compatible form by rewriting
  /// only when strictly necessary. This function ensures the resulting expression:
  ///
  /// - Has at most one multiplication per constraint (R1CS-compatible)
  /// - Avoids unnecessary introduction of auxiliary variables
  /// - Preserves semantic equivalence via auxiliary field equality constraints
  ///
  /// Rewriting is done **bottom-up** using post-order traversal of the def-use chain.
  /// The transformation is minimal:
  /// - Only rewrites Add/Sub where both operands are degree-2
  /// - Leaves multiplications intact unless their operands require rewriting due to constants
  /// - Avoids rewriting expressions that are already linear or already normalized
  ///
  /// The function memoizes all degrees and rewrites for efficiency and correctness,
  /// and records any auxiliary field assignments for later reconstruction in compute().
  ///
  /// \param root           The root felt-valued expression to normalize.
  /// \param structDef      The enclosing struct definition (for adding aux fields).
  /// \param constrainFunc  The constrain() function containing the constraint logic.
  /// \param degreeMemo     Memoized degrees of expressions (to avoid recomputation).
  /// \param rewrites       Memoized rewrites of expressions.
  /// \param auxAssignments Records auxiliary field assignments introduced during normalization.
  /// \param builder        Builder used to insert new ops in the constrain() block.
  /// \returns              A Value representing the normalized (possibly rewritten) expression.
  Value normalizeForR1CS(
      Value root, StructDefOp structDef, FuncDefOp constrainFunc,
      DenseMap<Value, unsigned> &degreeMemo, DenseMap<Value, Value> &rewrites,
      SmallVectorImpl<AuxAssignment> &auxAssignments, OpBuilder &builder
  ) {
    if (rewrites.count(root)) {
      return rewrites[root];
    }

    SmallVector<Value, 16> postOrder;
    getPostOrder(root, postOrder);

    // We perform a bottom up rewrite of the expressions. For any expression e := op(e_1, ...,
    // e_n) we first rewrite e_1, ..., e_n if necessary and then rewrite e based on op.
    for (Value val : postOrder) {
      if (rewrites.count(val)) {
        continue;
      }

      Operation *op = val.getDefiningOp();

      if (!op) {
        // Block arguments, etc.
        degreeMemo[val] = 1;
        rewrites[val] = val;
        continue;
      }

      // Case 1: Felt constant op. The degree is 0 and no rewrite is needed.
      if (auto c = val.getDefiningOp<FeltConstantOp>()) {
        degreeMemo[val] = 0;
        rewrites[val] = val;
        continue;
      }

      // Case 2: Field read op. The degree is 1 and no rewrite needed.
      if (auto fr = val.getDefiningOp<FieldReadOp>()) {
        degreeMemo[val] = 1;
        rewrites[val] = val;
        continue;
      }

      // Helper function for getting degree from memo map
      auto getDeg = [&](Value v) -> unsigned {
        auto it = degreeMemo.find(v);
        assert(it != degreeMemo.end() && "Missing degree");
        return it->second;
      };

      // Case 3: lhs +/- rhs. There are three subcases cases to consider:
      // 1) If deg(lhs) <= degree(rhs) < 2 then nothing needs to be done
      // 2) If deg(lhs) = 2 and degree(rhs) < 2 then nothing further has to be done.
      // 3) If deg(lhs) = deg(rhs) = 2 then we lower one of lhs or rhs.
      auto handleAddOrSub = [&](Value lhsOrig, Value rhsOrig, bool isAdd) {
        Value lhs = rewrites[lhsOrig];
        Value rhs = rewrites[rhsOrig];
        unsigned degLhs = getDeg(lhs);
        unsigned degRhs = getDeg(rhs);

        if (degLhs == 2 && degRhs == 2) {
          builder.setInsertionPoint(op);
          std::string auxName = R1CS_AUXILIARY_FIELD_PREFIX + std::to_string(auxCounter++);
          addAuxField(structDef, auxName);
          Value self = constrainFunc.getArgument(0);
          Value aux = builder.create<FieldReadOp>(
              val.getLoc(), val.getType(), self, builder.getStringAttr(auxName)
          );
          auto eqOp = builder.create<EmitEqualityOp>(val.getLoc(), aux, lhs);
          auxAssignments.push_back({auxName, lhs});
          degreeMemo[aux] = 1;
          rewrites[aux] = aux;
          replaceSubsequentUsesWith(lhs, aux, eqOp);
          lhs = aux;
          degLhs = 1;

          Operation *newOp = isAdd
                                 ? builder.create<AddFeltOp>(val.getLoc(), val.getType(), lhs, rhs)
                                 : builder.create<SubFeltOp>(val.getLoc(), val.getType(), lhs, rhs);
          Value result = newOp->getResult(0);
          degreeMemo[result] = std::max(degLhs, degRhs);
          rewrites[val] = result;
          rewrites[result] = result;
          val.replaceAllUsesWith(result);
          if (val.use_empty()) {
            op->erase();
          }
        } else {
          degreeMemo[val] = std::max(degLhs, degRhs);
          rewrites[val] = val;
        }
      };

      if (auto add = dyn_cast<AddFeltOp>(op)) {
        handleAddOrSub(add.getLhs(), add.getRhs(), /*isAdd=*/true);
        continue;
      }

      if (auto sub = dyn_cast<SubFeltOp>(op)) {
        handleAddOrSub(sub.getLhs(), sub.getRhs(), /*isAdd=*/false);
        continue;
      }

      // Case 4: lhs * rhs. Nothing further needs to be done assuming the degree lowering pass has
      // been run with maxDegree = 2. This is because both operands are normalized and at most one
      // operand can be quadratic.
      if (auto mul = val.getDefiningOp<MulFeltOp>()) {
        Value lhs = rewrites[mul.getLhs()];
        Value rhs = rewrites[mul.getRhs()];
        unsigned degLhs = getDeg(lhs);
        unsigned degRhs = getDeg(rhs);

        degreeMemo[val] = degLhs + degRhs;
        rewrites[val] = val;
        continue;
      }

      // Case 6: Neg. Similar to multiplication, nothing needs to be done since we are doing the
      // rewrite bottom up
      if (auto neg = val.getDefiningOp<NegFeltOp>()) {
        Value inner = rewrites[neg.getOperand()];
        unsigned deg = getDeg(inner);
        degreeMemo[val] = deg;
        rewrites[val] = val;
        continue;
      }

      llvm::errs() << "Unhandled op in normalize ForR1CS: " << *op << "\n";
      signalPassFailure();
    }

    return rewrites[root];
  }

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();

    moduleOp.walk([&](StructDefOp structDef) {
      FuncDefOp constrainFunc = structDef.getConstrainFuncOp();
      FuncDefOp computeFunc = structDef.getComputeFuncOp();
      if (!constrainFunc || !computeFunc) {
        structDef.emitOpError("Missing compute or constrain function");
        signalPassFailure();
        return;
      }

      if (failed(checkForAuxFieldConflicts(structDef, R1CS_AUXILIARY_FIELD_PREFIX))) {
        signalPassFailure();
        return;
      }

      DenseMap<Value, unsigned> degreeMemo;
      DenseMap<Value, Value> rewrites;
      SmallVector<AuxAssignment> auxAssignments;

      constrainFunc.walk([&](EmitEqualityOp eqOp) {
        OpBuilder builder(eqOp);
        Value lhs = normalizeForR1CS(
            eqOp.getLhs(), structDef, constrainFunc, degreeMemo, rewrites, auxAssignments, builder
        );
        Value rhs = normalizeForR1CS(
            eqOp.getRhs(), structDef, constrainFunc, degreeMemo, rewrites, auxAssignments, builder
        );

        unsigned degLhs = degreeMemo.lookup(lhs);
        unsigned degRhs = degreeMemo.lookup(rhs);

        // If both sides are degree 2, isolate one side
        if (degLhs == 2 && degRhs == 2) {
          builder.setInsertionPoint(eqOp);
          std::string auxName = R1CS_AUXILIARY_FIELD_PREFIX + std::to_string(auxAssignments.size());
          addAuxField(structDef, auxName);
          Value self = constrainFunc.getArgument(0);
          Value aux = builder.create<FieldReadOp>(
              eqOp.getLoc(), lhs.getType(), self, builder.getStringAttr(auxName)
          );
          auto eqAux = builder.create<EmitEqualityOp>(eqOp.getLoc(), aux, lhs);
          auxAssignments.push_back({auxName, lhs});
          degreeMemo[aux] = 1;
          replaceSubsequentUsesWith(lhs, aux, eqAux);
          lhs = aux;
        }

        builder.create<EmitEqualityOp>(eqOp.getLoc(), lhs, rhs);
        eqOp.erase();
      });

      Block &computeBlock = computeFunc.getBody().front();
      OpBuilder builder(&computeBlock, computeBlock.getTerminator()->getIterator());
      Value selfVal = getSelfValueFromCompute(computeFunc);
      DenseMap<Value, Value> rebuildMemo;

      for (const auto &assign : auxAssignments) {
        Value expr = rebuildExprInCompute(assign.computedValue, computeFunc, builder, rebuildMemo);
        builder.create<FieldWriteOp>(
            assign.computedValue.getLoc(), selfVal, builder.getStringAttr(assign.auxFieldName), expr
        );
      }
    });
  }
};
} // namespace

std::unique_ptr<mlir::Pass> llzk::createR1CSLoweringPass() {
  return std::make_unique<R1CSLoweringPass>();
}
