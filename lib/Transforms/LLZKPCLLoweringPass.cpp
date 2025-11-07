//===-- LLZKPCLLoweringPass.cpp --------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the `-llzk-pcl-lowering` pass.
///
//===----------------------------------------------------------------------===//

#include "llzk/Config/Config.h"
#include "llzk/Dialect/Array/IR/Ops.h"
#include "llzk/Dialect/Bool/IR/Ops.h"
#include "llzk/Dialect/Cast/IR/Ops.h"
#include "llzk/Dialect/Constrain/IR/Ops.h"
#include "llzk/Dialect/Felt/IR/Ops.h"
#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Transforms/LLZKLoweringUtils.h"
#include "llzk/Transforms/LLZKTransformationPasses.h"
#include "llzk/Util/DynamicAPIntHelper.h"
#include "r1cs/Dialect/IR/Attrs.h"
#include "r1cs/Dialect/IR/Ops.h"
#include "r1cs/Dialect/IR/Types.h"

#if LLZK_WITH_PCL
#include <pcl/Dialect/IR/Dialect.h>
#include <pcl/Dialect/IR/Ops.h>
#include <pcl/Dialect/IR/Types.h>

#include <mlir/Dialect/Func/IR/FuncOps.h>
#endif // LLZK_WITH_PCL

#include <mlir/IR/BuiltinOps.h>

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseMapInfo.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Debug.h>

#include <deque>
#include <memory>

// Include the generated base pass class definitions.
namespace llzk {
#define GEN_PASS_DECL_PCLLOWERINGPASS
#define GEN_PASS_DEF_PCLLOWERINGPASS
#include "llzk/Transforms/LLZKTransformationPasses.h.inc"
} // namespace llzk

using namespace mlir;
using namespace llzk;
using namespace llzk::cast;
using namespace llzk::boolean;
using namespace llzk::constrain;
using namespace llzk::felt;
using namespace llzk::function;
using namespace llzk::component;
using namespace llzk::constrain;

namespace {
#if LLZK_WITH_PCL

static FailureOr<Value> lookup(Value v, llvm::DenseMap<Value, Value> &m, Operation *onError) {
  if (auto it = m.find(v); it != m.end()) {
    return it->second;
  }
  return onError->emitError("missing operand mapping"), failure();
}

static void rememberResult(Value from, Value to, llvm::DenseMap<Value, Value> &m) {
  (void)m.try_emplace(from, to);
}

// Convert binary LLZK op to corresponding binary PCL op
template <typename SrcBinOp, typename DstBinOp>
static LogicalResult
lowerBinaryLike(OpBuilder &b, SrcBinOp src, llvm::DenseMap<Value, Value> &mapping) {
  auto loc = src.getLoc();
  auto lhs = lookup(src.getLhs(), mapping, src.getOperation());
  if (failed(lhs)) {
    return failure();
  }
  auto rhs = lookup(src.getRhs(), mapping, src.getOperation());
  if (failed(rhs)) {
    return failure();
  }

  auto dst = b.create<DstBinOp>(loc, *lhs, *rhs);
  rememberResult(src.getResult(), dst.getRes(), mapping);
  return success();
}

static LogicalResult
lowerConst(OpBuilder &b, FeltConstantOp cst, llvm::DenseMap<Value, Value> &mapping) {
  auto attr = pcl::FeltAttr::get(b.getContext(), cst.getValue());
  auto dst = b.create<pcl::ConstOp>(cst.getLoc(), attr);
  rememberResult(cst.getResult(), dst.getRes(), mapping);
  return success();
}

class PCLLoweringPass : public llzk::impl::PCLLoweringPassBase<PCLLoweringPass> {

private:
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<pcl::PCLDialect, func::FuncDialect>();
  }

  /// The translation only works now on LLZK structs where all the fields are structs
  LogicalResult validateStruct(StructDefOp structDef) {
    for (auto field : structDef.getFieldDefs()) {
      auto fieldType = field.getType();
      if (!llvm::dyn_cast<FeltType>(fieldType)) {
        return structDef.emitError() << "Field must be felt type. Found " << fieldType
                                     << " for field: " << field.getName();
      }
    }
    return success();
  }

  /// Emit assertions for an equality `lhs == rhs`, with fast paths when one side
  /// is a boolean and the other side is a constant {0,1}.
  ///
  /// Cases handled:
  ///   - bool == 1  → assert(bool)
  ///   - 1 == bool  → assert(bool)
  ///   - bool == 0  → assert(!bool)
  ///   - 0 == bool  → assert(!bool)
  ///   - otherwise  → assert(lhs == rhs)
  ///
  /// Returns success after emitting IR.
  static LogicalResult
  emitAssertEqOptimized(OpBuilder &b, Location loc, Value lhsVal, Value rhsVal) {
    // --- Small helpers --------------------------------------------------------
    auto isBool = [](mlir::Value v) { return llvm::isa<pcl::BoolType>(v.getType()); };

    auto getConstAPInt = [](Value v) -> std::optional<llvm::APInt> {
      if (auto c = llvm::dyn_cast_or_null<pcl::ConstOp>(v.getDefiningOp())) {
        // Chain: ConstOp -> FeltAttr (or BoolAttr-as-int) -> IntegerAttr -> APInt
        return c.getValue().getValue().getValue();
      }
      return std::nullopt;
    };

    auto isConstOne = [&](mlir::Value v) {
      if (auto ap = getConstAPInt(v)) {
        return ap->isOne();
      }
      return false;
    };
    auto isConstZero = [&](mlir::Value v) {
      if (auto ap = getConstAPInt(v)) {
        return ap->isZero();
      }
      return false;
    };

    auto emitEqAssert = [&](mlir::Value l, mlir::Value r) {
      auto eq = b.create<pcl::CmpEqOp>(loc, l, r);
      b.create<pcl::AssertOp>(loc, eq.getRes());
    };

    auto emitAssertTrue = [&](mlir::Value pred) { b.create<pcl::AssertOp>(loc, pred); };

    auto emitAssertFalse = [&](mlir::Value pred) {
      auto neg = b.create<pcl::NotOp>(loc, pred);
      b.create<pcl::AssertOp>(loc, neg.getRes());
    };

    // Optimized handling of boolean patterns
    if (isBool(lhsVal) && isConstOne(rhsVal)) {
      // bool == 1 → assert(bool)
      emitAssertTrue(lhsVal);
      return mlir::success();
    }
    if (isBool(rhsVal) && isConstOne(lhsVal)) {
      // 1 == bool → assert(bool)
      emitAssertTrue(rhsVal);
      return mlir::success();
    }
    if (isBool(lhsVal) && isConstZero(rhsVal)) {
      // bool == 0 → assert(!bool)
      emitAssertFalse(lhsVal);
      return mlir::success();
    }
    if (isBool(rhsVal) && isConstZero(lhsVal)) {
      // 0 == bool → assert(!bool)
      emitAssertFalse(rhsVal);
      return mlir::success();
    }

    // Fallback to assert(lhs == rhs)
    emitEqAssert(lhsVal, rhsVal);
    return mlir::success();
  }

  /// Lower the constraint ops to PCL opts
  LogicalResult lowerStructToPCLBody(
      StructDefOp structDef, func::FuncOp dstFunc, llvm::DenseMap<Value, Value> &llzkToPcl
  ) {
    OpBuilder b(dstFunc.getBody());
    // Map field name to PCL vars; public fields are outputs, privates are intermediates
    llvm::DenseMap<StringRef, Value> field2pclvar;
    llvm::SmallVector<Value> outVars;

    auto srcFunc = structDef.getConstrainFuncOp();
    auto srcArgs = srcFunc.getArguments().drop_front();
    auto dstArgs = dstFunc.getArguments();
    if (dstArgs.size() != srcArgs.size()) {
      return srcFunc.emitError("arg count mismatch after dropping self");
    }

    // 1-1 mapping of args from constraint args to PCL args
    for (auto [src, dst] : llvm::zip(srcArgs, dstArgs)) {
      llzkToPcl.try_emplace(src, dst);
    }
    for (auto fieldDef : structDef.getFieldDefs()) {
      // Create a PCL var for each struct field. Public fields are outputs in PCL
      auto pclVar =
          b.create<pcl::VarOp>(fieldDef.getLoc(), fieldDef.getName(), fieldDef.hasPublicAttr());
      field2pclvar.insert({fieldDef.getName(), pclVar});
      if (fieldDef.hasPublicAttr()) {
        outVars.push_back(pclVar);
      }
    }

    Block &srcEntry = srcFunc.getBody().front();
    // Translate each op. Almost 1-1 and currently only support Felt ops.
    // TODO: support calls.
    for (Operation &op : srcEntry) {
      LogicalResult res = success();
      llvm::TypeSwitch<Operation *, void>(&op)
          .Case<FeltConstantOp>([&b, &llzkToPcl, &res](auto c) {
        res = lowerConst(b, c, llzkToPcl);
      })
          .Case<AddFeltOp>([&b, &llzkToPcl, &res](auto a) {
        res = lowerBinaryLike<AddFeltOp, pcl::AddOp>(b, a, llzkToPcl);
      })
          .Case<SubFeltOp>([&b, &llzkToPcl, &res](auto s) {
        res = lowerBinaryLike<SubFeltOp, pcl::SubOp>(b, s, llzkToPcl);
      })
          .Case<MulFeltOp>([&b, &llzkToPcl, &res](auto m) {
        res = lowerBinaryLike<MulFeltOp, pcl::MulOp>(b, m, llzkToPcl);
      })
          .Case<IntToFeltOp>([&llzkToPcl, &res](IntToFeltOp m) {
        auto arg = lookup(m.getValue(), llzkToPcl, m.getOperation());
        if (failed(arg)) {
          res = failure();
          return;
        }
        rememberResult(m.getResult(), arg.value(), llzkToPcl);
      })
          .Case<CmpOp>([&b, &llzkToPcl, &res](CmpOp m) {
        auto pred = m.getPredicate();
        switch (pred) {
        case FeltCmpPredicate::EQ:
          // handle equality
          res = lowerBinaryLike<CmpOp, pcl::CmpEqOp>(b, m, llzkToPcl);
          break;
        case FeltCmpPredicate::NE:
          // handle inequality
          res = lowerBinaryLike<CmpOp, pcl::CmpEqOp>(b, m, llzkToPcl);
          break;
        case FeltCmpPredicate::LT:
          res = lowerBinaryLike<CmpOp, pcl::CmpLtOp>(b, m, llzkToPcl);
          break;
        case FeltCmpPredicate::LE:
          // handle less-than or less-equal
          res = lowerBinaryLike<CmpOp, pcl::CmpLeOp>(b, m, llzkToPcl);
          break;
        case FeltCmpPredicate::GT:
          res = lowerBinaryLike<CmpOp, pcl::CmpGtOp>(b, m, llzkToPcl);
          break;
        case FeltCmpPredicate::GE:
          res = lowerBinaryLike<CmpOp, pcl::CmpGeOp>(b, m, llzkToPcl);
          break;
        }
      })
          .Case<EmitEqualityOp>([&b, &llzkToPcl, &res](EmitEqualityOp m) {
        auto lhs = lookup(m.getLhs(), llzkToPcl, m.getOperation());
        auto rhs = lookup(m.getRhs(), llzkToPcl, m.getOperation());
        if (failed(lhs) || failed(rhs)) {
          res = failure();
          return;
        }

        Value lhsVal = *lhs, rhsVal = *rhs;
        auto loc = m.getLoc();
        if (failed(emitAssertEqOptimized(b, loc, lhsVal, rhsVal))) {
          res = failure();
          return;
        }
      })
          .Case<FieldReadOp>([&field2pclvar, &llzkToPcl, &srcFunc](FieldReadOp read) {
        // At this point every field in the struct should have a var associated with it
        // so we should simply retrieve the var associated with the field.
        assert(read.getComponent() == srcFunc.getArguments()[0]);
        if (auto it = field2pclvar.find(read.getFieldName()); it != field2pclvar.end()) {
          rememberResult(read.getResult(), it->getSecond(), llzkToPcl);
        } else {
          llvm_unreachable("Every field should have been mapped to a pcl var");
        }
      })
          .Case<ReturnOp>([&b, &outVars](ReturnOp op) {
        // We return all the output vars we defined above.
        b.create<pcl::ReturnOp>(
            op.getLoc(), (llvm::SmallVector<Value>(outVars.begin(), outVars.end()))
        );
      }).Default([](Operation *unknown) {
        unknown->emitError("unsupported op in PCL lowering: ") << unknown->getName();
      });
      if (failed(res)) {
        return failure();
      }
    }
    return success();
  }

  FailureOr<func::FuncOp> buildPCLFunc(StructDefOp structDef) {
    SmallVector<Type> pclInputTypes, pclOutputTypes;
    auto constrainFunc = structDef.getConstrainFuncOp();
    auto ctx = structDef.getContext();
    for (auto arg : constrainFunc.getArguments().drop_front()) {
      if (!llvm::dyn_cast<FeltType>(arg.getType())) {
        return constrainFunc.emitError() << "arg is expected to be a felt";
      }
      pclInputTypes.push_back(pcl::FeltType::get(ctx));
    }
    for (auto field : structDef.getFieldDefs()) {
      auto fieldType = field.getType();
      if (!llvm::dyn_cast<FeltType>(fieldType)) {
        return structDef.emitError() << "Field must be felt type. Found " << fieldType
                                     << " for field: " << field.getName();
      }
      if (field.hasPublicAttr()) {
        pclOutputTypes.push_back(pcl::FeltType::get(ctx));
      }
    }
    FunctionType fty = FunctionType::get(ctx, pclInputTypes, pclOutputTypes);
    auto func = func::FuncOp::create(constrainFunc.getLoc(), structDef.getName(), fty);
    func.addEntryBlock();
    return func;
  }

  // PCL programs require a module-level attribute specifying the prime.
  void setPrime(ModuleOp &newMod) {
    // Add an extra bit to avoid the prime being represented as a negative number
    auto newBitWidth = prime.getBitWidth() + 1;
    auto ty = IntegerType::get(newMod.getContext(), newBitWidth);
    auto intAttr = IntegerAttr::get(ty, prime.zext(newBitWidth));
    newMod->setAttr("pcl.prime", pcl::PrimeAttr::get(newMod.getContext(), intAttr));
  }

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    // check PCLDialect is loaded.
    assert(moduleOp->getContext()->getLoadedDialect<pcl::PCLDialect>() && "PCL dialect not loaded");
    // Create the PCL module
    auto newMod = ModuleOp::create(moduleOp.getLoc());
    // Set the prime attribute
    setPrime(newMod);
    // Convert each struct to a PCL function
    auto walkResult = moduleOp.walk([this, &newMod](StructDefOp structDef) -> WalkResult {
      // 1) verify the struct can be converted to PCL
      if (failed(validateStruct(structDef))) {
        signalPassFailure();
        return WalkResult::interrupt();
      }
      // 2) Construct the PCL function op but with an empty body
      FailureOr<func::FuncOp> pclFuncOp = buildPCLFunc(structDef);
      if (failed(pclFuncOp)) {
        signalPassFailure();
        return WalkResult::interrupt();
      }
      // 3) Fill in the PCL function body
      llvm::DenseMap<Value, Value> llzk2pcl;
      newMod.getBody()->push_back(*pclFuncOp);
      if (failed(lowerStructToPCLBody(structDef, pclFuncOp.value(), llzk2pcl))) {
        signalPassFailure();
        return WalkResult::interrupt();
      }

      return WalkResult::advance();
    });
    if (walkResult.wasInterrupted()) {
      return;
    }
    // clear the original ops
    moduleOp.getRegion().takeBody(newMod.getBodyRegion());
    // Replace the module attributes
    moduleOp->setAttrs(newMod->getAttrDictionary());
    newMod.erase();
  }
};
#else
class PCLLoweringPass : public llzk::impl::PCLLoweringPassBase<PCLLoweringPass> {
  void getDependentDialects(mlir::DialectRegistry &registry) const override {}

  void runOnOperation() override {}
};
#endif // LLZK_WITH_PCL
} // namespace

std::unique_ptr<mlir::Pass> llzk::createPCLLoweringPass() {
  return std::make_unique<PCLLoweringPass>();
}
