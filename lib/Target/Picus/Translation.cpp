//===-- Translation.cpp - LLZK to Picus translation -------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include <llzk/Dialect/LLZK/IR/Attrs.h>
#include <llzk/Dialect/LLZK/IR/Builders.h>
#include <llzk/Dialect/LLZK/IR/Ops.h>
#include <llzk/Dialect/LLZK/IR/Types.h>
#include <llzk/Dialect/LLZK/Util/AttributeHelper.h>
#include <llzk/Target/Picus/Export.h>
#include <llzk/Target/Picus/Language/Circuit.h>
#include <llzk/Target/Picus/Language/Expression.h>
#include <llzk/Target/Picus/Language/Statement.h>

#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/Visitors.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVectorExtras.h>
#include <llvm/ADT/TypeSwitch.h>

using namespace llzk;
using namespace mlir;
using namespace picus;

static bool validInputOperation(Operation *op) {
  if (auto modOp = mlir::dyn_cast_if_present<ModuleOp>(op)) {
    return isLLZKModule(modOp);
  }
  return false;
}

static StringAttr inputVarName(MLIRContext *ctx, unsigned argNo) {
  return StringAttr::get(ctx, "input_" + Twine(argNo));
}

static StringAttr inputVarName(MLIRContext *ctx, unsigned argNo, unsigned idx) {
  return StringAttr::get(ctx, "input_" + Twine(argNo) + "_" + Twine(idx));
}

namespace {

struct StructTranslatorHelper {
private:
  using OpTypeSwitch = llvm::TypeSwitch<Operation *, Expression::ptr>;

  template <typename BinOp, typename Expr> OpTypeSwitch *addBinOpCase(OpTypeSwitch *sw) {
    return &sw->Case<BinOp>([this](BinOp binOp) {
      return std::make_unique<Expr>(
          translateValueToExpr(binOp.getLhs()), translateValueToExpr(binOp.getRhs())
      );
    });
  }

public:
  FixedValues &fixedValues;
  Module &outMod;

  Expression::ptr translateOpToExpr(Operation *op) {
    OpTypeSwitch sw(op);
    OpTypeSwitch *p = &sw;
    p = addBinOpCase<SubFeltOp, SubExpr>(p);
    p = addBinOpCase<AddFeltOp, AddExpr>(p);
    p = addBinOpCase<MulFeltOp, MulExpr>(p);
    p = addBinOpCase<DivFeltOp, DivExpr>(p);
    return sw
        .Case<GlobalReadOp>([this](GlobalReadOp globalReadOp) {
      return fixedValues.getFixedValueRef(globalReadOp.getNameRef().getLeafReference().getValue());
    })
        .Case<FeltConstantOp>([](FeltConstantOp feltConst) {
      return std::make_unique<ConstExpr>(feltConst.getValue().getValue());
    })
        .Case<ReadArrayOp>([](ReadArrayOp readArrOp) -> Expression::ptr {
      // Only constant indexed arrays coming from arguments are currently allowed
      auto arrValue = readArrOp.getArrRef();
      if (!mlir::isa<BlockArgument>(arrValue)) {
        readArrOp->emitOpError() << "is an unsupported " << readArrOp->getName()
                                 << " operation: array must be a function argument";
        return nullptr;
      }
      auto arg = mlir::cast<BlockArgument>(arrValue);
      auto indices = readArrOp.getIndices();
      if (indices.size() != 1) {
        readArrOp->emitOpError() << "is an unsupported " << readArrOp->getName()
                                 << " operation: must have only one index";
        return nullptr;
      }
      if (auto constIndex =
              mlir::dyn_cast_if_present<arith::ConstantIndexOp>(indices.front().getDefiningOp())) {
        return std::make_unique<VarExpr>(inputVarName(
            readArrOp.getContext(), arg.getArgNumber(),
            llzk::fromAPInt(mlir::cast<IntegerAttr>(constIndex.getValue()).getValue())
        ));
      }
      readArrOp->emitOpError() << "is an unsupported " << readArrOp->getName()
                               << " operation: index must be constant";
      return nullptr;
    })
        .Case<FieldReadOp>([](FieldReadOp fieldRead) -> Expression::ptr {
      // Only reading from self is currently allowed.
      auto structRefArg = mlir::dyn_cast<BlockArgument>(fieldRead.getComponent());
      if (!structRefArg || structRefArg.getArgNumber() != 0) {
        fieldRead->emitOpError() << "is an unsupported " << fieldRead->getName()
                                 << " operation: only reading fields from self is allowed";
        return nullptr;
      }

      return std::make_unique<VarExpr>(fieldRead.getFieldName());
    })
        .Case<NegFeltOp>([this](NegFeltOp negOp) {
      return std::make_unique<NegExpr>(translateValueToExpr(negOp.getOperand()));
    }).Default([](Operation *op) {
      op->emitOpError() << "is an unsupported operation in the Picus backend";
      return nullptr;
    });
  }

  Expression::ptr translateValueToExpr(Value val) {
    return llvm::TypeSwitch<Value, Expression::ptr>(val)
        .Case<BlockArgument>([](BlockArgument arg) -> Expression::ptr {
      // XXX: Only scalar felt values are currently supported for direct reading.
      // Other types should have been intercepted previously in the current iteration of this
      // module.
      if (!mlir::isa<FeltType>(arg.getType())) {
        return nullptr;
      }
      return std::make_unique<VarExpr>(inputVarName(arg.getContext(), arg.getArgNumber()));
    }).Case<OpResult>([this](OpResult res) {
      return translateOpToExpr(res.getDefiningOp());
    }).Default([](Value) {
      llvm_unreachable("TypedValue classes are not handled");
      return nullptr;
    });
  }

  LogicalResult translateStructInputsAndOutputs(StructDefOp structOp) {
    auto constrainFunc = structOp.getConstrainFuncOp();
    // Function arguments (except self) are considered inputs
    auto region = constrainFunc.getCallableRegion();
    if (!region) {
      return failure();
    }
    unsigned argCount = 1; // Start at 1 because we ignore self.
    for (auto arg : region->getArguments().drop_front()) {
      if (auto arrType = mlir::dyn_cast<ArrayType>(arg.getType())) {
        auto shape = arrType.getShape();
        if (shape.size() != 1 || shape[0] < 0) {
          return structOp->emitOpError()
                 << "is not supported. Array arguments must only have 1 known dimension";
        }
        for (int64_t idx = 0; idx < shape[0]; idx++) {
          auto name = inputVarName(structOp.getContext(), argCount, idx);
          outMod.addStatement(std::make_unique<InputStmt>(VarExpr(name)));
        }
      } else {
        auto name = inputVarName(structOp.getContext(), argCount);
        outMod.addStatement(std::make_unique<InputStmt>(VarExpr(name)));
      }
      argCount++;
    }

    // Fields are considered outputs
    structOp.walk([this](FieldDefOp fieldOp) {
      outMod.addStatement(std::make_unique<OutputStmt>(fieldOp.getSymName()));
    });

    return success();
  }

  LogicalResult translateStructToPicus(StructDefOp structOp) {
    auto constrainFunc = structOp.getConstrainFuncOp();
    // Traverse the use-def chain of each emit_eq and emit_in ops to generate
    // the constraints.
    constrainFunc.walk([this](EmitEqualityOp emitEqOp) {
      outMod.addStatement(std::make_unique<AssertStmt>(std::make_unique<EqExpr>(
          translateValueToExpr(emitEqOp.getLhs()), translateValueToExpr(emitEqOp.getRhs())
      )));
    });

    constrainFunc.walk([this](EmitContainmentOp emitInOp) {
      outMod.addStatement(std::make_unique<LookupStmt>(
          translateValueToExpr(emitInOp.getLhs()), translateValueToExpr(emitInOp.getRhs())
      ));
    });

    constrainFunc.walk([](CallOp callOp) {
      if (callOp.getCallee().getLeafReference().getValue() != "constrain") {
        return;
      }
      // TODO: Since calls in Picus kinda define variables, generating them from LLZK IR is not a
      // direct 1:1. I need to make some tweaks and its midnight and my brain is basically scrambled
      // eggs.
    });

    return success();
  }
};

/// This class does book-keeping when translation a whole module.
/// NOTE: Not used right now but might as this module evolves. Will remove before merging if the
/// class is not used after all.
struct ModuleTranslatorHelper {
  llvm::StringMap<StructTranslatorHelper> structs;
};

} // namespace

static LogicalResult
translateGlobalToPicusFixedValue(GlobalDefOp globalOp, FixedValues &fixedValues) {
  if (!globalOp.isConstant()) {
    globalOp->emitWarning() << "Ignoring non-constant global " << globalOp.getSymName();
    return success();
  }

  auto initialValue = globalOp.getInitialValue();
  assert(initialValue);
  return llvm::TypeSwitch<Type, LogicalResult>(globalOp.getType())
      .Case<FeltType>([&fixedValues, &initialValue, name = globalOp.getSymName()](auto) {
    auto constFelt = mlir::cast<FeltConstAttr>(initialValue);
    fixedValues.addFixedValues(name, constFelt.getValue());
    return success();
  })
      .Case<ArrayType>(
          [&fixedValues, &globalOp, &initialValue,
           name = globalOp.getSymName()](auto type) -> LogicalResult {
    if (!mlir::isa<FeltType>(type.getElementType())) {
      return globalOp->emitOpError()
             << "Unsupported type for constant global: " << type.getElementType();
    }

    auto arrayAttr = mlir::cast<ArrayAttr>(initialValue);
    fixedValues.addFixedValues(name, llvm::map_to_vector(arrayAttr, [](Attribute attr) {
      return ConstExpr(mlir::cast<FeltConstAttr>(attr).getValue());
    }));
    return success();
  }
      )
      .Default([&globalOp](auto type) -> LogicalResult {
    return globalOp->emitOpError() << "Unsupported type for constant global: " << type;
  });
}

template <typename Op, typename Fn> LogicalResult translateOps(ModuleOp op, Fn fn) {
  auto result = op->walk([&fn](Op inner) {
    if (failed(fn(inner))) {
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return failure(result == WalkResult::interrupt());
}

std::unique_ptr<Circuit> llzk::translateModuleToPicus(Operation *op, llvm::APInt primeNumber) {
  assert(
      validInputOperation(op) &&
      "Operation has to be either llzk.struct or module with llzk attribute"
  );

  auto modOp = mlir::cast<ModuleOp>(op);
  auto circuit = std::make_unique<Circuit>(primeNumber);

  auto &fixedValues = circuit->getFixedValues();
  if (failed(translateOps<GlobalDefOp>(modOp, [&fixedValues](GlobalDefOp globalOp) {
    return translateGlobalToPicusFixedValue(globalOp, fixedValues);
  }))) {
    return nullptr;
  }

  if (failed(translateOps<StructDefOp>(modOp, [&circuit, &fixedValues](StructDefOp structOp) {
    auto &mod = circuit->emplaceModule(structOp.getSymName());
    StructTranslatorHelper helper {.fixedValues = fixedValues, .outMod = mod};
    return helper.translateStructInputsAndOutputs(structOp);
  }))) {
    return nullptr;
  }

  if (failed(translateOps<StructDefOp>(modOp, [&circuit, &fixedValues](StructDefOp structOp) {
    auto &mod = circuit->emplaceModule(structOp.getSymName());
    StructTranslatorHelper helper {.fixedValues = fixedValues, .outMod = mod};
    return helper.translateStructToPicus(structOp);
  }))) {
    return nullptr;
  }

  return circuit;
}

// void testFixed() {
//   picus::FixedValues fv;
//   fv.addFixedValues("x", picus::ConstExpr(/* dummy */ llvm::APInt(32, 1)));
// }

