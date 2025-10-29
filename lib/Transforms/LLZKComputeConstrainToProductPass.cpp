#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Dialect/Struct/IR/Ops.h"
#include "llzk/Transforms/LLZKTransformationPasses.h"
#include "llzk/Util/Constants.h"

#include <iterator>
#include <memory>

#include "mlir/IR/Builders.h"
#include "mlir/Transforms/InliningUtils.h"
namespace llzk {
#define GEN_PASS_DECL_COMPUTECONSTRAINTOPRODUCTPASS
#define GEN_PASS_DEF_COMPUTECONSTRAINTOPRODUCTPASS
#include "llzk/Transforms/LLZKTransformationPasses.h.inc"
} // namespace llzk

#define DEBUG_TYPE "llzk-compute-constrain-to-product-pass"

namespace llzk {
using std::make_unique;

using namespace llzk::component;
using namespace llzk::function;
using namespace mlir;

class Inliner : public InlinerInterface {
  using InlinerInterface::InlinerInterface;
  bool isLegalToInline(Operation *, Operation *, bool) const override { return true; }
  bool isLegalToInline(Region *, Region *, bool, IRMapping &) const override { return true; }
};

FuncDefOp alignStruct(StructDefOp structDef, FuncDefOp computeFunc, FuncDefOp constrainFunc) {
  OpBuilder funcBuilder(computeFunc);
  FuncDefOp productFunc = funcBuilder.create<FuncDefOp>(
      funcBuilder.getUnknownLoc(), FUNC_NAME_PRODUCT, computeFunc.getFunctionType()
  );

  Block *entryBlock = productFunc.addEntryBlock();
  OpBuilder bodyBuilder(entryBlock, entryBlock->begin());

  std::vector<Value> args;
  std::copy(
      productFunc.getArguments().begin(), productFunc.getArguments().end(), std::back_inserter(args)
  );

  CallOp computeCall = bodyBuilder.create<CallOp>(bodyBuilder.getUnknownLoc(), computeFunc, args);
  args.insert(args.begin(), computeCall->getResult(0));
  CallOp constrainCall =
      bodyBuilder.create<CallOp>(bodyBuilder.getUnknownLoc(), constrainFunc, args);
  bodyBuilder.create<ReturnOp>(bodyBuilder.getUnknownLoc(), computeCall->getResult(0));

  InlinerInterface inliner(productFunc.getContext());
  if (failed(inlineCall(inliner, computeCall, computeFunc, &computeFunc.getBody(), false))) {
    structDef->emitError() << "failed to inline " << FUNC_NAME_COMPUTE;
    return nullptr;
  }
  if (failed(inlineCall(inliner, constrainCall, constrainFunc, &constrainFunc.getBody(), false))) {
    structDef->emitError() << "failed to inline " << FUNC_NAME_CONSTRAIN;
    return nullptr;
  }

  computeCall->erase();
  constrainCall->erase();

  computeFunc.erase();
  constrainFunc.erase();
  return productFunc;
}

bool isValidRoot(StructDefOp structDef) {
  FuncDefOp computeFunc = structDef.getComputeFuncOp();
  FuncDefOp constrainFunc = structDef.getConstrainFuncOp();

  if (!computeFunc || !constrainFunc) {
    structDef->emitError() << "no " << FUNC_NAME_COMPUTE << "/" << FUNC_NAME_CONSTRAIN
                           << " to align";
    return false;
  }

  // TODO: Check to see if root::@compute and root::@constrain are called anywhere else

  return true;
}

class ComputeConstrainToProductPass
    : public llzk::impl::ComputeConstrainToProductPassBase<ComputeConstrainToProductPass> {

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    StructDefOp root;

    mod.walk([&root, this](StructDefOp structDef) {
      if (structDef.getSymName() == rootStruct) {
        root = structDef;
      }
    });

    if (!isValidRoot(root)) {
      signalPassFailure();
      return;
    }

    FuncDefOp product = alignStruct(root, root.getComputeFuncOp(), root.getConstrainFuncOp());
    if (!product) {
    }
  }
};

std::unique_ptr<mlir::Pass> createComputeConstrainToProductPass() {
  return make_unique<ComputeConstrainToProductPass>();
}

} // namespace llzk
