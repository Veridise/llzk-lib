#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Dialect/Struct/IR/Ops.h"
#include "llzk/Transforms/LLZKTransformationPasses.h"
#include "llzk/Util/Constants.h"

#include <iterator>
#include <memory>

#include "llvm/Support/Debug.h"
#include "mlir/IR/Builders.h"
#include "mlir/Transforms/InliningUtils.h"
namespace llzk {
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

LogicalResult transformStruct(StructDefOp structDef) {
  FuncDefOp computeFunc = structDef.getComputeFuncOp();
  FuncDefOp constrainFunc = structDef.getConstrainFuncOp();
  OpBuilder funcBuilder(computeFunc);
  FuncDefOp productFunc = funcBuilder.create<FuncDefOp>(
      funcBuilder.getUnknownLoc(), FUNC_NAME_PRODUCT, computeFunc.getFunctionType()
  );

  productFunc.setPrivate();
  Block *entryBlock = productFunc.addEntryBlock();
  OpBuilder bodyBuilder(entryBlock, entryBlock->begin());

  // std::vector<Value> args = productFunc.getArguments().vec();
  std::vector<Value> args;
  std::copy(
      productFunc.getArguments().begin(), productFunc.getArguments().end(), std::back_inserter(args)
  );
  // ValueRange args = productFunc.getArguments();
  CallOp computeCall = bodyBuilder.create<CallOp>(bodyBuilder.getUnknownLoc(), computeFunc, args);
  args.insert(args.begin(), computeCall->getResult(0));
  CallOp constrainCall =
      bodyBuilder.create<CallOp>(bodyBuilder.getUnknownLoc(), constrainFunc, args);
  bodyBuilder.create<ReturnOp>(bodyBuilder.getUnknownLoc(), computeCall->getResult(0));

  InlinerInterface inliner(productFunc.getContext());
  if (failed(inlineCall(inliner, computeCall, computeFunc, &computeFunc.getBody(), false))) {
    return failure();
  }
  if (failed(inlineCall(inliner, constrainCall, constrainFunc, &constrainFunc.getBody(), false))) {
    return failure();
  }

  computeCall->erase();
  constrainCall->erase();

  // Block *productBody = productFunc.addEntryBlock();
  // builder.setInsertionPointToEnd(productBody);

  // builder.create<ReturnOp>(builder.getInsertionPoint()->getLoc());

  computeFunc.erase();
  constrainFunc.erase();
  return success();
}

class ComputeConstrainToProductPass
    : public llzk::impl::ComputeConstrainToProductPassBase<ComputeConstrainToProductPass> {

  void runOnOperation() override {
    getOperation().walk([](StructDefOp structDef) { (void)transformStruct(structDef); });
  }
};

std::unique_ptr<mlir::Pass> createComputeConstrainToProductPass() {
  return make_unique<ComputeConstrainToProductPass>();
}

} // namespace llzk
