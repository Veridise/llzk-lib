#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Dialect/Struct/IR/Ops.h"
#include "llzk/Transforms/LLZKTransformationPasses.h"
#include "llzk/Util/Constants.h"

#include <memory>

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

void transformStruct(StructDefOp structDef) {
  OpBuilder builder(structDef.getRegion());
  FuncDefOp computeFunc = structDef.getComputeFuncOp();
  FuncDefOp productFunc = builder.create<FuncDefOp>(
      structDef.getRegion().getLoc(), FUNC_NAME_PRODUCT, computeFunc.getFunctionType()
  );
}

class ComputeConstrainToProductPass
    : public llzk::impl::ComputeConstrainToProductPassBase<ComputeConstrainToProductPass> {

  void runOnOperation() override {}
};

std::unique_ptr<mlir::Pass> createComputeConstrainToProductPass() {
  return make_unique<ComputeConstrainToProductPass>();
}

} // namespace llzk
