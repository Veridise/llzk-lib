#include "llzk/Dialect/LLZK/IR/Ops.h"
#include "llzk/Dialect/LLZK/Transforms/LLZKTransformationPasses.h"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallVector.h>

/// Include the generated base pass class definitions.
namespace llzk {
#define GEN_PASS_DEF_REDUNDANTFUNCTIONCALLELIMINATIONPASS
#include "llzk/Dialect/LLZK/Transforms/LLZKTransformationPasses.h.inc"
} // namespace llzk

using namespace mlir;
using namespace llzk;

namespace {

class RedundantFunctionCallEliminationPass
    : public llzk::impl::RedundantFunctionCallEliminationPassBase<
          RedundantFunctionCallEliminationPass> {
  void runOnOperation() override { llvm::errs() << "RedundantFunctionCallEliminationPass\n"; }
};

} // namespace

std::unique_ptr<mlir::Pass> llzk::createRedundantFunctionCallEliminationPass() {
  return std::make_unique<RedundantFunctionCallEliminationPass>();
};
