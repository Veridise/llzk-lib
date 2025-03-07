#include "llzk/Dialect/LLZK/IR/Ops.h"
#include "llzk/Dialect/LLZK/Transforms/LLZKTransformationPasses.h"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/DenseMap.h>

/// Include the generated base pass class definitions.
namespace llzk {
#define GEN_PASS_DEF_DUPLICATESTRUCTELIMINATIONPASS
#include "llzk/Dialect/LLZK/Transforms/LLZKTransformationPasses.h.inc"
} // namespace llzk

using namespace mlir;
using namespace llzk;

namespace {

class DuplicateStructEliminationPass : public llzk::impl::DuplicateStructEliminationPassBase<DuplicateStructEliminationPass> {
  void runOnOperation() override {
    mlir::ModuleOp mod = getOperation();

    mod.walk([](StructDefOp structDef) {
      llvm::errs() << structDef << "\n";
      auto computeFn = structDef.getComputeFuncOp();

      // Find all compute calls and categorize them by component type

      SmallVector<CallOp, 1> calls;
      computeFn.walk([&](CallOp call) {
        calls.push_back(call);
      });

      // replace
      // Based on the previous passes, we can assume
    });
  }
};

} // namespace

std::unique_ptr<mlir::Pass> llzk::createDuplicateStructEliminationPass() {
  return std::make_unique<DuplicateStructEliminationPass>();
};
