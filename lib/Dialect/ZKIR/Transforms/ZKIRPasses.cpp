#include "zkir/Dialect/ZKIR/Transforms/ZKIRPasses.h"
#include "zkir/Dialect/ZKIR/IR/Ops.h"
#include "zkir/Dialect/ZKIR/Util/IncludeHelper.h"
#include <zkir/Dialect/ZKIR/Util/SymbolHelper.h>

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>

/// Include the generated base pass class definitions.
namespace zkir {
#define GEN_PASS_DEF_INLINEINCLUDESPASS
#include "zkir/Dialect/ZKIR/Transforms/ZKIRPasses.h.inc"
} // namespace zkir

namespace {

class InlineIncludesPass : public zkir::impl::InlineIncludesPassBase<InlineIncludesPass> {
  void runOnOperation() override {
    mlir::ModuleOp topMod = getOperation();
    if (topMod->hasAttr(zkir::LANG_ATTR_NAME)) {
      mlir::MLIRContext *ctx = &getContext();

      std::vector<mlir::ModuleOp> currLevel = {topMod};
      do {
        std::vector<mlir::ModuleOp> nextLevel = {};
        for (mlir::ModuleOp currentMod : currLevel) {
          currentMod.walk([ctx, &nextLevel](zkir::IncludeOp mod) {
            mlir::FailureOr<mlir::ModuleOp> result = zkir::inlineTheInclude(ctx, mod);
            if (mlir::failed(result)) {
              return mlir::WalkResult::interrupt();
            }
            nextLevel.push_back(result.value());
            return mlir::WalkResult::advance();
          });
        }
        currLevel = nextLevel;
      } while (!currLevel.empty());
    }
    markAllAnalysesPreserved();
  }
};

} // namespace

std::unique_ptr<mlir::Pass> zkir::createInlineIncludesPass() {
  return std::make_unique<InlineIncludesPass>();
};
