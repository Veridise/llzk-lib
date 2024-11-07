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
        // TODO: may want to keep track of the specific include at each level that got
        //  us to the location where an error occurs so that the include trace can be
        //  backtracked easily enough. Basically I think each entry in the "levels"
        //  must also contain an IncludeOp backtrace stack in addition to the ModuleOp.
        std::vector<mlir::ModuleOp> nextLevel = {};
        for (mlir::ModuleOp currentMod : currLevel) {
          currentMod.walk([ctx, &nextLevel](zkir::IncludeOp mod) {
            mlir::FailureOr<mlir::ModuleOp> result = zkir::inlineTheInclude(ctx, mod);
            if (mlir::succeeded(result)) {
              nextLevel.push_back(result.value());
            }
            // Advance in either case so as many errors as possible are found in a single run.
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
