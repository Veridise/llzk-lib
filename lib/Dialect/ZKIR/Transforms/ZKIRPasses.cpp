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
using namespace std;
using namespace mlir;

using IncludeStack = vector<pair<StringRef, Location>>;

inline bool contains(IncludeStack &stack, StringRef &&loc) {
  auto path_match = [loc](pair<StringRef, Location> &p) { return p.first == loc; };
  return std::find_if(stack.begin(), stack.end(), path_match) != stack.end();
}

class InlineIncludesPass : public zkir::impl::InlineIncludesPassBase<InlineIncludesPass> {
  void runOnOperation() override {
    ModuleOp topMod = getOperation();
    if (topMod->hasAttr(zkir::LANG_ATTR_NAME)) {
      vector<pair<ModuleOp, IncludeStack>> currLevel = {make_pair(topMod, IncludeStack())};
      do {
        vector<pair<ModuleOp, IncludeStack>> nextLevel = {};
        for (pair<ModuleOp, IncludeStack> &curr : currLevel) {
          curr.first.walk([includeStack = std::move(curr.second),
                           &nextLevel](zkir::IncludeOp incOp) mutable {
            // Check for cyclic includes
            if (contains(includeStack, incOp.getPath())) {
              auto err = incOp.emitError().append("found cyclic include");
              for (auto it = includeStack.rbegin(); it != includeStack.rend(); ++it) {
                err.attachNote(it->second).append("included from here");
              }
            } else {
              includeStack.push_back(make_pair(incOp.getPath(), incOp.getLoc()));
              FailureOr<ModuleOp> result = incOp.inlineAndErase();
              if (succeeded(result)) {
                ModuleOp newMod = std::move(result.value());
                nextLevel.push_back(make_pair(newMod, includeStack));
              }
            }
            // Advance in either case so as many errors as possible are found in a single run.
            return WalkResult::advance();
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
