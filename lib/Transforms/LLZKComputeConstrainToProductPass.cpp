#include "llzk/Analysis/LightweightSignalEquivalenceAnalysis.h"
#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Dialect/Struct/IR/Ops.h"
#include "llzk/Transforms/LLZKTransformationPasses.h"
#include "llzk/Util/Constants.h"

#include <iterator>
#include <memory>
#include <ranges>

#include "llvm/Support/Debug.h"
#include "mlir/IR/Builders.h"
#include "mlir/Transforms/InliningUtils.h"
namespace llzk {
#define GEN_PASS_DECL_COMPUTECONSTRAINTOPRODUCTPASS
#define GEN_PASS_DEF_COMPUTECONSTRAINTOPRODUCTPASS
#include "llzk/Transforms/LLZKTransformationPasses.h.inc"
} // namespace llzk

#define DEBUG_TYPE "llzk-compute-constrain-to-product-pass"

using namespace llzk::component;
using namespace llzk::function;
using namespace mlir;

namespace llzk {
using std::make_unique;

bool isValidRoot(StructDefOp root) {
  FuncDefOp computeFunc = root.getComputeFuncOp();
  FuncDefOp constrainFunc = root.getConstrainFuncOp();

  if (!computeFunc || !constrainFunc) {
    root->emitError() << "no " << FUNC_NAME_COMPUTE << "/" << FUNC_NAME_CONSTRAIN << " to align";
    return false;
  }

  // TODO: If root::@compute and root::@constrain are called anywhere else, this is not a valid root
  // TODO: to start aligning from

  return true;
}

class ComputeConstrainToProductPass
    : public llzk::impl::ComputeConstrainToProductPassBase<ComputeConstrainToProductPass> {

  std::vector<StructDefOp> alignedStructs;
  // Given a @product function body, try to match up calls to @A::@compute and @A::@constrain for
  // every sub-struct @A and replace them with a call to @A::@product
  LogicalResult alignCalls(
      FuncDefOp product, SymbolTableCollection &tables,
      LightweightSignalEquivalenceAnalysis &equivalence
  );

  // Given a StructDefOp @root, replace the @root::@compute and @root::@constrain functions with a
  // @root::@product
  FuncDefOp alignFuncs(
      StructDefOp root, FuncDefOp compute, FuncDefOp constrain, SymbolTableCollection &tables,
      LightweightSignalEquivalenceAnalysis &equivalence
  );

public:
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    StructDefOp root;

    SymbolTableCollection tables;
    LightweightSignalEquivalenceAnalysis equivalence {
        getAnalysis<LightweightSignalEquivalenceAnalysis>()
    };

    // Find the indicated root struct and make sure its a valid place to start aligning
    mod.walk([&root, this](StructDefOp structDef) {
      if (structDef.getSymName() == rootStruct) {
        root = structDef;
      }
    });
    if (!isValidRoot(root)) {
      signalPassFailure();
      return;
    }

    // Try aligning the root functions
    if (!alignFuncs(
            root, root.getComputeFuncOp(), root.getConstrainFuncOp(), tables, equivalence
        )) {
      signalPassFailure();
    }

    for (auto s : alignedStructs) {
      s.getComputeFuncOp()->erase();
      s.getConstrainFuncOp()->erase();
    }
  }
};

FuncDefOp ComputeConstrainToProductPass::alignFuncs(
    StructDefOp root, FuncDefOp compute, FuncDefOp constrain, SymbolTableCollection &tables,
    LightweightSignalEquivalenceAnalysis &equivalence
) {
  OpBuilder funcBuilder(compute);

  // Create an empty @product func...
  FuncDefOp productFunc = funcBuilder.create<FuncDefOp>(
      funcBuilder.getUnknownLoc(), FUNC_NAME_PRODUCT, compute.getFunctionType()
  );
  Block *entryBlock = productFunc.addEntryBlock();
  OpBuilder bodyBuilder(entryBlock, entryBlock->begin());

  // ...with the right arguments
  std::vector<Value> args;
  std::copy(
      productFunc.getArguments().begin(), productFunc.getArguments().end(), std::back_inserter(args)
  );

  // Add calls to @compute and @constrain...
  CallOp computeCall = bodyBuilder.create<CallOp>(bodyBuilder.getUnknownLoc(), compute, args);
  args.insert(args.begin(), computeCall->getResult(0));
  CallOp constrainCall = bodyBuilder.create<CallOp>(bodyBuilder.getUnknownLoc(), constrain, args);
  bodyBuilder.create<ReturnOp>(bodyBuilder.getUnknownLoc(), computeCall->getResult(0));

  // ..and inline them
  InlinerInterface inliner(productFunc.getContext());
  if (failed(inlineCall(inliner, computeCall, compute, &compute.getBody(), true))) {
    root->emitError() << "failed to inline " << FUNC_NAME_COMPUTE;
    return nullptr;
  }
  if (failed(inlineCall(inliner, constrainCall, constrain, &constrain.getBody(), true))) {
    root->emitError() << "failed to inline " << FUNC_NAME_CONSTRAIN;
    return nullptr;
  }
  computeCall->erase();
  constrainCall->erase();

  // Mark the compute/constrain for deletion
  alignedStructs.push_back(root);

  // Make sure we can align sub-calls to @compute and @constrain
  if (failed(alignCalls(productFunc, tables, equivalence))) {
    return nullptr;
  }
  return productFunc;
}

LogicalResult ComputeConstrainToProductPass::alignCalls(
    FuncDefOp product, SymbolTableCollection &tables,
    LightweightSignalEquivalenceAnalysis &equivalence
) {
  // Gather up all the remaining calls to @compute and @constrain
  llvm::SetVector<CallOp> computeCalls, constrainCalls;
  product.walk([&](CallOp callOp) {
    if (callOp.getCallee().getLeafReference() == FUNC_NAME_COMPUTE) {
      computeCalls.insert(callOp);
    } else if (callOp.getCallee().getLeafReference() == FUNC_NAME_CONSTRAIN) {
      constrainCalls.insert(callOp);
    }
  });

  llvm::SetVector<std::pair<CallOp, CallOp>> alignedCalls;

  // A @compute matches a @constrain if they belong to the same struct and all their input signals
  // are pairwise equivalent
  auto doCallsMatch = [&](CallOp compute, CallOp constrain) -> bool {
    LLVM_DEBUG({
      llvm::outs() << "Asking for equivalence between calls\n"
                   << compute << "\nand\n"
                   << constrain << "\n\n";
      llvm::outs() << "In block:\n\n" << *compute->getBlock() << "\n";
    });

    auto computeStruct = compute.getCallee().getNestedReferences().drop_back(1);
    auto constrainStruct = constrain.getCallee().getNestedReferences().drop_back(1);
    if (computeStruct != constrainStruct) {
      return false;
    }

    for (unsigned i = 0; i < compute->getNumOperands(); i++) {
      if (!equivalence.areSignalsEquivalent(compute->getOperand(i), constrain->getOperand(i + 1))) {
        return false;
      }
    }

    return true;
  };

  for (auto compute : computeCalls) {
    // If there is exactly one @compute that matches a given @constrain, we can align them
    auto matches = llvm::filter_to_vector(constrainCalls, [&](CallOp constrain) {
      return doCallsMatch(compute, constrain);
    });

    if (matches.size() == 1) {
      alignedCalls.insert({compute, matches[0]});
      computeCalls.remove(compute);
      constrainCalls.remove(matches[0]);
    }
  }

  // TODO: If unaligned calls remain, fully inline their structs and continue instead of failing
  if (!computeCalls.empty() && constrainCalls.empty()) {
    product->emitError() << "failed to align some @" << FUNC_NAME_COMPUTE << " and @"
                         << FUNC_NAME_CONSTRAIN;
    return failure();
  }

  for (auto [compute, constrain] : alignedCalls) {
    // If @A::@compute matches @A::@constrain, recursively align the functions in @A...
    auto newRoot = compute.getCalleeTarget(tables)->get()->getParentOfType<StructDefOp>();
    assert(newRoot);
    FuncDefOp newProduct = alignFuncs(
        newRoot, newRoot.getComputeFuncOp(), newRoot.getConstrainFuncOp(), tables, equivalence
    );
    if (!newProduct) {
      return failure();
    }

    // ...and replace the two calls with a single call to @A::@product
    OpBuilder callBuilder(compute);
    CallOp newCall =
        callBuilder.create<CallOp>(callBuilder.getUnknownLoc(), newProduct, compute.getOperands());
    compute->replaceAllUsesWith(newCall.getResults());
    compute->erase();
    constrain->erase();
  }

  return success();
}

std::unique_ptr<mlir::Pass> createComputeConstrainToProductPass() {
  return make_unique<ComputeConstrainToProductPass>();
}

} // namespace llzk
