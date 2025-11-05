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

namespace llzk {
using std::make_unique;

using namespace llzk::component;
using namespace llzk::function;
using namespace mlir;

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

  std::vector<StructDefOp> alignedStructs;

  LogicalResult alignCalls(
      FuncDefOp product, SymbolTableCollection &tables,
      LightweightSignalEquivalenceAnalysis &equivalence
  );
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

    mod.walk([&root, this](StructDefOp structDef) {
      if (structDef.getSymName() == rootStruct) {
        root = structDef;
      }
    });

    if (!isValidRoot(root)) {
      signalPassFailure();
      return;
    }

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

  alignedStructs.push_back(root); // Mark the compute/constrain for deletion

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

  auto doCallsAlign = [&](CallOp compute, CallOp constrain) -> bool {
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
    auto matches = llvm::filter_to_vector(constrainCalls, [&](CallOp constrain) {
      return doCallsAlign(compute, constrain);
    });

    if (matches.size() == 1) {
      alignedCalls.insert({compute, matches[0]});
      computeCalls.remove(compute);
      constrainCalls.remove(matches[0]);
    }
  }

  for (auto [compute, constrain] : alignedCalls) {
    auto newRoot = compute.getCalleeTarget(tables)->get()->getParentOfType<StructDefOp>();
    assert(newRoot);
    FuncDefOp newProduct = alignFuncs(
        newRoot, newRoot.getComputeFuncOp(), newRoot.getConstrainFuncOp(), tables, equivalence
    );
    if (!newProduct) {
      return failure();
    }

    OpBuilder callBuilder(compute);
    CallOp newCall =
        callBuilder.create<CallOp>(callBuilder.getUnknownLoc(), newProduct, compute.getOperands());
    compute->replaceAllUsesWith(newCall.getResults());
    compute->erase();
    constrain->erase();
  }

  for (auto call : llvm::concat<const CallOp>(computeCalls, constrainCalls)) {
    InlinerInterface inliner(product.getContext());
    auto funcDef = call.getCalleeTarget(tables)->get();
    if (failed(inlineCall(inliner, call, funcDef, &funcDef.getBody(), true))) {
      call->emitError() << "failed to inline";
      return failure();
    }
    call->erase();
  }

  product.walk<WalkOrder::PostOrder>([](Operation *op) {
    if (op->getNumResults() > 0 && op->getUses().empty()) {
      op->erase();
      return WalkResult::skip();
    }
    return WalkResult::advance();
  });

  return success();
}

std::unique_ptr<mlir::Pass> createComputeConstrainToProductPass() {
  return make_unique<ComputeConstrainToProductPass>();
}

} // namespace llzk
