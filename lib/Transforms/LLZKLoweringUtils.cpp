//===-- LLZKLoweringUtils.cpp --------------------------------*- C++ -*----===//
//
// Shared utility function implementations for LLZK lowering passes.
//
//===----------------------------------------------------------------------===//

#include "llzk/Transforms/LLZKLoweringUtils.h"

#include <mlir/IR/Block.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Operation.h>
#include <mlir/Support/LogicalResult.h>

#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/raw_ostream.h>

using namespace mlir;
using namespace llzk;
using namespace llzk::felt;
using namespace llzk::function;
using namespace llzk::component;
using namespace llzk::constrain;

namespace llzk {

Value getSelfValueFromCompute(FuncDefOp computeFunc) {
  Region &body = computeFunc.getBody();
  assert(!body.empty() && "compute() function body is empty");
  Block &block = body.front();
  Operation *terminator = block.getTerminator();
  assert(terminator && "compute() function has no terminator");
  auto retOp = dyn_cast<ReturnOp>(terminator);
  if (!retOp) {
    llvm::errs() << "Expected ReturnOp but found: " << terminator->getName() << "\n";
    llvm_unreachable("compute() function must end with ReturnOp");
  }
  return retOp.getOperands().front();
}

Value rebuildExprInCompute(
    Value val, FuncDefOp computeFunc, OpBuilder &builder, DenseMap<Value, Value> &memo
) {
  if (auto it = memo.find(val); it != memo.end()) {
    return it->second;
  }

  if (auto barg = val.dyn_cast<BlockArgument>()) {
    unsigned index = barg.getArgNumber();
    Value mapped = computeFunc.getArgument(index - 1);
    return memo[val] = mapped;
  }

  if (auto readOp = val.getDefiningOp<FieldReadOp>()) {
    Value self = getSelfValueFromCompute(computeFunc);
    Value rebuilt = builder.create<FieldReadOp>(
        readOp.getLoc(), readOp.getType(), self, readOp.getFieldNameAttr().getAttr()
    );
    return memo[val] = rebuilt;
  }

  if (auto add = val.getDefiningOp<AddFeltOp>()) {
    Value lhs = rebuildExprInCompute(add.getLhs(), computeFunc, builder, memo);
    Value rhs = rebuildExprInCompute(add.getRhs(), computeFunc, builder, memo);
    return memo[val] = builder.create<AddFeltOp>(add.getLoc(), add.getType(), lhs, rhs);
  }

  if (auto sub = val.getDefiningOp<SubFeltOp>()) {
    Value lhs = rebuildExprInCompute(sub.getLhs(), computeFunc, builder, memo);
    Value rhs = rebuildExprInCompute(sub.getRhs(), computeFunc, builder, memo);
    return memo[val] = builder.create<SubFeltOp>(sub.getLoc(), sub.getType(), lhs, rhs);
  }

  if (auto mul = val.getDefiningOp<MulFeltOp>()) {
    Value lhs = rebuildExprInCompute(mul.getLhs(), computeFunc, builder, memo);
    Value rhs = rebuildExprInCompute(mul.getRhs(), computeFunc, builder, memo);
    return memo[val] = builder.create<MulFeltOp>(mul.getLoc(), mul.getType(), lhs, rhs);
  }

  if (auto neg = val.getDefiningOp<NegFeltOp>()) {
    Value operand = rebuildExprInCompute(neg.getOperand(), computeFunc, builder, memo);
    return memo[val] = builder.create<NegFeltOp>(neg.getLoc(), neg.getType(), operand);
  }

  if (auto div = val.getDefiningOp<DivFeltOp>()) {
    Value lhs = rebuildExprInCompute(div.getLhs(), computeFunc, builder, memo);
    Value rhs = rebuildExprInCompute(div.getRhs(), computeFunc, builder, memo);
    return memo[val] = builder.create<DivFeltOp>(div.getLoc(), div.getType(), lhs, rhs);
  }

  if (auto c = val.getDefiningOp<FeltConstantOp>()) {
    return memo[val] = builder.create<FeltConstantOp>(c.getLoc(), c.getValue());
  }

  llvm::errs() << "Unhandled op in rebuildExprInCompute: " << val << "\n";
  llvm_unreachable("Unsupported op kind");
}

LogicalResult checkForAuxFieldConflicts(StructDefOp structDef, StringRef prefix) {
  bool conflictFound = false;

  structDef.walk([&](FieldDefOp fieldDefOp) {
    if (fieldDefOp.getName().starts_with(prefix)) {
      fieldDefOp.emitError() << "Field name '" << fieldDefOp.getName()
                             << "' conflicts with reserved prefix '" << prefix << "'";
      conflictFound = true;
    }
  });

  return conflictFound ? failure() : success();
}

/// Replaces all *subsequent uses* of `oldVal` with `newVal`, starting *after* `afterOp`.
///
/// Specifically:
/// - Uses of `oldVal` in operations that come **after** `afterOp` in the same block are replaced.
/// - Uses in `afterOp` itself are **not replaced** (to avoid self-trivializing rewrites).
/// - Uses in other blocks are replaced (if applicable).
///
/// Typical use case:
/// - You introduce an auxiliary value (e.g., via EmitEqualityOp) and want to replace
///   all *later* uses of the original value while preserving the constraint itself.
///
/// \param oldVal  The original value whose uses should be redirected.
/// \param newVal  The new value to replace subsequent uses with.
/// \param afterOp The operation after which uses of `oldVal` will be replaced.
void replaceSubsequentUsesWith(Value oldVal, Value newVal, Operation *afterOp) {
  assert(afterOp && "afterOp must be a valid Operation*");

  for (auto &use : llvm::make_early_inc_range(oldVal.getUses())) {
    Operation *user = use.getOwner();

    // Skip uses that are:
    // - Before afterOp in the same block.
    // - Inside afterOp itself.
    if ((user->getBlock() == afterOp->getBlock()) &&
        (user->isBeforeInBlock(afterOp) || user == afterOp)) {
      continue;
    }

    // Replace this use of oldVal with newVal.
    use.set(newVal);
  }
}

void addAuxField(StructDefOp structDef, StringRef name) {
  OpBuilder builder(structDef);
  builder.setInsertionPointToEnd(&structDef.getBody().front());
  builder.create<FieldDefOp>(
      structDef.getLoc(), builder.getStringAttr(name), builder.getType<FeltType>()
  );
}

unsigned getFeltDegree(Value val, DenseMap<Value, unsigned> &memo) {
  if (auto it = memo.find(val); it != memo.end()) {
    return it->second;
  }

  if (val.isa<BlockArgument>()) {
    return memo[val] = 1;
  }
  if (isa<FeltConstantOp>(val.getDefiningOp())) {
    return memo[val] = 0;
  }
  if (isa<FeltNonDetOp>(val.getDefiningOp())) {
    return memo[val] = 1;
  }
  if (isa<FieldReadOp>(val.getDefiningOp())) {
    return memo[val] = 1;
  }

  if (auto add = val.getDefiningOp<AddFeltOp>()) {
    return memo[val] =
               std::max(getFeltDegree(add.getLhs(), memo), getFeltDegree(add.getRhs(), memo));
  }
  if (auto sub = val.getDefiningOp<SubFeltOp>()) {
    return memo[val] =
               std::max(getFeltDegree(sub.getLhs(), memo), getFeltDegree(sub.getRhs(), memo));
  }
  if (auto mul = val.getDefiningOp<MulFeltOp>()) {
    return memo[val] = getFeltDegree(mul.getLhs(), memo) + getFeltDegree(mul.getRhs(), memo);
  }
  if (auto div = val.getDefiningOp<DivFeltOp>()) {
    return memo[val] = getFeltDegree(div.getLhs(), memo) + getFeltDegree(div.getRhs(), memo);
  }
  if (auto neg = val.getDefiningOp<NegFeltOp>()) {
    return memo[val] = getFeltDegree(neg.getOperand(), memo);
  }

  llvm::errs() << "Unhandled felt op in degree computation: " << val << "\n";
  llvm_unreachable("Unhandled op in getFeltDegree");
}

} // namespace llzk
