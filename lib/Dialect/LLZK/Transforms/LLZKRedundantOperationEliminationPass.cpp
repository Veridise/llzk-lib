#include "llzk/Dialect/LLZK/IR/Ops.h"
#include "llzk/Dialect/LLZK/Transforms/LLZKTransformationPasses.h"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallVector.h>

/// Include the generated base pass class definitions.
namespace llzk {
#define GEN_PASS_DEF_REDUNDANTOPERATIONELIMINATIONPASS
#include "llzk/Dialect/LLZK/Transforms/LLZKTransformationPasses.h.inc"
} // namespace llzk

using namespace mlir;
using namespace llzk;

namespace {

static auto EMPTY_OP_KEY = reinterpret_cast<Operation *>(1);
static auto TOMBSTONE_OP_KEY = reinterpret_cast<Operation *>(2);

// Maps original -> replacement value
using TranslationMap = DenseMap<Value, Value>;

/// @brief A wrapper for an operation that provides comparators for operations
/// to determine if their outputs will be equal. In general, this will compare
/// to see if the translated operands for a given operation are equal.
class OperationComparator {
public:
  explicit OperationComparator(Operation *o) : op(o) {
    if (op != EMPTY_OP_KEY && op != TOMBSTONE_OP_KEY) {
      operands = SmallVector<Value>(op->getOperands());
    }
  }

  OperationComparator(Operation *o, const TranslationMap &m) : op(o) {
    for (auto operand : op->getOperands()) {
      if (auto it = m.find(operand); it != m.end()) {
        operands.push_back(it->second);
      } else {
        operands.push_back(operand);
      }
    }
  }

  Operation *getOp() const { return op; }

  const SmallVector<Value> &getOperands() const { return operands; }

  bool isCommutative() const { return op->hasTrait<OpTrait::IsCommutative>(); }

  friend bool operator==(const OperationComparator &lhs, const OperationComparator &rhs) {
    if (lhs.op->getName() != rhs.op->getName()) {
      return false;
    }
    // uninterested in operating over non llzk/arith ops
    auto dialectName = lhs.op->getDialect()->getNamespace();
    if (dialectName != "llzk" && dialectName != "arith") {
      return false;
    }

    // This may be overly restrictive in some cases, but without knowing what
    // potential future attributes we may have, it's safer to assume that
    // unequal attributes => unequal operations.
    // This covers constant operations too, as the constant is an attribute,
    // not an operand.
    if (lhs.op->getAttrs() != rhs.op->getAttrs()) {
      return false;
    }
    // For commutative operations, just check if the operands contain the same set in any order
    if (lhs.isCommutative()) {
      ensure(
          lhs.operands.size() == 2 && rhs.operands.size() == 2,
          "No known commutative ops have more than two arguments"
      );
      return (lhs.operands[0] == rhs.operands[0] && lhs.operands[1] == rhs.operands[1]) ||
             (lhs.operands[0] == rhs.operands[1] && lhs.operands[1] == rhs.operands[0]);
    }

    // The default case requires an exact match per argument
    return lhs.operands == rhs.operands;
  }

private:
  Operation *op;
  SmallVector<Value> operands;
};

} // namespace

namespace llvm {

template <> struct DenseMapInfo<OperationComparator> {
  static OperationComparator getEmptyKey() { return OperationComparator(EMPTY_OP_KEY); }
  static inline OperationComparator getTombstoneKey() {
    return OperationComparator(TOMBSTONE_OP_KEY);
  }
  static unsigned getHashValue(const OperationComparator &oc) {
    if (oc.getOp() == EMPTY_OP_KEY || oc.getOp() == TOMBSTONE_OP_KEY) {
      return hash_value(oc.getOp());
    }
    // Just hash on name to force more thorough equality checks by operation type.
    return hash_value(oc.getOp()->getName());
  }
  static bool isEqual(const OperationComparator &lhs, const OperationComparator &rhs) {
    if (lhs.getOp() == EMPTY_OP_KEY || rhs.getOp() == EMPTY_OP_KEY ||
        lhs.getOp() == TOMBSTONE_OP_KEY || rhs.getOp() == TOMBSTONE_OP_KEY) {
      return lhs.getOp() == rhs.getOp();
    }
    return lhs == rhs;
  }
};

} // namespace llvm

namespace {

class RedundantOperationEliminationPass
    : public llzk::impl::RedundantOperationEliminationPassBase<RedundantOperationEliminationPass> {
  void runOnOperation() override {
    getOperation().walk([&](FuncOp fn) { runOnFunc(fn); });
  }

  void runOnFunc(FuncOp fn) {
    TranslationMap map;
    SmallVector<Operation *> redundantOps;
    DenseSet<OperationComparator> uniqueOps;
    fn.walk([&](Operation *op) {
      // Case 1: The operation itself is unnecessary
      if (isa<EmitEqualityOp>(op) && op->getOperand(0) == op->getOperand(1)) {
        redundantOps.push_back(op);
        return WalkResult::advance();
      }

      // Case 2: An equivalent operation has already been performed.
      OperationComparator comp(op, map);
      if (auto it = uniqueOps.find(comp); it != uniqueOps.end()) {
        redundantOps.push_back(op);
        for (unsigned opNum = 0; opNum < op->getNumResults(); opNum++) {
          map[op->getResult(opNum)] = it->getOp()->getResult(opNum);
        }
      } else {
        uniqueOps.insert(comp);
      }

      return WalkResult::advance();
    });

    for (auto *op : redundantOps) {
      for (auto result : op->getResults()) {
        result.replaceAllUsesWith(map.at(result));
      }
      op->erase();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> llzk::createRedundantOperationEliminationPass() {
  return std::make_unique<RedundantOperationEliminationPass>();
};
