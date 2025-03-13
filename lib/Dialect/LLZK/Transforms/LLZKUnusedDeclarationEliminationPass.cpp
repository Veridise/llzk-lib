#include "llzk/Dialect/LLZK/IR/Ops.h"
#include "llzk/Dialect/LLZK/Transforms/LLZKTransformationPasses.h"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>

#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Debug.h>

/// Include the generated base pass class definitions.
namespace llzk {
#define GEN_PASS_DEF_UNUSEDDECLARATIONELIMINATIONPASS
#include "llzk/Dialect/LLZK/Transforms/LLZKTransformationPasses.h.inc"
} // namespace llzk

using namespace mlir;
using namespace llzk;

#define DEBUG_TYPE "llzk-unused-declaration-elimination"

namespace {

inline bool isMainComponent(StructDefOp structDef) {
  return structDef.getName() == COMPONENT_NAME_MAIN;
}

/// @brief Removes unused fields
/// A field is unused if it is never read from (only written to).
/// @param structDef
void removeUnusedFields(StructDefOp structDef) {
  structDef.walk([&](FieldDefOp field) {
    if (field->getUsers().empty()) {
      LLVM_DEBUG(llvm::dbgs() << "Removing unused field " << field << '\n');
      field->erase();
      return WalkResult::skip();
    }

    // Check if all users are writes, and if so, this is still "unused".
    SmallVector<Operation *> toRemove;
    for (auto user : field->getUsers()) {
      auto writef = dyn_cast<FieldWriteOp>(user);
      if (!writef) {
        return WalkResult::advance();
      }
      toRemove.push_back(writef);
    }
    LLVM_DEBUG(llvm::dbgs() << "Removing write-only field " << field << '\n');
    field->erase();

    // Erase all users and the field op, since this private field is only
    // ever written to.
    for (auto user : field->getUsers()) {
      LLVM_DEBUG(llvm::dbgs() << "    > removing field user " << user << '\n');
      user->erase();
    }

    return WalkResult::advance();
  });
}

class UnusedDeclarationEliminationPass
    : public llzk::impl::UnusedDeclarationEliminationPassBase<UnusedDeclarationEliminationPass> {
  void runOnOperation() override {
    // First, remove unused fields. This may allow more structs to be removed,
    // if their final remaining uses are as types for unused fields.
    walkStructs(removeUnusedFields);
    // Last, remove unused structs.
    walkStructs([this](StructDefOp s) { this->removeIfUnused(s); });
  }

  /// @brief Apply the given function for all non-Main structs contained within the current module.
  void walkStructs(function_ref<void(StructDefOp)> structTransformFn) {
    getOperation().walk([&](StructDefOp structDef) {
      if (!isMainComponent(structDef)) {
        structTransformFn(structDef);
      }
      // A walk optimization: since structs cannot be nested, skip instead of
      // advance to prevent unnecessary inner operation walks.
      return WalkResult::skip();
    });
  }

  /// @brief Removes unused structs. A struct is unused if compute and constrain
  /// are never called, and if the struct is not used in any declarations.
  void removeIfUnused(StructDefOp structDef) {
    if (structDef.getComputeFuncOp()->getUsers().empty() &&
        structDef.getConstrainFuncOp()->getUsers().empty() && structTypeUnused(structDef)) {
      LLVM_DEBUG(
          llvm::dbgs() << "Removing unused struct " << structDef.getFullyQualifiedName() << '\n'
      );
      structDef->erase();
    }
  }

  /// @brief Determines whether or not a struct type is used in the current module.
  /// Since struct types may be used in field declarations and as type variables,
  /// we cannot just look for field defs with the given struct type. Rather,
  /// we examine the types over all operations.
  bool structTypeUnused(StructDefOp structDef) {
    bool res = true;
    StructType targetTy = structDef.getType();

    // Pre-order traversal to avoid traversing the target struct.
    getOperation().walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (auto s = dyn_cast<StructDefOp>(op); s && s == structDef) {
        // Skip anything within the structDef itself
        return WalkResult::skip();
      }

      // Check operands
      for (Value operand : op->getOperands()) {
        if (typeContainsTarget(operand.getType(), targetTy)) {
          res = false;
          return WalkResult::interrupt();
        }
      }

      // Check results
      for (Value result : op->getResults()) {
        if (typeContainsTarget(result.getType(), targetTy)) {
          res = false;
          return WalkResult::interrupt();
        }
      }

      // Check block arguments
      for (Region &region : op->getRegions()) {
        for (Block &block : region) {
          for (BlockArgument arg : block.getArguments()) {
            if (typeContainsTarget(arg.getType(), targetTy)) {
              res = false;
              return WalkResult::interrupt();
            }
          }
        }
      }

      // Check attributes
      for (const auto &namedAttr : op->getAttrs()) {
        auto typeAttr = dyn_cast<TypeAttr>(namedAttr.getValue());
        if (typeAttr && typeContainsTarget(typeAttr.getValue(), targetTy)) {
          res = false;
          return WalkResult::interrupt();
        }
      }

      return WalkResult::advance();
    });

    return res;
  }

  /// @brief Determine if `ty` contains `targetTy` as one of its subtypes.
  bool typeContainsTarget(Type ty, Type targetTy) {
    bool found = false;
    ty.walk([&found, &targetTy](Type t) {
      if (t == targetTy) {
        found = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    return found;
  }
};

} // namespace

std::unique_ptr<mlir::Pass> llzk::createUnusedDeclarationEliminationPass() {
  return std::make_unique<UnusedDeclarationEliminationPass>();
};
