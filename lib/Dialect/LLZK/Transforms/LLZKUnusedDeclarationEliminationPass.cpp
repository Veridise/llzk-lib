#include "llzk/Dialect/LLZK/IR/Ops.h"
#include "llzk/Dialect/LLZK/Transforms/LLZKTransformationPasses.h"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallVector.h>

#include <unordered_set>

/// Include the generated base pass class definitions.
namespace llzk {
#define GEN_PASS_DEF_UNUSEDDECLARATIONELIMINATIONPASS
#include "llzk/Dialect/LLZK/Transforms/LLZKTransformationPasses.h.inc"
} // namespace llzk

using namespace mlir;
using namespace llzk;

namespace {

class UnusedDeclarationEliminationPass
    : public llzk::impl::UnusedDeclarationEliminationPassBase<UnusedDeclarationEliminationPass> {
  void runOnOperation() override {
    auto modOp = getOperation();
    modOp.walk([&](StructDefOp s) {
      // Skip the main circuit, as this determines our external interface (inputs/outputs)
      if (s.getName() == COMPONENT_NAME_MAIN) {
        return;
      }
      removeUnusedFields(s);
      removeUnusedStruct(s);
    });
  }

  /// @brief Removes unused fields
  /// A field is unused if it is never read from (only written to).
  /// @param structDef
  void removeUnusedFields(StructDefOp structDef) {
    structDef.walk([&](FieldDefOp field) {
      if (field->getUsers().empty()) {
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
      field->erase();

      // Erase all users and the field op, since this private field is only
      // ever written to.
      for (auto user : field->getUsers()) {
        user->erase();
      }

      return WalkResult::advance();
    });
  }

  /// @brief Removes unused structs. A struct is unused if compute and constrain
  /// are never called.
  void removeUnusedStruct(StructDefOp structDef) {
    if (structDef.getComputeFuncOp()->getUsers().empty() &&
        structDef.getConstrainFuncOp()->getUsers().empty()) {
      structDef->erase();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> llzk::createUnusedDeclarationEliminationPass() {
  return std::make_unique<UnusedDeclarationEliminationPass>();
};
