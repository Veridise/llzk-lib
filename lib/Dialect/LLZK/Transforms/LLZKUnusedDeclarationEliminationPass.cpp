#include "llzk/Dialect/LLZK/Analysis/CallGraphAnalyses.h"
#include "llzk/Dialect/LLZK/IR/Ops.h"
#include "llzk/Dialect/LLZK/Transforms/LLZKTransformationPasses.h"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/PostOrderIterator.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Debug.h>

#include <unordered_set>

/// Include the generated base pass class definitions.
namespace llzk {
#define GEN_PASS_DEF_UNUSEDDECLARATIONELIMINATIONPASS
#include "llzk/Dialect/LLZK/Transforms/LLZKTransformationPasses.h.inc"
} // namespace llzk

using namespace mlir;
using namespace llzk;

#define DEBUG_TYPE "llzk-unused-declaration-elimination"

namespace {

class UnusedDeclarationEliminationPass
    : public llzk::impl::UnusedDeclarationEliminationPassBase<UnusedDeclarationEliminationPass> {
  void runOnOperation() override {
    // Traverse structs from the bottom of the call graph up.
    // This way, we may create more unused structs along the way.
    DenseSet<StructDefOp> visited;
    SmallVector<StructDefOp> orderedStructs;
    auto &cga = getAnalysis<CallGraphAnalysis>();
    const llzk::CallGraph *callGraph = &cga.getCallGraph();
    for (auto it = llvm::po_begin(callGraph); it != llvm::po_end(callGraph); ++it) {
      const llzk::CallGraphNode *node = *it;
      if (node->isExternal()) {
        continue;
      }
      auto structDefRes = getParentOfType<StructDefOp>(node->getCalledFunction());
      if (succeeded(structDefRes)) {
        StructDefOp structDef = *structDefRes;
        if (visited.find(structDef) == visited.end()) {
          visited.insert(structDef);
          if (structDef.getName() != COMPONENT_NAME_MAIN) {
            orderedStructs.push_back(structDef);
          }
        }
      }
    }

    // We do these operations in a separate loop because once removal begins, we can no
    // longer safely iterate over the call graph we created.
    // - First remove all unused fields
    for (auto &structDef : orderedStructs) {
      removeUnusedFields(structDef);
    }
    // - Then check if any structs are now unused
    for (auto &structDef : orderedStructs) {
      removeUnusedStruct(structDef);
    }
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

  /// @brief Removes unused structs. A struct is unused if compute and constrain
  /// are never called, and if the struct is not used in any declarations.
  void removeUnusedStruct(StructDefOp structDef) {
    if (structDef.getComputeFuncOp()->getUsers().empty() &&
        structDef.getConstrainFuncOp()->getUsers().empty() && structTypeUnused(structDef)) {
      LLVM_DEBUG(
          llvm::dbgs() << "Removing unused struct " << structDef.getFullyQualifiedName() << '\n'
      );
      structDef->erase();
    }
  }

  bool structTypeUnused(StructDefOp structDef) {
    bool res = true;
    getOperation().walk([&res, &structDef](FieldDefOp fieldDef) {
      if (auto structTy = dyn_cast<StructType>(fieldDef.getType());
          structTy && structTy == structDef.getType()) {
        res = false;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    return res;
  }
};

} // namespace

std::unique_ptr<mlir::Pass> llzk::createUnusedDeclarationEliminationPass() {
  return std::make_unique<UnusedDeclarationEliminationPass>();
};
