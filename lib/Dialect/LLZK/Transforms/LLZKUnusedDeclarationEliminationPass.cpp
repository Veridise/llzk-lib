#include "llzk/Dialect/LLZK/IR/Ops.h"
#include "llzk/Dialect/LLZK/Transforms/LLZKTransformationPasses.h"
#include "llzk/Dialect/LLZK/Util/SymbolHelper.h"
#include "llzk/Dialect/LLZK/Util/SymbolLookup.h"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>

#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Debug.h>

/// Include the generated base pass class definitions.
namespace llzk {
// the *DECL* macro is required when a pass has options to declare the option struct
#define GEN_PASS_DECL_UNUSEDDECLARATIONELIMINATIONPASS
#define GEN_PASS_DEF_UNUSEDDECLARATIONELIMINATIONPASS
#include "llzk/Dialect/LLZK/Transforms/LLZKTransformationPasses.h.inc"
} // namespace llzk

using namespace mlir;
using namespace llzk;

#define DEBUG_TYPE "llzk-unused-declaration-elim"

namespace {

inline bool isMainComponent(StructDefOp structDef) {
  return structDef.getName() == COMPONENT_NAME_MAIN;
}

/// @brief Get the fully-qualified field symbol.
/// @tparam FieldUserOp Either a FieldReadOp or a FieldWriteOp
template <typename FieldUserOp> SymbolRefAttr getFullFieldSymbol(FieldUserOp op) {
  FlatSymbolRefAttr fieldSym = op.getFieldNameAttr();
  auto structType = dyn_cast<StructType>(op.getComponent().getType());
  ensure(structType != nullptr, "given op type must operate on a struct type");
  SymbolRefAttr structSym = structType.getNameRef(); // this is fully qualified
  return appendLeaf(structSym, fieldSym);
}

class UnusedDeclarationEliminationPass
    : public llzk::impl::UnusedDeclarationEliminationPassBase<UnusedDeclarationEliminationPass> {

  /// @brief Shared context between the operations in this pass (field removal, struct removal)
  /// that doesn't need to be persisted after the pass completes.
  struct PassContext {
    DenseMap<SymbolRefAttr, StructDefOp> symbolToStruct;
    DenseMap<StructDefOp, SymbolRefAttr> structToSymbol;

    const SymbolRefAttr &getSymbol(StructDefOp s) { return structToSymbol.at(s); }
    StructDefOp getStruct(const SymbolRefAttr &sym) { return symbolToStruct.at(sym); }

    static PassContext populate(ModuleOp modOp) {
      PassContext ctx;

      modOp.walk<WalkOrder::PreOrder>([&ctx](StructDefOp structDef) {
        auto structSymbolRes = getPathFromTopRoot(structDef);
        ensure(succeeded(structSymbolRes), "failed to lookup struct symbol");
        SymbolRefAttr structSym = *structSymbolRes;
        ctx.symbolToStruct[structSym] = structDef;
        ctx.structToSymbol[structDef] = structSym;
      });
      return ctx;
    }
  };

  void runOnOperation() override {
    PassContext ctx = PassContext::populate(getOperation());
    // First, remove unused fields. This may allow more structs to be removed,
    // if their final remaining uses are as types for unused fields.
    removeUnusedFields(ctx);

    // Last, remove unused structs if configured
    if (removeStructs) {
      removeUnusedStructs(ctx);
    }
  }

  /// @brief Removes unused fields.
  /// A field is unused if it is never read from (only written to).
  /// @param structDef
  void removeUnusedFields(PassContext &ctx) {
    ModuleOp modOp = getOperation();

    // Map fully-qualified field symbols -> field ops
    DenseMap<SymbolRefAttr, FieldDefOp> fields;
    for (auto &[structDef, structSym] : ctx.structToSymbol) {
      structDef.walk([&](FieldDefOp field) {
        // We don't consider public fields in the main component for removal,
        // as these are output values and removing them would result in modifying
        // the overall circuit interface.
        if (!isMainComponent(structDef) || !field.hasPublicAttr()) {
          SymbolRefAttr fieldSym =
              appendLeaf(structSym, FlatSymbolRefAttr::get(field.getSymNameAttr()));
          llvm::errs() << "adding field " << fieldSym << "\n";
          fields[fieldSym] = field;
        }
      });
    }

    // Remove all fields that are read.
    modOp.walk([&](FieldReadOp readf) {
      SymbolRefAttr readFieldSym = getFullFieldSymbol(readf);
      fields.erase(readFieldSym);
    });

    // Remove all writes that reference the remaining fields, as these writes
    // are now known to only update write-only fields.
    modOp.walk([&](FieldWriteOp writef) {
      SymbolRefAttr writtenField = getFullFieldSymbol(writef);
      if (fields.contains(writtenField)) {
        // We need not check the users of a writef, since it produces no results.
        LLVM_DEBUG(
            llvm::dbgs() << "Removing write " << writef << " to write-only field " << writtenField
                         << '\n'
        );
        writef.erase();
      }
    });

    // Finally, erase the remaining fields.
    for (auto &[_, fieldDef] : fields) {
      LLVM_DEBUG(llvm::dbgs() << "Removing field " << fieldDef << '\n');
      fieldDef->erase();
    }
  }

  /// @brief Remove unused structs by looking for any uses of the struct's fully-qualified
  /// symbol. This catches any uses, such as field declarations of the struct's type
  /// or calls to any of the struct's methods.
  /// @param ctx
  void removeUnusedStructs(PassContext &ctx) {
    DenseMap<SymbolRefAttr, StructDefOp> unusedStructs = ctx.symbolToStruct;

    getOperation().walk([&](Operation *op) {
      // All structs are used, so stop looking
      if (unusedStructs.empty()) {
        return WalkResult::interrupt();
      }

      auto structParent = op->getParentOfType<StructDefOp>();
      if (structParent == nullptr) {
        return WalkResult::advance();
      }

      auto tryRemoveFromUnused = [&](Type ty) {
        if (auto structTy = dyn_cast<StructType>(ty)) {
          // This name ref is required to be fully qualified
          SymbolRefAttr sym = structTy.getNameRef();
          StructDefOp refStruct = ctx.getStruct(sym);
          if (refStruct != structParent) {
            // Only remove if this isn't a self reference
            unusedStructs.erase(sym);
          }
        }
      };

      // LLZK requires fully-qualified references to struct symbols. So, we
      // simply need to look for the struct symbol within this op's symbol uses.

      // Check operands
      for (Value operand : op->getOperands()) {
        tryRemoveFromUnused(operand.getType());
      }

      // Check results
      for (Value result : op->getResults()) {
        tryRemoveFromUnused(result.getType());
      }

      // Check block arguments
      for (Region &region : op->getRegions()) {
        for (Block &block : region) {
          for (BlockArgument arg : block.getArguments()) {
            tryRemoveFromUnused(arg.getType());
          }
        }
      }

      // Check attributes
      for (const auto &namedAttr : op->getAttrs()) {
        if (auto typeAttr = dyn_cast<TypeAttr>(namedAttr.getValue())) {
          tryRemoveFromUnused(typeAttr.getValue());
        }
      }

      return WalkResult::advance();
    });

    // All remaining structs are unused and should be removed.
    for (auto &[_, structDef] : unusedStructs) {
      if (!isMainComponent(structDef)) {
        LLVM_DEBUG(llvm::dbgs() << "Removing unused struct " << structDef.getName() << '\n');
        structDef->erase();
      }
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> llzk::createUnusedDeclarationEliminationPass() {
  return std::make_unique<UnusedDeclarationEliminationPass>();
};
