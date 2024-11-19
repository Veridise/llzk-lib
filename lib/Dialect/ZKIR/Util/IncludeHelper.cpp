#include "zkir/Dialect/ZKIR/Util/IncludeHelper.h"
#include "zkir/Dialect/ZKIR/Util/SymbolHelper.h"

#include <functional>
#include <llvm/Support/Casting.h>
#include <llvm/Support/MemoryBuffer.h>
#include <mlir/IR/AsmState.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Parser/Parser.h>

#include <llvm/Support/SourceMgr.h>
#include <mlir/Support/LogicalResult.h>
#include <zkir/Dialect/ZKIR/IR/Ops.h>

namespace zkir {
using namespace mlir;

struct OpenFile {
  std::string resolvedPath;
  std::unique_ptr<llvm::MemoryBuffer> buffer;
};

inline mlir::FailureOr<OpenFile>
openFile(std::function<InFlightDiagnostic()> &&emitError, const mlir::StringRef filename) {
  OpenFile r;

  auto buffer = zkir::GlobalSourceMgr::get().openIncludeFile(filename, r.resolvedPath);
  if (!buffer) {
    return emitError() << "could not find file \"" << filename << "\"";
  }
  r.buffer = std::move(*buffer);
  return std::move(r);
}

mlir::FailureOr<mlir::OwningOpRef<mlir::ModuleOp>>
parseFile(const mlir::StringRef filename, mlir::Operation *origin) {
  auto of = openFile([&]() { return origin->emitOpError(); }, filename);
  if (mlir::failed(of)) {
    return mlir::failure();
  }

  ParserConfig parseConfig(origin->getContext());
  llvm::StringRef contents = of->buffer->getBuffer();
  if (auto r =
          parseSourceString<ModuleOp>(contents, parseConfig, /*sourceName=*/of->resolvedPath)) {
    return r;
  } else {
    return origin->emitOpError() << "could not parse file \"" << filename << "\"";
  }
}

LogicalResult parseFile(const mlir::StringRef filename, Operation *origin, Block *container) {
  // Load raw contents of the file
  auto of = openFile([&]() { return origin->emitOpError(); }, filename);
  if (mlir::failed(of)) {
    return of;
  }

  // Parse the IR and write it in the destination block
  ParserConfig parseConfig(origin->getContext());
  llvm::StringRef contents = of->buffer->getBuffer();
  auto res = parseSourceString(contents, container, parseConfig, /*sourceName=*/of->resolvedPath);
  if (mlir::failed(res)) {
    return origin->emitOpError() << "could not parse file \"" << filename << "\"";
  }
  return mlir::success();
}

inline LogicalResult
validateLoadedModuleOp(std::function<InFlightDiagnostic()> &&emitError, ModuleOp importedMod) {
  if (!importedMod->hasAttr(LANG_ATTR_NAME)) {
    return emitError()
        .append(
            "expected '", mlir::ModuleOp::getOperationName(), "' from included file to have \"",
            LANG_ATTR_NAME, "\" attribute"
        )
        .attachNote(importedMod.getLoc())
        .append("this should have \"", LANG_ATTR_NAME, "\" attribute");
  }
  if (importedMod.getSymNameAttr()) {
    return emitError()
        .append("expected '", ModuleOp::getOperationName(), "' from included file to be unnamed")
        .attachNote(importedMod.getLoc())
        .append("this should be unnamed");
  }
  return mlir::success();
}

/// Manages the inlining and the associated memory used.
/// It has a SQL-esque workflow. The operation can be commited if everything looks fine
/// Or it will rollback when its lifetime
/// ends unless it was commited.
class InlineOperationsGuard {
public:
  InlineOperationsGuard(MLIRContext *ctx, IncludeOp &tIncOp)
      : incOp(tIncOp), rewriter(ctx), dest(rewriter.createBlock(incOp->getBlock()->getParent())) {}

  ~InlineOperationsGuard() {
    // We don't want the include op anymore so we can get rid of it
    if (commited) {
      rewriter.eraseOp(incOp);
      return;
    }

    // We did not replace the include op with the container
    // so we need to get cleanup.
    dest->erase();
  }

  /// Tells the guard that is safe to assume that the module was inserted into the destionation
  void moduleWasLoaded() {
    assert(!dest->empty());
    blockWritten = true;
  }

  // Attempts to get the module written into the block
  FailureOr<ModuleOp> getModule() {
    // If the block is not ready return failure but do not emit diagnostics.
    if (!blockWritten) {
      return failure();
    }

    if (dest->empty()) {
      return incOp->emitOpError() << "failed to inline the module. No operation was written";
    }

    auto &op = dest->front();
    if (!mlir::isa<mlir::ModuleOp>(op)) {
      return op.emitError()
          .append(
              "expected '", mlir::ModuleOp::getOperationName(),
              "' as top level operation of included file. Got: ", op.getName()
          )
          .attachNote(incOp.getLoc())
          .append("from file included here");
    }
    return mlir::cast<mlir::ModuleOp>(op);
  }

  Block *getDest() { return dest; }

  FailureOr<ModuleOp> commit() {
    // Locate where to insert the inlined module
    rewriter.setInsertionPointAfter(incOp);
    auto insertionPoint = rewriter.getInsertionPoint();
    {
      // This op will be invalid after inlining the block
      auto modRes = getModule();
      // Won't commit on a failed result
      if (mlir::failed(modRes)) {
        return modRes;
      }

      // Add the destination block after the insertion point.
      // dest becomes the source from which to move operations.
      rewriter.inlineBlockBefore(dest, rewriter.getInsertionBlock(), insertionPoint);
    }

    rewriter.setInsertionPointAfter(incOp);
    auto modOp = rewriter.getInsertionPoint();
    auto mod = llvm::dyn_cast<ModuleOp>(modOp);

    mod.setSymNameAttr(incOp.getSymNameAttr());

    // All good so we mark as commited and return a reference to the newly generated module.
    commited = true;
    return mod;
  }

private:
  bool commited = false, blockWritten = false;
  IncludeOp &incOp;
  IRRewriter rewriter;
  Block *dest;
};

/// Loads the file referenced by the IncludeOp and returns the ModuleOp contained in included file.
FailureOr<ModuleOp> inlineTheInclude(MLIRContext *ctx, IncludeOp &incOp) {
  InlineOperationsGuard guard(ctx, incOp);

  auto loadResult = parseFile(incOp.getPath(), incOp.getOperation(), guard.getDest());
  if (failed(loadResult)) {
    return loadResult;
  }
  guard.moduleWasLoaded();

  auto importedMod = guard.getModule();
  if (mlir::failed(importedMod)) {
    return importedMod; // getModule() already generates an error message
  }

  // Check properties of the included file to ensure symbol resolution will still work.
  auto validationResult =
      validateLoadedModuleOp([&]() { return incOp.emitOpError(); }, *importedMod);
  if (mlir::failed(validationResult)) {
    return validationResult;
  }

  return guard.commit();
}

} // namespace zkir
