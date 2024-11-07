#include "zkir/Dialect/ZKIR/Util/IncludeHelper.h"
#include "zkir/Dialect/ZKIR/Util/SymbolHelper.h"

#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/Parser/Parser.h>

#include <llvm/Support/SourceMgr.h>

namespace zkir {
using namespace mlir;

/// Parse the given `filename` and return the produced ModuleOp.
FailureOr<OwningOpRef<ModuleOp>> parseFile(const std::string &filename, Operation *origin) {
  std::string resolvedPath;
  auto buffer = zkir::GlobalSourceMgr::get().openIncludeFile(filename, resolvedPath);
  if (!buffer) {
    return origin->emitOpError() << "could not find file \"" << filename << "\"";
  }
  ParserConfig parseConfig(origin->getContext());
  llvm::StringRef contents = buffer.get().get()->getBuffer();
  llvm::outs() << "[parseFile] BEFORE [parseSourceString]" << "\n";
  if (auto r = parseSourceString<ModuleOp>(contents, parseConfig, resolvedPath)) {
    llvm::outs() << "[parseFile] AFTER [parseSourceString]" << "\n";
    return r;
  } else {
    return origin->emitOpError() << "could not parse file \"" << filename << "\"";
  }
}

FailureOr<ModuleOp> inlineTheInclude(MLIRContext *ctx, IncludeOp &incOp) {
  FailureOr<OwningOpRef<ModuleOp>> loadResult = incOp.loadModule();
  if (failed(loadResult)) {
    return failure();
  }

  ModuleOp importedMod = loadResult->release();
  // Check properties of the included file to ensure symbol resolution will still work.
  if (!importedMod->hasAttr(LANG_ATTR_NAME)) {
    return incOp.emitOpError()
        .append(
            "expected '", ModuleOp::getOperationName(), "' from included file to have \"",
            LANG_ATTR_NAME, "\" attribute"
        )
        .attachNote(importedMod.getLoc())
        .append("this should have \"", LANG_ATTR_NAME, "\" attribute");
  }
  if (importedMod.getSymNameAttr()) {
    return incOp.emitOpError()
        .append("expected '", ModuleOp::getOperationName(), "' from included file to be unnamed")
        .attachNote(importedMod.getLoc())
        .append("this should be unnamed");
  }

  // Rename the ModuleOp using the alias symbol name from the IncludeOp.
  importedMod.setSymNameAttr(incOp.getSymNameAttr());

  // Replace the IncludeOp with the loaded ModuleOp
  Operation *thisOp = incOp.getOperation();
  IRRewriter rewriter(ctx);
  rewriter.setInsertionPointAfter(thisOp);
  rewriter.insert(importedMod);
  rewriter.eraseOp(thisOp);

  return importedMod;
}

} // namespace zkir
