#include "zkir/Dialect/ZKIR/Util/IncludeHelper.h"
#include "zkir/Dialect/ZKIR/Util/SymbolHelper.h"

#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/Parser/Parser.h>

namespace zkir {
using namespace mlir;

namespace {

/// Parse the given `filename` and return the produced ModuleOp.
OwningOpRef<ModuleOp> parseFile(MLIRContext *context, const std::string &filename) {
  ParserConfig parseConfig(context);
  return parseSourceFile<ModuleOp>(filename, parseConfig);
}
} // namespace

FailureOr<ModuleOp> loadModule(IncludeOp incOp) {
  StringAttr importPath = incOp.getPathAttr();
  ModuleOp importedMod = parseFile(incOp.getContext(), importPath.str()).release();
  if (!importedMod) {
    return incOp.emitOpError() << "could not load file " << importPath;
  }
  return importedMod;
}

FailureOr<ModuleOp> inlineTheInclude(MLIRContext *ctx, IncludeOp &incOp) {
  FailureOr<ModuleOp> otherMod = loadModule(incOp);
  if (failed(otherMod)) {
    return failure();
  }

  ModuleOp importedMod = otherMod.value();
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
