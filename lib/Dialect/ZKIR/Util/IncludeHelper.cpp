#include "zkir/Dialect/ZKIR/Util/IncludeHelper.h"
#include "zkir/Dialect/ZKIR/Util/SymbolHelper.h"

#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/Parser/Parser.h>

#include <llvm/Support/SourceMgr.h>

namespace zkir {
using namespace mlir;

/// Parse the given `filename` and return the produced ModuleOp.
FailureOr<ModuleOp> parseFile(const std::string &filename, Operation *origin) {

  // NOTE: must use the override of parseSourceFile() that accepts SourceMgr instead of std::string
  // or else an extra error message with Unknown location is printed when the file does not exist.
  auto fileOrErr = llvm::MemoryBuffer::getFileOrSTDIN(filename);
  if (fileOrErr.getError()) {
    return origin->emitOpError() << "could not find file " << filename;
  }
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), SMLoc());
  ParserConfig parseConfig(origin->getContext());
  if (auto r = parseSourceFile<ModuleOp>(sourceMgr, parseConfig)) {
    return r.release();
  } else {
    return origin->emitOpError() << "could not parse file " << filename;
  }
}

FailureOr<ModuleOp> inlineTheInclude(MLIRContext *ctx, IncludeOp &incOp) {
  FailureOr<ModuleOp> loadResult = incOp.loadModule();
  if (succeeded(loadResult)) {
    ModuleOp importedMod = loadResult.value();
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
  }
  return loadResult;
}

} // namespace zkir
