#include "Dialect/ZKIR/Util/SymbolHelper.h"

#include <mlir/IR/BuiltinOps.h>

mlir::FailureOr<mlir::ModuleOp> zkir::getRootModule(mlir::Operation *op) {
  mlir::Operation *check = op;
  do {
    if (mlir::ModuleOp m = llvm::dyn_cast_if_present<mlir::ModuleOp>(check)) {
      // We add this attribute restriction because some stages of parsing have
      //  an extra module wrapping the top-level module from the input file.
      if (m->hasAttr(LANG_ATTR_NAME)) {
        return m;
      }
    }
  } while ((check = check->getParentOp()));
  //
  return op->emitOpError() << "has no ancestor module with \"" << LANG_ATTR_NAME << "\" attribute";
}
