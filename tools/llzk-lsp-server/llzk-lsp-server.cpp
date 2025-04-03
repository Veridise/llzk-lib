#include "llzk/Dialect/InitDialects.h"

#include <mlir/IR/DialectRegistry.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Tools/mlir-lsp-server/MlirLspServerMain.h>

#include <llvm/Support/PrettyStackTrace.h>

#include "tools/config.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  llvm::setBugReportMsg(
      "PLEASE submit a bug report to " BUG_REPORT_URL
      " and include the crash backtrace and inciting LLZK files.\n"
  );
  llzk::registerAllDialects(registry);
  return mlir::failed(mlir::MlirLspServerMain(argc, argv, registry));
}
