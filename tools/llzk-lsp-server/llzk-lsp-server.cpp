#include "llzk/Dialect/InitDialects.h"

#include <mlir/Dialect/Index/IR/IndexDialect.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/Tools/mlir-lsp-server/MlirLspServerMain.h>

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  llzk::registerAllDialects(registry);
  return mlir::failed(mlir::MlirLspServerMain(argc, argv, registry));
}
