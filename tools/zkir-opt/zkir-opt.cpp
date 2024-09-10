#include "Dialect/InitDialects.h"

#include <mlir/IR/DialectRegistry.h>
#include <mlir/Tools/mlir-opt/MlirOptMain.h>

#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Signals.h>

int main(int argc, char **argv) {
  llvm::sys::PrintStackTraceOnErrorSignal(llvm::StringRef());

  // MLIR initialization
  mlir::DialectRegistry registry;
  zkir::registerAllDialects(registry);

  auto result = mlir::MlirOptMain(argc, argv, "zkir-opt", registry);
  return asMainReturnCode(result);
}
