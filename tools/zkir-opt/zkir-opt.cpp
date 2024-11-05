#include "zkir/Dialect/InitDialects.h"
#include "zkir/Dialect/ZKIR/Transforms/ZKIRPasses.h"
#include "zkir/Dialect/ZKIR/Util/IncludeHelper.h"

#include <mlir/IR/DialectRegistry.h>
#include <mlir/Tools/mlir-opt/MlirOptMain.h>

#include <llvm/ADT/StringRef.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/Signals.h>

static llvm::cl::list<std::string> IncludeDirs(
    "I", llvm::cl::desc("Directory of include files"), llvm::cl::value_desc("directory"),
    llvm::cl::Prefix
);

int main(int argc, char **argv) {
  llvm::sys::PrintStackTraceOnErrorSignal(llvm::StringRef());

  // MLIR initialization
  mlir::DialectRegistry registry;
  zkir::registerAllDialects(registry);
  zkir::registerPasses();

  // Register and parse command line options.
  std::string inputFilename, outputFilename;
  std::tie(inputFilename, outputFilename) =
      registerAndParseCLIOptions(argc, argv, "zkir-opt", registry);

  // Set the include directories from CL option
  if (mlir::failed(zkir::GlobalSourceMgr::get().setup(IncludeDirs))) {
    return EXIT_FAILURE;
  }

  // Run 'mlir-opt'
  auto result = mlir::MlirOptMain(argc, argv, inputFilename, outputFilename, registry);
  return asMainReturnCode(result);
}
