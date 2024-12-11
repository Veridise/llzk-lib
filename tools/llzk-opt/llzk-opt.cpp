#include "llzk/Dialect/InitDialects.h"
#include "llzk/Dialect/LLZK/Analysis/CallGraphPasses.h"
#include "llzk/Dialect/LLZK/Transforms/LLZKPasses.h"
#include "llzk/Dialect/LLZK/Util/IncludeHelper.h"

#include <mlir/IR/DialectRegistry.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Pass/PassRegistry.h>
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
  llzk::registerAllDialects(registry);
  llzk::registerAnalysisPasses();
  llzk::registerTransformationPasses();

  // Register and parse command line options.
  std::string inputFilename, outputFilename;
  std::tie(inputFilename, outputFilename) =
      registerAndParseCLIOptions(argc, argv, "llzk-opt", registry);

  // Set the include directories from CL option
  if (mlir::failed(llzk::GlobalSourceMgr::get().setup(IncludeDirs))) {
    return EXIT_FAILURE;
  }

  // Run 'mlir-opt'
  auto result = mlir::MlirOptMain(argc, argv, inputFilename, outputFilename, registry);
  return asMainReturnCode(result);
}
