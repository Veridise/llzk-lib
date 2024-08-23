#include "Dialect/InitDialects.h"

#include <mlir/IR/AsmState.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/Parser/Parser.h>

#include <llvm/ADT/StringRef.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/Signals.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/raw_ostream.h>

#include <cstdlib>
#include <string>

using namespace zkir;
namespace cl = llvm::cl;

static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input zkir file>"),
                                          cl::Required,
                                          cl::value_desc("filename"));

namespace {
enum Action { None, DumpAST, DumpMLIR };
} // namespace
static cl::opt<enum Action> emitAction(
    "emit", cl::desc("Select the kind of output desired"),
    cl::values(clEnumValN(DumpMLIR, "mlir", "output the MLIR dump")));

int dumpMLIR(mlir::Operation *ownedModule) {
  ownedModule->dump();
  return EXIT_SUCCESS;
}

int main(int argc, char **argv) {
  llvm::sys::PrintStackTraceOnErrorSignal(llvm::StringRef());

  // CLI option initialization
  cl::HideUnrelatedOptions({});
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  cl::ParseCommandLineOptions(argc, argv, "zkir compiler dev dump\n");

  // MLIR initialization
  mlir::DialectRegistry registry;
  zkir::registerAllDialects(registry);
  mlir::MLIRContext context(registry);

  llvm::SourceMgr sourceMgr;
  mlir::SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &context);

  // Read the source file
  mlir::OwningOpRef<mlir::Operation *> ownedModule =
      mlir::parseSourceFile<mlir::ModuleOp>(inputFilename.c_str(), sourceMgr,
                                            &context);
  if (!ownedModule) {
    return EXIT_FAILURE;
  }

  switch (emitAction) {
  case Action::DumpMLIR:
    return dumpMLIR(ownedModule.get());
  default:
    llvm::errs() << "No action specified (parsing only?), use -emit=<action>\n";
  }

  return EXIT_SUCCESS;
}
