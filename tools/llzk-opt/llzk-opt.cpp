//===-- llzk-opt.cpp - LLZK opt tool ----------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements a version of the mlir-opt tool configured for use on
/// LLZK files.
///
//===----------------------------------------------------------------------===//

#include "llzk/Config/Config.h"
#include "llzk/Dialect/InitDialects.h"
#include "llzk/Dialect/LLZK/Analysis/AnalysisPasses.h"
#include "llzk/Dialect/LLZK/Transforms/LLZKTransformationPasses.h"
#include "llzk/Dialect/LLZK/Util/IncludeHelper.h"
#include "llzk/Dialect/LLZK/Validators/LLZKValidationPasses.h"

#include <mlir/IR/DialectRegistry.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Pass/PassRegistry.h>
#include <mlir/Tools/mlir-opt/MlirOptMain.h>

#include <llvm/ADT/StringRef.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/Signals.h>

#include "tools/config.h"

static llvm::cl::list<std::string> IncludeDirs(
    "I", llvm::cl::desc("Directory of include files"), llvm::cl::value_desc("directory"),
    llvm::cl::Prefix
);

static llvm::cl::opt<bool>
    PrintAllOps("print-llzk-ops", llvm::cl::desc("Print a list of all ops registered in LLZK"));

int main(int argc, char **argv) {
  llvm::sys::PrintStackTraceOnErrorSignal(llvm::StringRef());
  llvm::setBugReportMsg("PLEASE submit a bug report to " BUG_REPORT_URL
                        " and include the crash backtrace, relevant LLZK files,"
                        " and associated run script(s).\n");
  llvm::cl::AddExtraVersionPrinter([](llvm::raw_ostream &os) {
    os << "\nLLZK (" LLZK_URL "):\n  LLZK version " LLZK_VERSION_STRING "\n";
  });

  // MLIR initialization
  mlir::DialectRegistry registry;
  llzk::registerAllDialects(registry);
  llzk::registerAnalysisPasses();
  llzk::registerTransformationPasses();
  llzk::registerTransformationPassPipelines();
  llzk::registerValidationPasses();

  // Register and parse command line options.
  std::string inputFilename, outputFilename;
  std::tie(inputFilename, outputFilename) =
      registerAndParseCLIOptions(argc, argv, "llzk-opt", registry);

  if (PrintAllOps) {
    mlir::MLIRContext context;
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
    llvm::outs() << "All ops registered in LLZK IR: {\n";
    for (const auto &opName : context.getRegisteredOperations()) {
      llvm::outs() << "  " << opName.getStringRef() << '\n';
    }
    llvm::outs() << "}\n";
    return EXIT_SUCCESS;
  }

  // Set the include directories from CL option
  if (mlir::failed(llzk::GlobalSourceMgr::get().setup(IncludeDirs))) {
    return EXIT_FAILURE;
  }

  // Run 'mlir-opt'
  auto result = mlir::MlirOptMain(argc, argv, inputFilename, outputFilename, registry);
  return asMainReturnCode(result);
}
