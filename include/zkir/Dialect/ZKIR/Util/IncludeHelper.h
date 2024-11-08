#pragma once

#include "zkir/Dialect/ZKIR/IR/Ops.h"

#include <mlir/IR/BuiltinOps.h>

#include <llvm/ADT/SmallString.h>
#include <llvm/Support/Path.h>
#include <llvm/Support/SourceMgr.h>

namespace zkir {

class GlobalSourceMgr {
  std::vector<std::string> includeDirectories;

public:
  static GlobalSourceMgr &get() {
    static GlobalSourceMgr theInstance;
    return theInstance;
  }

  mlir::LogicalResult setup(const std::vector<std::string> &includeDirs) {
    includeDirectories = includeDirs;
    return mlir::success();
  }

  // Adapted from mlir::SourceMgr::OpenIncludeFile() because SourceMgr is
  //   not a mature, usable component of MLIR.
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>>
  openIncludeFile(const std::string &filename, std::string &resolvedFile) {
    auto result = llvm::MemoryBuffer::getFile(filename);

    llvm::SmallString<64> pathBuffer(filename);
    // If the file didn't exist directly, see if it's in an include path.
    for (unsigned i = 0, e = includeDirectories.size(); i != e && !result; ++i) {
      pathBuffer = includeDirectories[i];
      llvm::sys::path::append(pathBuffer, filename);
      result = llvm::MemoryBuffer::getFile(pathBuffer);
    }

    if (result) {
      resolvedFile = static_cast<std::string>(pathBuffer);
    }

    return result;
  }
};

mlir::FailureOr<ImportedModuleOp> parseFile(const std::string &filename, mlir::Operation *origin);

mlir::FailureOr<mlir::ModuleOp> inlineTheInclude(mlir::MLIRContext *ctx, zkir::IncludeOp &incOp);

} // namespace zkir
