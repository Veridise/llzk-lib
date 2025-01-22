#pragma once

#include "llzk/Dialect/LLZK/IR/Ops.h"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

#include <llvm/ADT/SmallString.h>
#include <llvm/Support/Path.h>
#include <llvm/Support/SourceMgr.h>

namespace llzk {

class GlobalSourceMgr {
  std::vector<std::string> includeDirectories;

  static GlobalSourceMgr theInstance;

public:
  static GlobalSourceMgr &get() { return theInstance; }

  mlir::LogicalResult setup(const std::vector<std::string> &includeDirs) {
    includeDirectories = includeDirs;
    return mlir::success();
  }

  // Adapted from mlir::SourceMgr::OpenIncludeFile() because SourceMgr is
  //   not a mature, usable component of MLIR.
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>>
  openIncludeFile(const mlir::StringRef filename, std::string &resolvedFile) {
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

} // namespace llzk
