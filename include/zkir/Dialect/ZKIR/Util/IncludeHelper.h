#pragma once

#include <mlir/IR/BuiltinOps.h>

#include <llvm/ADT/SmallString.h>
#include <llvm/Support/Path.h>
#include <llvm/Support/SourceMgr.h>

namespace zkir {
class IncludeOp;

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

template <typename OpTy> class ManagedOpPtr {
private:
  /// Pointer to the specific nested object (nullptr is allowed)
  OpTy *_nestedPtr;
  /// Keeps ModuleOp alive for the lifetime of this pointer
  mlir::OwningOpRef<mlir::ModuleOp> _module;

public:
  // Constructor to initialize with a pointer to an op nested within a ModuleOp whose lifetime
  // should be controlled by this object.
  ManagedOpPtr(OpTy *nestedOp, mlir::OwningOpRef<mlir::ModuleOp> &&owningModule)
      : _module(std::move(owningModule)), _nestedPtr(nestedOp) {}

  // Constructor to initialize with a pointer to an op whose lifetime is controlled externally.
  ManagedOpPtr(OpTy *op) : _module(), _nestedPtr(op) {}

  // Move constructor
  ManagedOpPtr(ManagedOpPtr &&other) noexcept
      : _nestedPtr(other._nestedPtr), _module(std::move(other._module)) {
    other._nestedPtr = nullptr;
  }

  // Move assignment operator
  ManagedOpPtr &operator=(ManagedOpPtr &&other) noexcept {
    if (this != &other) {
      _nestedPtr = other._nestedPtr;
      _module = std::move(other._module);
      other._nestedPtr = nullptr;
    }
    return *this;
  }

  // Deleted copy constructor and copy assignment to enforce unique ownership
  ManagedOpPtr(const ManagedOpPtr &) = delete;
  ManagedOpPtr &operator=(const ManagedOpPtr &) = delete;

  ~ManagedOpPtr() {
    if (ownsTheModule()) {
      llvm::outs() << "Destructing..." << _module.get().getSymName() << "\n";
    }
  }

  // Access the underlying nested pointer
  OpTy *get() const { return _nestedPtr; }
  OpTy *operator->() const { return _nestedPtr; }
  OpTy &operator*() const { return *_nestedPtr; }

  // bool isValid() const { return _nestedPtr != nullptr; }
  bool ownsTheModule() const { return _module.get() != nullptr; }

  explicit operator bool() const { return _nestedPtr; }

  template <class NewOpTy> ManagedOpPtr<NewOpTy> replacePtr(NewOpTy *ptr) {
    //_nestedPtr = nullptr;  // Invalidate original pointer to prevent accidental access.
    return ManagedOpPtr<NewOpTy>(ptr, std::move(_module));
  }
};

mlir::FailureOr<mlir::OwningOpRef<mlir::ModuleOp>>
parseFile(const std::string &filename, mlir::Operation *origin);

mlir::FailureOr<mlir::ModuleOp> inlineTheInclude(mlir::MLIRContext *ctx, zkir::IncludeOp &incOp);

} // namespace zkir
