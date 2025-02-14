#pragma once

#include <mlir/IR/BuiltinAttributes.h>

#include <llvm/ADT/Twine.h>
#include <llvm/Support/ErrorHandling.h>

namespace llzk {

inline void ensure(bool condition, llvm::Twine errMsg) {
  if (!condition) {
    llvm::report_fatal_error(errMsg);
  }
}

inline bool isNullOrEmpty(mlir::ArrayAttr a) { return !a || a.empty(); }
inline bool isNullOrEmpty(mlir::DenseArrayAttr a) { return !a || a.empty(); }
inline bool isNullOrEmpty(mlir::DictionaryAttr a) { return !a || a.empty(); }

} // namespace llzk
