#pragma once

#include <llvm/ADT/Twine.h>
#include <llvm/Support/ErrorHandling.h>

namespace llzk {

inline void ensure(bool condition, llvm::Twine errMsg) {
  if (!condition) {
    llvm::report_fatal_error(errMsg);
  }
}

} // namespace llzk
