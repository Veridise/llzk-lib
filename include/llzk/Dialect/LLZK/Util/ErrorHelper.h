#pragma once

#include <llvm/ADT/STLFunctionalExtras.h>
#include <llvm/ADT/Twine.h>
#include <llvm/Support/ErrorHandling.h>

namespace llzk {

using EmitErrorFn = llvm::function_ref<mlir::InFlightDiagnostic()>;

inline void ensure(bool condition, llvm::Twine errMsg) {
  if (!condition) {
    llvm::report_fatal_error(errMsg);
  }
}

} // namespace llzk
