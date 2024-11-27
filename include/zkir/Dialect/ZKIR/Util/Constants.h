#pragma once

#include <mlir/IR/Diagnostics.h>

#include <llvm/ADT/DenseMapInfo.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/raw_ostream.h>

// Include TableGen'd declarations
#include "zkir/Dialect/ZKIR/Util/Constants.h.inc"

namespace zkir {
inline llvm::StringRef getInlineIncludesPassName() {
  return stringifyEnum(InlineIncludesPassName::I);
}
inline llvm::StringRef getInlineIncludesPassSummary() {
  return stringifyEnum(InlineIncludesPassSummary::I);
}
} // namespace zkir
