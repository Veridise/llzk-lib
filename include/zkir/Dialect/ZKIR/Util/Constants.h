#pragma once

#include <llvm/ADT/StringRef.h>

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
