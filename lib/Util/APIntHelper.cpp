//===-- APIntHelper.cpp - APInt helpers -------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Util/APIntHelper.h"

namespace llzk {

llvm::APSInt expandingAdd(llvm::APSInt lhs, llvm::APSInt rhs) {
  lhs = safeToSigned(lhs), rhs = safeToSigned(rhs);
  // +2: +1 for expansion, +1 to preserve the sign bit
  unsigned requiredBits = std::max(lhs.getActiveBits(), rhs.getActiveBits()) + 2;
  unsigned newBitwidth = std::max({requiredBits, lhs.getBitWidth(), rhs.getBitWidth()});
  return lhs.extend(newBitwidth) + rhs.extend(newBitwidth);
}

llvm::APSInt expandingSub(llvm::APSInt lhs, llvm::APSInt rhs) {
  lhs = safeToSigned(lhs), rhs = safeToSigned(rhs);
  // +2: +1 for expansion, +1 to preserve the sign bit
  unsigned requiredBits = std::max(lhs.getActiveBits(), rhs.getActiveBits()) + 2;
  unsigned newBitwidth = std::max({requiredBits, lhs.getBitWidth(), rhs.getBitWidth()});
  return lhs.extend(newBitwidth) - rhs.extend(newBitwidth);
}

llvm::APSInt expandingMul(llvm::APSInt lhs, llvm::APSInt rhs) {
  lhs = safeToSigned(lhs), rhs = safeToSigned(rhs);
  // +1 to preserve the sign bit
  unsigned requiredBits = lhs.getActiveBits() + rhs.getActiveBits() + 1;
  unsigned newBitwidth = std::max({requiredBits, lhs.getBitWidth(), rhs.getBitWidth()});
  return lhs.extend(newBitwidth) * rhs.extend(newBitwidth);
}

llvm::APSInt safeToSigned(llvm::APSInt i) {
  if (i.isSigned()) {
    return i;
  }
  i = i.extend(i.getBitWidth() + 1);
  i.setIsSigned(true);
  return i;
}

std::strong_ordering safeCmp(llvm::APSInt lhs, llvm::APSInt rhs) {
  lhs = safeToSigned(lhs), rhs = safeToSigned(rhs);
  unsigned requiredBits = std::max(lhs.getBitWidth(), rhs.getBitWidth());
  lhs = lhs.extend(requiredBits), rhs = rhs.extend(requiredBits);
  if (lhs < rhs) {
    return std::strong_ordering::less;
  } else if (lhs > rhs) {
    return std::strong_ordering::greater;
  } else {
    return std::strong_ordering::equal;
  }
}

} // namespace llzk
