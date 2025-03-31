#include "llzk/Dialect/LLZK/Util/APIntHelper.h"

namespace llzk {

llvm::APSInt expandingAdd(const llvm::APSInt &lhs, const llvm::APSInt &rhs) {
  unsigned requiredBits = std::max(lhs.getActiveBits(), rhs.getActiveBits()) + 1;
  unsigned newBitwidth = std::max({requiredBits, lhs.getBitWidth(), rhs.getBitWidth()});
  return lhs.extend(newBitwidth) + rhs.extend(newBitwidth);
}

llvm::APSInt expandingSub(const llvm::APSInt &lhs, const llvm::APSInt &rhs) {
  unsigned requiredBits = std::max(lhs.getActiveBits(), rhs.getActiveBits()) + 1;
  unsigned newBitwidth = std::max({requiredBits, lhs.getBitWidth(), rhs.getBitWidth()});
  return lhs.extend(newBitwidth) - rhs.extend(newBitwidth);
}

llvm::APSInt expandingMul(const llvm::APSInt &lhs, const llvm::APSInt &rhs) {
  unsigned requiredBits = lhs.getActiveBits() + rhs.getActiveBits();
  unsigned newBitwidth = std::max({requiredBits, lhs.getBitWidth(), rhs.getBitWidth()});
  return lhs.extend(newBitwidth) * rhs.extend(newBitwidth);
}

std::strong_ordering safeCmp(const llvm::APSInt &lhs, const llvm::APSInt &rhs) {
  unsigned requiredBits = std::max(lhs.getBitWidth(), rhs.getBitWidth());
  auto a = lhs.extend(requiredBits), b = rhs.extend(requiredBits);
  if (a < b) {
    return std::strong_ordering::less;
  } else if (a > b) {
    return std::strong_ordering::greater;
  } else {
    return std::strong_ordering::equal;
  }
}

} // namespace llzk
