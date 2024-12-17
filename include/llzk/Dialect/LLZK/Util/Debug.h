#pragma once

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/raw_ostream.h>

#include <string>
#include <vector>

namespace llzk {
namespace debug {

template <class InputIt> std::string toString(InputIt begin, InputIt end) {
  std::string output;
  llvm::raw_string_ostream oss(output);
  oss << "[";
  for (auto it = begin; it != end; ++it) {
    oss << *it;
    if (std::next(it) != end) {
      oss << ", ";
    }
  }
  oss << "]";
  return output;
}

/// Generate a string representation of a std::vector
template <typename T> inline std::string toString(const std::vector<T> &vec) {
  return toString(vec.begin(), vec.end());
}

/// Generate a string representation of a llvm::SmallVector
template <typename T, unsigned N> inline std::string toString(const llvm::SmallVector<T, N> &vec) {
  return toString(vec.begin(), vec.end());
}

/// Generate a string representation of a llvm::ArrayRef
template <typename T> inline std::string toString(const llvm::ArrayRef<T> &vec) {
  return toString(vec.begin(), vec.end());
}

} // namespace debug
} // namespace llzk
