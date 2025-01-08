#pragma once

#include <llvm/Support/raw_ostream.h>

#include <string>
#include <vector>

namespace llzk {
namespace debug {

/// Generate a comma-separated string representation by traversing elements from `begin` to `end`
/// where the element type implements `operator<<`.
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

/// Generate a comma-separated string representation by traversing elements from
/// `collection.begin()` to `collection.end()` where the element type implements `operator<<`.
template <class InputIt> inline std::string toString(const InputIt &collection) {
  return toString(collection.begin(), collection.end());
}

} // namespace debug
} // namespace llzk
