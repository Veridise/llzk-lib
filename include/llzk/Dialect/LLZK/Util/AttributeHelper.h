#pragma once

#include <mlir/IR/BuiltinAttributes.h>

namespace llzk {

inline bool isNullOrEmpty(mlir::ArrayAttr a) { return !a || a.empty(); }
inline bool isNullOrEmpty(mlir::DenseArrayAttr a) { return !a || a.empty(); }
inline bool isNullOrEmpty(mlir::DictionaryAttr a) { return !a || a.empty(); }

inline void appendWithoutType(mlir::raw_ostream &os, mlir::Attribute a) { a.print(os, true); }
inline std::string stringWithoutType(mlir::Attribute a) {
  std::string output;
  llvm::raw_string_ostream oss(output);
  appendWithoutType(oss, a);
  return output;
}

} // namespace llzk
