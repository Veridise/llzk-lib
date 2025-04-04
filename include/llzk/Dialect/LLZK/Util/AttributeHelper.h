//===-- AttributeHelper.h ---------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include <mlir/IR/BuiltinAttributes.h>

#include <llvm/ADT/APInt.h>

namespace llzk {

inline llvm::APInt toAPInt(int64_t i) { return llvm::APInt(64, i); }
inline int64_t fromAPInt(llvm::APInt i) { return i.getZExtValue(); }

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
