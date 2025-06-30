//===-- Field.cpp -----------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Analysis/Field.h"

#include <llvm/ADT/Twine.h>

namespace llzk {

Field::Field(std::string_view primeStr) : primeMod(llvm::APSInt(primeStr)) {
  halfPrime = (primeMod + felt(1)) / felt(2);
}

const Field &Field::getField(const char *fieldName) {
  static llvm::DenseMap<llvm::StringRef, Field> knownFields;
  static std::once_flag fieldsInit;
  std::call_once(fieldsInit, initKnownFields, knownFields);

  if (auto it = knownFields.find(fieldName); it != knownFields.end()) {
    return it->second;
  }
  llvm::report_fatal_error("field \"" + llvm::Twine(fieldName) + "\" is unsupported");
}

void Field::initKnownFields(llvm::DenseMap<llvm::StringRef, Field> &knownFields) {
  // bn128/254, default for circom
  knownFields.try_emplace(
      "bn128",
      Field("21888242871839275222246405745257275088696311157297823662689037894645226208583")
  );
  knownFields.try_emplace("bn254", knownFields.at("bn128"));
  // 15 * 2^27 + 1, default for zirgen
  knownFields.try_emplace("babybear", Field("2013265921"));
  // 2^64 - 2^32 + 1, used for plonky2
  knownFields.try_emplace("goldilocks", Field("18446744069414584321"));
  // 2^31 - 1, used for Plonky3
  knownFields.try_emplace("mersenne31", Field("2147483647"));
}

llvm::APSInt Field::reduce(llvm::APSInt i) const {
  unsigned maxBits = std::max(i.getBitWidth(), bitWidth());
  llvm::APSInt m = (i.extend(maxBits) % prime().extend(maxBits)).trunc(bitWidth());
  if (m.isNegative()) {
    return prime() + m;
  }
  return m;
}

llvm::APSInt Field::reduce(unsigned i) const {
  auto ap = llvm::APSInt(llvm::APInt(bitWidth(), i));
  return reduce(ap);
}

} // namespace llzk
