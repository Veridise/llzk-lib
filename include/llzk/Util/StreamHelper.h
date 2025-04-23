//===-- StreamHelper.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include <llvm/Support/raw_ostream.h>

namespace llzk {

/// Wrapper for `llvm::raw_ostream` that filters out certain characters selected by a function.
class filtered_raw_ostream : public llvm::raw_ostream {
  llvm::raw_ostream &underlyingStream;
  std::function<bool(char)> filterFunc;

  void write_impl(const char *ptr, size_t size) override {
    for (size_t i = 0; i < size; ++i) {
      if (!filterFunc(ptr[i])) {
        underlyingStream << ptr[i];
      }
    }
  }

  uint64_t current_pos() const override { return underlyingStream.tell(); }

public:
  filtered_raw_ostream(llvm::raw_ostream &os, std::function<bool(char)> filter)
      : underlyingStream(os), filterFunc(std::move(filter)) {}

  ~filtered_raw_ostream() override { flush(); }
};

} // namespace llzk
