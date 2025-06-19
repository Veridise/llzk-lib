//===-- Debug.h -------------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include <mlir/IR/Attributes.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/IR/Value.h>
#include <mlir/Interfaces/MemorySlotInterfaces.h>

#include <llvm/ADT/DenseMap.h>
#include <llvm/Support/raw_ostream.h>

#include <optional>
#include <string>

namespace llzk {
namespace debug {

namespace {

// Define this concept instead of `std::ranges::range` because certain classes (like OperandRange)
// do not work with `std::ranges::range`.
template <typename T>
concept Iterable = requires(T t) {
  std::begin(t);
  std::end(t);
};

struct Appender {
  llvm::raw_string_ostream stream;
  Appender(std::string &out) : stream(out) {}

  void append(const mlir::MemorySlot &a);
  void append(const mlir::DestructurableMemorySlot &a);
  void append(const mlir::OpOperand &a);
  void append(const mlir::NamedAttribute &a);
  void append(const mlir::SymbolTable::SymbolUse &a);
  template <typename T> void append(const std::optional<T> &a);
  template <typename Any> void append(const Any &value);
  template <typename A, typename B> void append(const std::pair<A, B> &a);
  template <typename A, typename B> void append(const llvm::detail::DenseMapPair<A, B> &a);
  template <Iterable InputIt> void append(const InputIt &collection);
  template <typename InputIt> void appendList(InputIt begin, InputIt end);
  template <typename Any> Appender &operator<<(const Any &v);
};

void Appender::append(const mlir::MemorySlot &a) {
  stream << "ptr: " << a.ptr << "; type: " << a.elemType;
}

void Appender::append(const mlir::DestructurableMemorySlot &a) {
  stream << "ptr: " << a.ptr << "; type: " << a.elemType << "; elementPtrs:\n";
  for (auto &p : a.elementPtrs) {
    stream << "  ";
    append(p);
    stream << '\n';
  }
}

void Appender::append(const mlir::OpOperand &a) { stream << a.get(); }

void Appender::append(const mlir::NamedAttribute &a) {
  stream << a.getName() << '=' << a.getValue();
}

void Appender::append(const mlir::SymbolTable::SymbolUse &a) { stream << a.getUser()->getName(); }

template <typename T> inline void Appender::append(const std::optional<T> &a) {
  if (a.has_value()) {
    append(a.value());
  } else {
    stream << "NONE";
  }
}

template <typename Any> void Appender::append(const Any &value) { stream << value; }

template <typename A, typename B> void Appender::append(const std::pair<A, B> &a) {
  stream << '(';
  append(a.first);
  stream << ',';
  append(a.second);
  stream << ')';
}

template <typename A, typename B> void Appender::append(const llvm::detail::DenseMapPair<A, B> &a) {
  stream << '(';
  append(a.first);
  stream << ',';
  append(a.second);
  stream << ')';
}

template <Iterable InputIt> inline void Appender::append(const InputIt &collection) {
  appendList(std::begin(collection), std::end(collection));
}

template <typename InputIt> void Appender::appendList(InputIt begin, InputIt end) {
  stream << '[';
  llvm::interleave(begin, end, [this](const auto &n) { append(n); }, [this] { stream << ", "; });
  stream << ']';
}

template <typename Any> Appender &Appender::operator<<(const Any &v) {
  append(v);
  return *this;
}

} // namespace

/// Generate a comma-separated string representation by traversing elements from `begin` to `end`
/// where the element type implements `operator<<`.
template <typename InputIt> inline std::string toStringList(InputIt begin, InputIt end) {
  std::string output;
  Appender(output).appendList(begin, end);
  return output;
}

/// Generate a comma-separated string representation by traversing elements from
/// `collection.begin()` to `collection.end()` where the element type implements `operator<<`.
template <typename InputIt> inline std::string toStringList(const InputIt &collection) {
  return toStringList(collection.begin(), collection.end());
}

template <typename InputIt>
inline std::string toStringList(const std::optional<InputIt> &optionalCollection) {
  if (optionalCollection.has_value()) {
    return toStringList(optionalCollection.value());
  } else {
    return "NONE";
  }
}

template <typename T> inline std::string toStringOne(const T &value) {
  std::string output;
  Appender(output).append(value);
  return output;
}

void dumpSymbolTableWalk(mlir::Operation *symbolTableOp);

void dumpSymbolTable(llvm::raw_ostream &stream, mlir::SymbolTable &symTab, unsigned indent = 0);

void dumpSymbolTable(mlir::SymbolTable &symTab);

void dumpSymbolTables(llvm::raw_ostream &stream, mlir::SymbolTableCollection &tables);

void dumpSymbolTables(mlir::SymbolTableCollection &tables);

void dumpToFile(mlir::Operation *op, llvm::StringRef filename);

} // namespace debug
} // namespace llzk
