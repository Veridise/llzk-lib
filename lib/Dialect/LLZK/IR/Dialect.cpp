//===-- Dialect.cpp - Dialect method implementations ------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Config/Config.h"
#include "llzk/Dialect/LLZK/IR/Attrs.h"
#include "llzk/Dialect/LLZK/IR/Dialect.h"
#include "llzk/Dialect/LLZK/IR/Ops.h"
#include "llzk/Dialect/LLZK/IR/Types.h"

#include <mlir/Bytecode/BytecodeImplementation.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/Support/LLVM.h>

#include <compare>
#include <string>

// TableGen'd implementation files
#include "llzk/Dialect/LLZK/IR/Dialect.cpp.inc"

template <> struct mlir::FieldParser<llvm::APInt> {
  static mlir::FailureOr<llvm::APInt> parse(mlir::AsmParser &parser) {
    auto loc = parser.getCurrentLocation();
    llvm::APInt val;
    auto result = parser.parseOptionalInteger(val);
    if (!result.has_value() || *result) {
      return parser.emitError(loc, "expected integer value");
    } else {
      return val;
    }
  }
};

struct LLZKDialectVersion : public mlir::DialectVersion {
  static const LLZKDialectVersion &CurrentVersion() {
    static LLZKDialectVersion current(LLZK_VERSION_MAJOR, LLZK_VERSION_MINOR, LLZK_VERSION_PATCH);
    return current;
  }

  static mlir::FailureOr<LLZKDialectVersion> read(mlir::DialectBytecodeReader &reader) {
    LLZKDialectVersion v;
    if (mlir::failed(reader.readVarInt(v.majorVersion)) ||
        mlir::failed(reader.readVarInt(v.minorVersion)) ||
        mlir::failed(reader.readVarInt(v.patchVersion))) {
      return mlir::failure();
    }
    return v;
  }

  LLZKDialectVersion() : LLZKDialectVersion(0, 0, 0) {}
  LLZKDialectVersion(uint64_t majorV, uint64_t minorV, uint64_t patchV)
      : majorVersion(majorV), minorVersion(minorV), patchVersion(patchV) {}

  void write(mlir::DialectBytecodeWriter &writer) const {
    writer.writeVarInt(majorVersion);
    writer.writeVarInt(minorVersion);
    writer.writeVarInt(patchVersion);
  }

  std::string str() const {
    return (mlir::Twine(majorVersion) + "." + mlir::Twine(minorVersion) + "." +
            mlir::Twine(patchVersion))
        .str();
  }

  std::strong_ordering operator<=>(const LLZKDialectVersion &other) const {
    if (auto cmp = majorVersion <=> other.majorVersion; cmp != 0) {
      return cmp;
    }
    if (auto cmp = minorVersion <=> other.minorVersion; cmp != 0) {
      return cmp;
    }
    return patchVersion <=> other.patchVersion;
  }

  bool operator==(const LLZKDialectVersion &other) const { return std::is_eq(operator<=>(other)); }

  uint64_t majorVersion, minorVersion, patchVersion;
};

/// @brief This implements the bytecode interface for the LLZK dialect.
struct LLZKDialectBytecodeInterface : public mlir::BytecodeDialectInterface {

  LLZKDialectBytecodeInterface(mlir::Dialect *dialect) : mlir::BytecodeDialectInterface(dialect) {}

  /// @brief Writes the current version of the LLZK-lib to the given writer.
  void writeVersion(mlir::DialectBytecodeWriter &writer) const override {
    auto versionOr = writer.getDialectVersion<llzk::LLZKDialect>();
    // Check if a target version to emit was specified on the writer configs.
    if (mlir::succeeded(versionOr)) {
      reinterpret_cast<const LLZKDialectVersion *>(*versionOr)->write(writer);
    } else {
      // Otherwise, write the current version
      LLZKDialectVersion::CurrentVersion().write(writer);
    }
  }

  /// @brief Read the version of this dialect from the provided reader and return it as
  /// a `unique_ptr` to a dialect version object (or nullptr on failure).
  std::unique_ptr<mlir::DialectVersion> readVersion(mlir::DialectBytecodeReader &reader
  ) const override {
    auto versionOr = LLZKDialectVersion::read(reader);
    if (mlir::failed(versionOr)) {
      return nullptr;
    }
    return std::make_unique<LLZKDialectVersion>(std::move(*versionOr));
  }

  /// Hook invoked after parsing completed, if a version directive was present
  /// and included an entry for the current dialect. This hook offers the
  /// opportunity to the dialect to visit the IR and upgrades constructs emitted
  /// by the version of the dialect corresponding to the provided version.
  mlir::LogicalResult upgradeFromVersion(
      mlir::Operation *topLevelOp, const mlir::DialectVersion &version
  ) const override {
    auto llzkVersion = reinterpret_cast<const LLZKDialectVersion *>(&version);
    if (!llzkVersion) {
      return mlir::success();
    }
    const auto &current = LLZKDialectVersion::CurrentVersion();
    if (*llzkVersion > current) {
      topLevelOp->emitError(
          mlir::Twine("Cannot upgrade from current version ") + current.str() +
          " to future version " + llzkVersion->str()
      );
      return mlir::failure();
    }
    if (*llzkVersion == current) {
      // No work to do, versions match.
      return mlir::success();
    }
    // NOTE: This is the point at which upgrade functionality should be added
    // for backwards compatibility.
    topLevelOp->emitWarning(
        mlir::Twine("LLZK version ") + llzkVersion->str() + " is older than current version " +
        current.str() + " and no upgrade methods have been implemented. Proceed with caution."
    );
    return mlir::failure();
  }
};

// Need a complete declaration of storage classes for below
#define GET_TYPEDEF_CLASSES
#include "llzk/Dialect/LLZK/IR/Types.cpp.inc"
#define GET_ATTRDEF_CLASSES
#include "llzk/Dialect/LLZK/IR/Attrs.cpp.inc"

//===------------------------------------------------------------------===//
// LLZKDialect
//===------------------------------------------------------------------===//

auto llzk::LLZKDialect::initialize() -> void {
  // clang-format off
  addOperations<
    #define GET_OP_LIST
    #include "llzk/Dialect/LLZK/IR/Ops.cpp.inc"
  >();

  addTypes<
    #define GET_TYPEDEF_LIST
    #include "llzk/Dialect/LLZK/IR/Types.cpp.inc"
  >();

  addAttributes<
    #define GET_ATTRDEF_LIST
    #include "llzk/Dialect/LLZK/IR/Attrs.cpp.inc"
  >();
  // clang-format on
  addInterfaces<LLZKDialectBytecodeInterface>();
}
