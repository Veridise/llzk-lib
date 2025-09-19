//===-- Versioning.h --------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include <mlir/Bytecode/BytecodeImplementation.h>

namespace llzk {

struct LLZKDialectVersion : public mlir::DialectVersion {
  static const LLZKDialectVersion &CurrentVersion();

  static mlir::FailureOr<LLZKDialectVersion> read(mlir::DialectBytecodeReader &reader);

  LLZKDialectVersion() : LLZKDialectVersion(0, 0, 0) {}
  LLZKDialectVersion(uint64_t majorV, uint64_t minorV, uint64_t patchV)
      : majorVersion(majorV), minorVersion(minorV), patchVersion(patchV) {}

  void write(mlir::DialectBytecodeWriter &writer) const;

  std::string str() const;

  std::strong_ordering operator<=>(const LLZKDialectVersion &other) const;

  bool operator==(const LLZKDialectVersion &other) const { return std::is_eq(operator<=>(other)); }

  uint64_t majorVersion, minorVersion, patchVersion;
};

/// @brief This implements the bytecode interface for the LLZK dialect.
template <typename DialectTy>
struct LLZKDialectBytecodeInterface : public mlir::BytecodeDialectInterface {

  LLZKDialectBytecodeInterface(mlir::Dialect *dia) : mlir::BytecodeDialectInterface(dia) {}

  /// @brief Writes the current version of the LLZK-lib to the given writer.
  void writeVersion(mlir::DialectBytecodeWriter &writer) const override {
    auto versionOr = writer.getDialectVersion<DialectTy>();
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
  std::unique_ptr<mlir::DialectVersion>
  readVersion(mlir::DialectBytecodeReader &reader) const override {
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
    const auto *llzkVersion = reinterpret_cast<const LLZKDialectVersion *>(&version);
    if (!llzkVersion) {
      return mlir::success();
    }
    const auto &current = LLZKDialectVersion::CurrentVersion();
    if (*llzkVersion > current) {
      return topLevelOp->emitError().append(
          "Cannot upgrade from current version ", current.str(), " to future version ",
          llzkVersion->str()
      );
    }
    if (*llzkVersion == current) {
      // No work to do, versions match.
      return mlir::success();
    }
    // NOTE: This is the point at which upgrade functionality should be added
    // for backwards compatibility.
    return topLevelOp->emitWarning().append(
        "LLZK version ", llzkVersion->str(), " is older than current version ", current.str(),
        " and no upgrade methods have been implemented. Proceed with caution."
    );
  }
};

} // namespace llzk
