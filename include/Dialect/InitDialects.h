#pragma once

#include "Dialect/ZKIR/IR/Dialect.h"

#include <mlir/IR/DialectRegistry.h>

namespace zkir {
inline void registerAllDialects(mlir::DialectRegistry &registry) {
  // clang-format off
  registry.insert<
    zkir::ZKIRDialect
  >();
  // clang-format on
}
} // namespace zkir
