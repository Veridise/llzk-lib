#pragma once

namespace mlir {
class DialectRegistry;
} // namespace mlir

namespace zkir {
void registerAllDialects(mlir::DialectRegistry &registry);
} // namespace zkir
