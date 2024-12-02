#pragma once

namespace mlir {
class DialectRegistry;
} // namespace mlir

namespace llzk {
void registerAllDialects(mlir::DialectRegistry &registry);
} // namespace llzk
