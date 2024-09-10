#pragma once

#include "Dialect/ZKIR/IR/Dialect.h"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Index/IR/IndexDialect.h>
#include <mlir/IR/DialectRegistry.h>

namespace zkir {
inline void registerAllDialects(mlir::DialectRegistry &registry) {
  registry.insert<zkir::ZKIRDialect, mlir::func::FuncDialect, mlir::index::IndexDialect>();
}
} // namespace zkir
