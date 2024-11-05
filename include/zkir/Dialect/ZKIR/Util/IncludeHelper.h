#pragma once

#include "zkir/Dialect/ZKIR/IR/Ops.h"

#include <mlir/IR/BuiltinOps.h>

namespace zkir {

mlir::FailureOr<mlir::ModuleOp> parseFile(const std::string &filename, mlir::Operation *origin);

mlir::FailureOr<mlir::ModuleOp> inlineTheInclude(mlir::MLIRContext *ctx, zkir::IncludeOp &incOp);

} // namespace zkir
