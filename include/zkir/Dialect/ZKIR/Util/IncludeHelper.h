#pragma once

#include "zkir/Dialect/ZKIR/IR/Ops.h"

#include <mlir/IR/BuiltinOps.h>

namespace zkir {

mlir::FailureOr<mlir::ModuleOp> loadModule(zkir::IncludeOp incOp);

mlir::FailureOr<mlir::ModuleOp> inlineTheInclude(mlir::MLIRContext *ctx, zkir::IncludeOp &incOp);

} // namespace zkir
