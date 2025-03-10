#pragma once

#include <mlir/Pass/Pass.h>

namespace llzk {

std::unique_ptr<mlir::Pass> createDuplicateStructEliminationPass();

std::unique_ptr<mlir::Pass> createInlineIncludesPass();

std::unique_ptr<mlir::Pass> createRedundantReadAndWriteEliminationPass();

std::unique_ptr<mlir::Pass> createRedundantFunctionCallEliminationPass();

std::unique_ptr<mlir::Pass> createRedundantOperationEliminationPass();

#define GEN_PASS_REGISTRATION
#include "llzk/Dialect/LLZK/Transforms/LLZKTransformationPasses.h.inc"

}; // namespace llzk
