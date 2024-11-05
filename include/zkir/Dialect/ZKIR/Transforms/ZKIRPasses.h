#pragma once

#include <mlir/Pass/Pass.h>

namespace zkir {

std::unique_ptr<mlir::Pass> createInlineIncludesPass();

#define GEN_PASS_REGISTRATION
#include "zkir/Dialect/ZKIR/Transforms/ZKIRPasses.h.inc"

}; // namespace zkir
