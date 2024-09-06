#pragma once

#include <mlir/Pass/Pass.h>

namespace zkir {
std::unique_ptr<mlir::Pass> createHelloWorldPass();

#define GEN_PASS_REGISTRATION
#include "Dialect/ZKIR/Transforms/ZKIRPasses.h.inc"
}; // namespace zkir
