#pragma once

#include <mlir/Pass/Pass.h>

namespace llzk {

std::unique_ptr<mlir::Pass> createInlineIncludesPass();

#define GEN_PASS_REGISTRATION
#include "llzk/Dialect/LLZK/Transforms/LLZKPasses.h.inc"

}; // namespace llzk
