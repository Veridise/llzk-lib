#pragma once

#include "mlir/IR/DialectRegistry.h"

namespace r1cs {
void registerAllDialects(mlir::DialectRegistry &registry);
}
