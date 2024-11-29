#pragma once

#include "zkir/Dialect/ZKIR/IR/Dialect.h"
#include "zkir/Dialect/ZKIR/IR/Enums.h"

#include <mlir/IR/DialectImplementation.h>

// Include TableGen'd declarations
#define GET_ATTRDEF_CLASSES
#include "zkir/Dialect/ZKIR/IR/Attrs.h.inc"

namespace zkir {} // namespace zkir
