#pragma once

#include "llzk/Dialect/LLZK/IR/Dialect.h"
#include "llzk/Dialect/LLZK/IR/Enums.h"

#include <mlir/IR/DialectImplementation.h>

// Include TableGen'd declarations
#define GET_ATTRDEF_CLASSES
#include "llzk/Dialect/LLZK/IR/Attrs.h.inc"

namespace llzk {} // namespace llzk
