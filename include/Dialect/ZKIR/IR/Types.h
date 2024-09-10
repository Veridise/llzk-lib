#pragma once

#include "Dialect/ZKIR/IR/Attrs.h"
#include "Dialect/ZKIR/IR/Dialect.h"
#include "Dialect/ZKIR/IR/Enums.h"

#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/IR/Types.h>
#include <mlir/Interfaces/DataLayoutInterfaces.h>
#include <mlir/Interfaces/MemorySlotInterfaces.h>

#include <llvm/ADT/TypeSwitch.h>

// forward-declare ops
#define GET_OP_FWD_DEFINES
#include "Dialect/ZKIR/IR/Ops.h.inc"

// Include TableGen'd declarations
#define GET_TYPEDEF_CLASSES
#include "Dialect/ZKIR/IR/Types.h.inc"
