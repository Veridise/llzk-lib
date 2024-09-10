#pragma once

#include "Dialect/ZKIR/IR/Attrs.h"
#include "Dialect/ZKIR/IR/Dialect.h"
#include "Dialect/ZKIR/IR/Types.h"

#include <mlir/Bytecode/BytecodeOpInterface.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/Interfaces/CallInterfaces.h>
#include <mlir/Interfaces/DataLayoutInterfaces.h>
#include <mlir/Interfaces/InferTypeOpInterface.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

#include <llvm/ADT/SmallVector.h>

#include <optional>

// Include TableGen'd declarations
#define GET_OP_CLASSES
#include "Dialect/ZKIR/IR/Ops.h.inc"
