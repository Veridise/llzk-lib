#pragma once

#include <llzk/Dialect/Struct/IR/Ops.h>

#include <mlir/IR/SymbolTable.h>
#include <mlir/Support/LogicalResult.h>

mlir::LogicalResult performInlining(
    mlir::SymbolTableCollection &tables,
    mlir::SmallVector<
        std::pair<llzk::component::StructDefOp, mlir::SmallVector<llzk::component::StructDefOp>>>
        &plan
);
