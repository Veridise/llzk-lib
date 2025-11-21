#pragma once

#include <mlir/IR/Operation.h>

#include <llvm/ADT/EquivalenceClasses.h>

namespace llzk {

struct ValueLess {
  bool operator()(const mlir::Value &v1, const mlir::Value &v2) const {
    return v1.getAsOpaquePointer() < v2.getAsOpaquePointer();
  }
};

class LightweightSignalEquivalenceAnalysis {
  llvm::EquivalenceClasses<mlir::Value, ValueLess> equivalentSignals;

public:
  LightweightSignalEquivalenceAnalysis(mlir::Operation *op);
  bool areSignalsEquivalent(mlir::Value v1, mlir::Value v2);
};

} // namespace llzk
