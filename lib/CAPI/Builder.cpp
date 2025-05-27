//===-- Builder.cpp - C API for op builder ----------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include <mlir/CAPI/IR.h>
#include <mlir/CAPI/Support.h>
#include <mlir/IR/Builders.h>

#include <llzk-c/Builder.h>

using namespace mlir;

using OpBuilderT = OpBuilder;

namespace {

class ListenerT : public OpBuilder::Listener {
public:
  ListenerT(MlirNotifyOperationInserted op, MlirNotifyBlockInserted block, void *data)
      : opInsertedCb(op), blockInsertedCb(block), userData(data) {}

  void notifyOperationInserted(Operation *op) final { opInsertedCb(wrap(op), userData); }

  void notifyBlockCreated(Block *block) final {
    blockInsertedCb(wrap(block), userData);
    ;
  }

private:
  MlirNotifyOperationInserted opInsertedCb;
  MlirNotifyBlockInserted blockInsertedCb;
  void *userData = nullptr;
};

} // namespace

//===----------------------------------------------------------------------===//
// MlirOpBuilder
//===----------------------------------------------------------------------===//

MlirOpBuilder mlirOpBuilderCreate(MlirContext ctx) {
  return MlirOpBuilder {.ptr = new OpBuilderT(unwrap(ctx))};
}

MlirOpBuilder mlirOpBuilderCreateWithListener(MlirContext ctx, MlirOpBuilderListener listener) {
  auto *l = reinterpret_cast<ListenerT *>(listener.ptr);
  return MlirOpBuilder {.ptr = new OpBuilderT(unwrap(ctx), l)};
}

void mlirOpBuilderDestroy(MlirOpBuilder builder) {
  delete reinterpret_cast<OpBuilderT *>(builder.ptr);
}

//===----------------------------------------------------------------------===//
// MlirOpBuilderListener
//===----------------------------------------------------------------------===//

MlirOpBuilderListener mlirOpBuilderListenerCreate(
    MlirNotifyOperationInserted opCb, MlirNotifyBlockInserted blockCb, void *userData
) {
  return MlirOpBuilderListener {.ptr = new ListenerT(opCb, blockCb, userData)};
}

void mlirOpBuilderListenerDestroy(MlirOpBuilderListener listener) {
  delete reinterpret_cast<ListenerT *>(listener.ptr);
}
