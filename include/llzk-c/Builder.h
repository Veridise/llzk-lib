//===-- Builder.h - C API for op builder --------------------------*- C -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// This header declares a type that supports the creation of operations and
// handles their insertion in blocks.
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_C_BUILDER_H
#define LLZK_C_BUILDER_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

#define DEFINE_C_API_STRUCT(name, storage)                                                         \
  struct name {                                                                                    \
    storage *ptr;                                                                                  \
  };                                                                                               \
  typedef struct name name

DEFINE_C_API_STRUCT(MlirOpBuilder, void);
DEFINE_C_API_STRUCT(MlirOpBuilderListener, void);

#undef DEFINE_C_API_STRUCT

// Commented out because they are not used for the callbacks in MLIR 18
// but are used for MLIR 20+, which we plan to move on in the near future.
#if 0
struct MlirOpInsertionPoint {
  MlirBlock block;
  MlirOperation point;
};
typedef struct MlirOpInsertionPoint MlirOpInsertionPoint;

struct MlirBlockInsertionPoint {
  MlirRegion region;
  MlirBlock point;
};
typedef struct MlirBlockInsertionPoint MlirBlockInsertionPoint;
#endif

typedef void (*MlirNotifyOperationInserted)(MlirOperation, void *);
typedef void (*MlirNotifyBlockInserted)(MlirBlock, void *);

//===----------------------------------------------------------------------===//
// MlirOpBuilder
//===----------------------------------------------------------------------===//

// The API for OpBuilder is left barebones for now since we only need a reference that we can pass
// to op build methods that we expose. More methods can be added as the need for them arises.

#define DECLARE_OP_BUILDER_CREATE_FN(suffix, ...)                                                  \
  MLIR_CAPI_EXPORTED MlirOpBuilder mlirOpBuilderCreate##suffix(__VA_ARGS__);                       \
  MLIR_CAPI_EXPORTED MlirOpBuilder mlirOpBuilderCreate##suffix##WithListener(                      \
      __VA_ARGS__, MlirOpBuilderListener                                                           \
  );

DECLARE_OP_BUILDER_CREATE_FN(, MlirContext)

#undef DECLARE_OP_BUILDER_CREATE_FN

/// Destroys the given builder.
MLIR_CAPI_EXPORTED void mlirOpBuilderDestroy(MlirOpBuilder);

//===----------------------------------------------------------------------===//
// MlirOpBuilderListener
//===----------------------------------------------------------------------===//

/// Creates a new mlir::OpBuilder::Listener. Takes one callback for each method of the Listener
/// interface and a pointer to user defined data.
MLIR_CAPI_EXPORTED MlirOpBuilderListener
mlirOpBuilderListenerCreate(MlirNotifyOperationInserted, MlirNotifyBlockInserted, void *);

/// Destroys the given listener.
MLIR_CAPI_EXPORTED void mlirOpBuilderListenerDestroy(MlirOpBuilderListener);

#ifdef __cplusplus
}
#endif

#endif
