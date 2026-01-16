//===-- Ops.td ---------------------------------------------*- tablegen -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_POD_OPS
#define LLZK_POD_OPS

include "llzk/Dialect/POD/IR/Dialect.td"
include "llzk/Dialect/POD/IR/Types.td"
include "llzk/Dialect/Shared/OpTraits.td"

include "mlir/IR/OpBase.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class PODDialectOp<string mnemonic, list<Trait> traits = []>
    : Op<PODDialect, mnemonic, traits>;

def LLZK_NewPodOp
    : PODDialectOp<"new", [Pure, AttrSizedOperandSegments,
                           VerifySizesForMultiAffineOps<1>,
                           DeclareOpInterfaceMethods<
                               OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "create a new plain-old-data struct";
  let arguments = (ins
      // Initialization values
      Variadic<AnyLLZKType>:$initialValues,
      DefaultValuedAttr<StrArrayAttr, "{}">:$initializedRecords,
      // Affine map arguments
      VariadicOfVariadic<Index, "mapOpGroupSizes">:$mapOperands,
      DefaultValuedAttr<DenseI32ArrayAttr, "{}">:$numDimsPerMap,
      DenseI32ArrayAttr:$mapOpGroupSizes);
  let results = (outs LLZK_PODType:$result);
  let skipDefaultBuilders = 1;
  let builders =
      [OpBuilder<
           (ins CArg<"::llzk::pod::InitializedRecords", "{}">:$initialValues),
           [{
        auto resultType = ::llzk::pod::PodType::fromInitialValues($_builder.getContext(), initialValues);
        build($_builder, $_state, resultType, initialValues);
      }]>,
       OpBuilder<(ins "::llzk::pod::PodType":$resultType,
           CArg<"::llzk::pod::InitializedRecords", "{}">:$initialValues)>,
       OpBuilder<(ins "::llzk::pod::PodType":$resultType,
           "::llvm::ArrayRef<::mlir::ValueRange>":$mapOperands,
           "::mlir::DenseI32ArrayAttr":$numDimsPerMap,
           CArg<"::llzk::pod::InitializedRecords", "{}">:$initialValues)>,
       OpBuilder<
           (ins "::llzk::pod::PodType":$resultType,
               "::llvm::ArrayRef<::mlir::ValueRange>":$mapOperands,
               "::llvm::ArrayRef<int32_t>":$numDimsPerMap,
               CArg<"::llzk::pod::InitializedRecords", "{}">:$initialValues),
           [{
      build($_builder, $_state, resultType, mapOperands,
         $_builder.getDenseI32ArrayAttr(numDimsPerMap), initialValues);
    }]>];
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    ::mlir::SmallVector<::llzk::pod::RecordValue> getInitializedRecordValues();
  }];
}

def LLZK_ReadPodOp : PODDialectOp<"read", [MemoryEffects<[MemRead]>]> {
  let summary = "reads the contents of a plain-old-data struct record";
  let arguments =
      (ins Arg<LLZK_PODType, "the struct to read from", [MemRead]>:$pod_ref,
          StrAttr:$record_name);
  let results = (outs AnyLLZKType:$result);
  let assemblyFormat = [{
    $pod_ref `[` custom<RecordName>($record_name) `]` `:` type($pod_ref) `,` type($result) attr-dict
  }];
  let hasVerifier = 1;
}

def LLZK_WritePodOp : PODDialectOp<"write", [MemoryEffects<[MemWrite]>]> {
  let summary = "writes content into a plain-old-data struct record";
  let arguments =
      (ins Arg<LLZK_PODType, "the struct to write into", [MemWrite]>:$pod_ref,
          StrAttr:$record_name, AnyLLZKType:$value);

  let assemblyFormat = [{
    $pod_ref `[` custom<RecordName>($record_name) `]` `=` $value `:` type($pod_ref) `,` type($value) attr-dict
  }];
  let hasVerifier = 1;
}

#endif // LLZK_POD_OPS
