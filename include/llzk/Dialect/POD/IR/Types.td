//===-- Types.td -------------------------------------------*- tablegen -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_POD_TYPES
#define LLZK_POD_TYPES

include "llzk/Dialect/Shared/Types.td"
include "llzk/Dialect/POD/IR/Dialect.td"
include "llzk/Dialect/POD/IR/Attrs.td"

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/Interfaces/MemorySlotInterfaces.td"
include "mlir/IR/BuiltinTypeInterfaces.td"

class PODDialectType<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<PODDialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def LLZK_PODType : PODDialectType<"Pod", "type"> {
  let summary = "plain-old-data struct";
  let description = [{
    Contains a list of `RecordAttr` defining the fields of the struct. 
    It is possible to have a struct with no fields and the field order matters.

    Two `PODType` types unify if they have the same field names, defined in the 
    same order, and if the types of each field unify.

    This type can be used in the `struct.field` op, and it can be declared as a 
    column, transferring that annotations to the inner types that can also be columns.

    ```llzk
    // Type with no records.
    !pod.type

    // Type with two records of felt type.
    !pod.type<[@a: !felt.type, @b: !felt.type]>
    ```
  }];

  let parameters =
      (ins ArrayRefParameter<"::llzk::pod::RecordAttr",
                             "List of records in the struct">:$records);

  let assemblyFormat = "`<` $records `>`";
}

#endif // LLZK_POD_TYPES
