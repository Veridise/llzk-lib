//===-- Attrs.td -------------------------------------------*- tablegen -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef POD_ATTRS
#define POD_ATTRS

include "llzk/Dialect/POD/IR/Dialect.td"

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/EnumAttr.td"

class PODDialectAttr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<PODDialect, name, traits> {
  let mnemonic = attrMnemonic;
}

def LLZK_RecordAttr : PODDialectAttr<"Record", "record"> {
  let summary = "A named entry in a plain-old-data struct";
  let parameters = (ins AttrParameter<"::mlir::FlatSymbolRefAttr",
                                      "Name of the record entry.">:$name,
      AttrParameter<"::mlir::TypeAttr", "Type of the record entry.">:$type);

  let assemblyFormat = " $name `:` $type ";
}

#endif // POD_ATTRS
