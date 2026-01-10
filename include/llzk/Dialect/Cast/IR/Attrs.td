//===-- Attrs.td --------------------------------------------*- tablegen -*-===//
//
// Attrs used by the Cast dialect.
//
// This file defines an attribute that encodes overflow semantics for narrowing
// casts. The attribute is named `OverflowSemanticsAttr` and supports two values:
// - "trunc" (truncate, i.e., wrap/truncate)
// - "sat"   (saturate)
//
// The Ops.td file requires this attribute on all cast ops that need explicit
// overflow semantics.
//
//===----------------------------------------------------------------------===//

include "mlir/IR/OpBase.td"

// A small enum-like attribute to encode overflow semantics.
def OverflowSemanticsAttr : EnumAttr<"llzk.cast.overflow"> {
  let summary = "Overflow semantics for cast operations.";
  let description = [{
    Indicates how to handle overflow when converting values that do not fit in
    the target representation. Valid values are "trunc" (truncate) and "sat" (saturate).
  }];

  let elements = [
    { let name = "trunc"; let value = 0; let cppEnumName = "Trunc"; },
    { let name = "sat"; let value = 1; let cppEnumName = "Sat"; }
  ];
}