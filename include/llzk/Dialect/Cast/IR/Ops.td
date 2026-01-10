//===-- Ops.td ---------------------------------------------*- tablegen -*-===//
// (header omitted for brevity — keep the file header from your existing Ops.td)

#ifndef LLZK_CAST_OPS
#define LLZK_CAST_OPS

include "llzk/Dialect/Shared/Types.td"
include "llzk/Dialect/Cast/IR/Dialect.td"
include "llzk/Dialect/Cast/IR/Attrs.td"
include "llzk/Dialect/Felt/IR/Types.td"

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"

def LLZK_IntToFeltOp : Op<CastDialect, "tofelt", [Pure]> {
  let summary = "convert an integer into a field element";
  let description = [{
    This operation converts a supported integer type value into a field element value.

    Example:

    ```llzk
    %0 = bool.cmp lt(%a, %b)
    %1 = cast.tofelt %0 {overflow = "trunc"} : i1
    ```
  }];

  // Require an explicit overflow semantics attribute. This attribute is required
  // and must be one of "trunc" or "sat".
  let arguments = (ins AnySignlessIntegerOrIndex:$value);
  let attributes = (ins OverflowSemanticsAttr:$overflow);
  let results = (outs LLZK_FeltType:$result);
  let assemblyFormat = [{ $value `:` type($value) attr-dict }];

  let hasVerifier = 1;
  let extraClassDeclaration = [{
    ::mlir::LogicalResult verify();
  }];
}

def LLZK_FeltToIndexOp : Op<CastDialect, "toindex", [Pure]> {
  let summary = "convert a field element into an index";
  let description = [{
    This operation converts a field element value into an index value to allow
    use as an array index or loop bound. In struct @constrain functions, the
    argument to this op is not allowed to be derived from a Signal struct.

    Example:
    ```llzk
    %0 = cast.toindex %a {overflow = "trunc"}
    %1 = array.read %b[%0]
    ```
  }];

  let arguments = (ins LLZK_FeltType:$value);
  let attributes = (ins OverflowSemanticsAttr:$overflow);
  let results = (outs Index:$result);
  let assemblyFormat = [{ $value attr-dict }];

  let hasVerifier = 1;
  let extraClassDeclaration = [{
    ::mlir::LogicalResult verify();
  }];
}

def LLZK_FeltToIntOp : Op<CastDialect, "toint", [Pure]> {
  let summary = "convert a field element into an integer";
  let description = [{
    This operation converts a field element value into an integer type value.

    Example:

    ```llzk
    %0 = cast.toint %a {overflow = "trunc"} : i32
    ```
  }];

  let arguments = (ins LLZK_FeltType:$value);
  let attributes = (ins OverflowSemanticsAttr:$overflow);
  let results = (outs AnySignlessInteger:$result);
  let assemblyFormat = [{ $value `:` type($result) attr-dict }];

  let hasVerifier = 1;
  let extraClassDeclaration = [{
    ::mlir::LogicalResult verify();
  }];
}
#endif // LLZK_CAST_OPS