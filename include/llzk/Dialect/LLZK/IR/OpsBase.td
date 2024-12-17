#ifndef LLZK_OPS_BASE
#define LLZK_OPS_BASE

include "llzk/Dialect/LLZK/IR/Types.td"
include "llzk/Dialect/LLZK/IR/OpInterfaces.td"

include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"

class LLZK_Op<string mnemonic, list<Trait> traits = []>
    : Op<LLZKDialect, mnemonic, traits>;

class LLZK_FieldRefOp<string mnemonic, list<Trait> traits = []>
    : LLZK_Op<mnemonic,
              traits#[DeclareOpInterfaceMethods<SymbolUserOpInterface>,
                      DeclareOpInterfaceMethods<FieldRefOpInterface>]> {
  let extraClassDeclaration = [{
    /// Gets the `field` definition for the field referenced in this op.
    ::mlir::FailureOr<SymbolLookupResult<FieldDefOp>> getFieldDefOp(::mlir::SymbolTableCollection &symbolTable);
  }];
}

class LLZK_BinaryFeltOp<string mnemonic, list<Trait> traits = []>
    : LLZK_Op<mnemonic, traits#[SameOperandsAndResultType, Pure]> {

  let arguments = (ins LLZK_FeltType:$lhs, LLZK_FeltType:$rhs);
  let results = (outs LLZK_FeltType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict";
}

class LLZK_UnaryFeltOp<string mnemonic, list<Trait> traits = []>
    : LLZK_Op<mnemonic, traits#[SameOperandsAndResultType, Pure]> {

  let arguments = (ins LLZK_FeltType:$operand);
  let results = (outs LLZK_FeltType:$result);
  let assemblyFormat = "$operand attr-dict";
}

class LLZK_BinaryBoolOp<string mnemonic, list<Trait> traits = []>
    : LLZK_Op<mnemonic, traits#[SameOperandsAndResultType, Pure]> {

  let arguments = (ins I1:$lhs, I1:$rhs);
  let results = (outs I1:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict";
}

class LLZK_UnaryBoolOp<string mnemonic, list<Trait> traits = []>
    : LLZK_Op<mnemonic, traits#[SameOperandsAndResultType, Pure]> {

  let arguments = (ins I1:$operand);
  let results = (outs I1:$result);
  let assemblyFormat = "$operand attr-dict";
}

//===----------------------------------------------------------------------===//
// Custom traits that can be applied to Ops
//===----------------------------------------------------------------------===//

def ComputeOnly : NativeOpTrait<"ComputeOnly">, StructuralOpTrait {
  string cppNamespace = "::llzk";
}

def InStruct : NativeOpTrait<"InStruct">, StructuralOpTrait {
  string cppNamespace = "::llzk";
}

class InStructFunctionNamed<string name>
    : ParamNativeOpTrait<"InStructFunctionNamed", name>, StructuralOpTrait {
  string cppNamespace = "::llzk";
}

// Same as TypesMatchWith but if `rhsArg` is Variadic and empty returns success.
class VariadicTypesMatchWith<string summary, string lhsArg, string rhsArg,
                             string transform,
                             string comparator = "std::equal_to<>()">
    : TypesMatchWith<
          summary, lhsArg, rhsArg, transform,
          "get"#snakeCaseToCamelCase<rhsArg>.ret#"().empty() || "#comparator>;

#endif // LLZK_OPS_BASE
