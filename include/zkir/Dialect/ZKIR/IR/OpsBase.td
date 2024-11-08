#ifndef ZKIR_OPS_BASE
#define ZKIR_OPS_BASE

include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"

include "zkir/Dialect/ZKIR/IR/Types.td"
include "zkir/Dialect/ZKIR/IR/OpInterfaces.td"

class ZKIR_Op<string mnemonic, list<Trait> traits = []>
    : Op<ZKIRDialect, mnemonic, traits>;

class ZKIR_FieldRefOp<string mnemonic, list<Trait> traits = []>
    : ZKIR_Op<mnemonic,
              traits#[DeclareOpInterfaceMethods<SymbolUserOpInterface>,
                      DeclareOpInterfaceMethods<FieldRefOpInterface>]> {
  let extraClassDeclaration = [{
    /// Gets the `field` definition for the field referenced in this op.
    ::mlir::FailureOr<FieldDefOp> getFieldDefOp(::mlir::SymbolTableCollection &symbolTable);
  }];
}

class ZKIR_BinaryFeltOp<string mnemonic, list<Trait> traits = []>
    : ZKIR_Op<mnemonic, traits#[SameOperandsAndResultType, Pure]> {

  let arguments = (ins ZKIR_FeltType:$lhs, ZKIR_FeltType:$rhs);
  let results = (outs ZKIR_FeltType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict";
}

class ZKIR_UnaryFeltOp<string mnemonic, list<Trait> traits = []>
    : ZKIR_Op<mnemonic, traits#[SameOperandsAndResultType, Pure]> {

  let arguments = (ins ZKIR_FeltType:$operand);
  let results = (outs ZKIR_FeltType:$result);
  let assemblyFormat = "$operand attr-dict";
}

//===----------------------------------------------------------------------===//
// Custom traits that can be applied to Ops
//===----------------------------------------------------------------------===//

def ComputeOnly : NativeOpTrait<"ComputeOnly">, StructuralOpTrait {
  string cppNamespace = "::zkir";
}

def InStruct : NativeOpTrait<"InStruct">, StructuralOpTrait {
  string cppNamespace = "::zkir";
}

class InStructFunctionNamed<string name>
    : ParamNativeOpTrait<"InStructFunctionNamed", name>, StructuralOpTrait {
  string cppNamespace = "::zkir";
}

#endif // ZKIR_OPS_BASE
