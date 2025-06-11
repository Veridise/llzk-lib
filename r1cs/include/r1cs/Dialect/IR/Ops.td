//===- Ops.td ------------------------------------------------------------===//

#ifndef R1CS_OPS
#define R1CS_OPS
include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "r1cs/Dialect/IR/Types.td"
include "r1cs/Dialect/IR/Attrs.td"


class R1CS_Op<string mnemonic, list<Trait> traits = []> :
    Op<R1CSDialect, mnemonic, traits>;

def R1CS_CircuitOp : R1CS_Op<"circuit", [
  Symbol, IsolatedFromAbove, SingleBlock, NoTerminator
]> {
  let summary = "Circuit operation with named inputs and outputs";

  let arguments = (ins 
    SymbolNameAttr:$sym_name, 
    Variadic<R1CS_SignalType>:$inputs
  );

  let results = (outs 
    Variadic<R1CS_SignalType>:$outputs
  );

  let regions = (region SizedRegion<1>:$body);

  let hasCustomAssemblyFormat = 1;
}

def R1CS_ReturnOp : R1CS_Op<"return", [Terminator]> {
  let summary = "Returns signal values from a circuit";
  let arguments = (ins Variadic<R1CS_SignalType>:$results);

  let assemblyFormat = [{
    $results `:` type($results) attr-dict
  }];
}

def SignalDefOp : R1CS_Op<"def", [Pure]> {
  let summary = "Define a signal and its label ID";
  let arguments = (ins UI64Attr:$label);
  let results = (outs R1CS_SignalType:$out);
  let assemblyFormat = [{
    $label `:` type($out) attr-dict
  }];
}

def R1CS_ToLinearOp : R1CS_Op<"to_linear", [Pure]> {
  let summary = "Casts a signal to a linear expression";
  let arguments = (ins R1CS_SignalType:$input);
  let results = (outs R1CS_LinearType:$output);
  let assemblyFormat = "$input attr-dict `:` type($input) `to` type($output)";
}

def R1CS_ConstOp : R1CS_Op<"const", [Pure, Commutative]> {
  let summary = "Create a constant linear value";
  let arguments = (ins FeltAttr:$value);
  let results = (outs R1CS_LinearType:$out);
  let assemblyFormat = "$value attr-dict `:` type($out)";
}

def R1CS_AddOp : R1CS_Op<"add", [Pure, Commutative]> {
  let summary = "Add two linear expressions";
  let arguments = (ins R1CS_LinearType:$lhs, R1CS_LinearType:$rhs);
  let results = (outs R1CS_LinearType:$sum);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($sum)";
}

def R1CS_MulConstOp : R1CS_Op<"mul_const", [Pure]> {
  let summary = "Scale a linear expression by a constant";
  let arguments = (ins R1CS_LinearType:$input, FeltAttr:$constValue);
  let results = (outs R1CS_LinearType:$scaled);
  let assemblyFormat = "$input `,` $constValue attr-dict `:` type($scaled)";
}

def R1CS_NegOp : R1CS_Op<"neg", [Pure]> {
  let summary = "Negate a linear expression";
  let arguments = (ins R1CS_LinearType:$input);
  let results = (outs R1CS_LinearType:$neg);
  let assemblyFormat = "$input attr-dict `:` type($neg)";
}

def R1CS_AssignOp : R1CS_Op<"assign", [Pure]> {
  let summary = "Assign a linear expression to a new signal";
  let arguments = (ins R1CS_LinearType:$input);
  let results = (outs R1CS_SignalType:$out);
  let assemblyFormat = "$input attr-dict `:` type($out)";
}

def R1CS_ConstrainOp : R1CS_Op<"constrain", [Pure]> {
  let summary = "Enforce a * b = c R1CS constraint";
  let arguments = (ins R1CS_LinearType:$a, R1CS_LinearType:$b, R1CS_LinearType:$c);
  let assemblyFormat = "$a `,` $b `,` $c attr-dict `:` type($a)";
}

#endif // R1CS_OPS