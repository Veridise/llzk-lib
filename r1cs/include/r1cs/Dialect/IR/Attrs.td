
#ifndef R1CS_ATTRS
#define R1CS_ATTRS

include "mlir/IR/AttrTypeBase.td"
include "r1cs/Dialect/IR/Dialect.td"

def FeltAttr : AttrDef<R1CSDialect, "Felt"> {
  let summary = "Field element represented as an arbitrary precision integer";
  let cppClassName = "FeltAttr";
  let mnemonic = "felt";
  let parameters = (ins "mlir::IntegerAttr":$value);

  let assemblyFormat = "$value";

  let builders = [
    AttrBuilder<(ins "const llvm::APInt &":$v), [{
      return Base::get(context, mlir::IntegerAttr::get(mlir::IntegerType::get(context, v.getBitWidth()), v));
    }]>,
    AttrBuilder<(ins "const llvm::APSInt &":$v), [{
      return Base::get(context, mlir::IntegerAttr::get(mlir::IntegerType::get(context, v.getBitWidth()), v));
    }]>
  ];
}

#endif // R1CS_ATTRS
