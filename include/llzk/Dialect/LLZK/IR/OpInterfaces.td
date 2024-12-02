#ifndef LLZK_OP_INTERFACES
#define LLZK_OP_INTERFACES

include "llzk/Dialect/LLZK/IR/Types.td"

def FieldRefOpInterface : OpInterface<"FieldRefOpInterface"> {
  let description = [{
    Common interface for operations that read or write from a field.
  }];
  let cppNamespace = "::llzk";

  let methods = [
      // Requires implementors to have a '$component' argument.
      InterfaceMethod<
          [{Gets the SSA value with the target component from the FieldRefOp.}],
          "::mlir::Value", "getComponent", (ins)>,
      // Requires implementors to have a '$field_name' argument.
      InterfaceMethod<[{Gets the field name attribute from the FieldRefOp.}],
                      "::llvm::StringRef", "getFieldName", (ins)>,
      // Requires the '$component' argument type to be LLZK_StructType.
      InterfaceMethod<[{Gets the struct type of the target component.}],
                      "::llzk::StructType", "getStructType", (ins),
                      /*methodBody=*/[{
        mlir::Type t = $_op.getComponent().getType();
        assert(t.isa<StructType>());
        return t.template cast<StructType>();
      }]>];
}

#endif // LLZK_OP_INTERFACES
