#ifndef ZKIR_DIALECT
#define ZKIR_DIALECT

include "mlir/IR/DialectBase.td"

def ZKIRDialect : Dialect {
  let name = "zkir";
  let summary =
      "Veridise's intermediate representation for zero knowledge languages";
  let description = [{
    This dialect contains types and operations for zero knowledge languages.
  }];

  let cppNamespace = "zkir";

  let useDefaultTypePrinterParser = true;
  // let useDefaultAttributePrinterParser = true;
  // let hasConstantMaterializer = 1;
}

#endif // ZKIR_DIALECT
