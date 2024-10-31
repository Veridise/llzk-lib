#ifndef ZKIR_ZKIRPASSES_TD
#define ZKIR_ZKIRPASSES_TD

include "mlir/Pass/PassBase.td"

def HelloWorldPass : Pass<"zkir-hello-world", "::zkir::FuncOp"> {
  let summary = "A hello world pass";
  let description =
      [{This pass is an example demonstrating how to create a new MLIR Pass.}];

  let constructor = "zkir::createHelloWorldPass()";
  let dependentDialects = ["zkir::ZKIRDialect"];
}

#endif // ZKIR_ZKIRPASSES_TD
