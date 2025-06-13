// tools/r1cs-opt/r1cs-opt.cpp

#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "r1cs/Dialect/IR/Dialect.h"
#include "r1cs/InitAllDialects.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  r1cs::registerAllDialects(registry);

  return mlir::asMainReturnCode(mlir::MlirOptMain(argc, argv, "R1CS Optimizer\n", registry));
}
