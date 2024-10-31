#include "zkir/Dialect/InitDialects.h"
#include "zkir/Dialect/ZKIR/IR/Dialect.h" // IWYU pragma: keep
#include <mlir/Dialect/Index/IR/IndexDialect.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/DialectRegistry.h>

namespace zkir {
void registerAllDialects(mlir::DialectRegistry &registry) {
  registry.insert<zkir::ZKIRDialect, mlir::index::IndexDialect, mlir::scf::SCFDialect>();
}
} // namespace zkir
