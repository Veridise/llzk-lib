#include "llzk/Dialect/InitDialects.h"
#include "llzk/Dialect/LLZK/IR/Dialect.h" // IWYU pragma: keep

#include <mlir/Dialect/Index/IR/IndexDialect.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/DialectRegistry.h>

namespace llzk {
void registerAllDialects(mlir::DialectRegistry &registry) {
  registry.insert<llzk::LLZKDialect, mlir::index::IndexDialect, mlir::scf::SCFDialect>();
}
} // namespace llzk
