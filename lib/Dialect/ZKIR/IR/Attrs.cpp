#include "zkir/Dialect/ZKIR/IR/Attrs.h"
#include "zkir/Dialect/ZKIR/IR/Types.h"

namespace zkir {

mlir::ParseResult parseAPInt(mlir::AsmParser &parser, llvm::APInt &value) {
  auto loc = parser.getCurrentLocation();
  llvm::APInt val;
  auto result = parser.parseOptionalInteger(val);
  if (!result.has_value() || *result) {
    return parser.emitError(loc, "expected integer value");
  }
  value = std::move(val);
  return mlir::success();
}

void printAPInt(mlir::AsmPrinter &printer, const llvm::APInt &value) {
  llvm::SmallString<32> buf;
  value.toStringSigned(buf);
  printer << buf;
}

mlir::Type FeltConstAttr::getType() const { return FeltType::get(this->getContext()); }

} // namespace zkir
