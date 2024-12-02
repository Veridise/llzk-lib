#include "llzk/Dialect/LLZK/IR/Attrs.h"
#include "llzk/Dialect/LLZK/IR/Types.h"

namespace llzk {

mlir::Type FeltConstAttr::getType() const { return FeltType::get(this->getContext()); }

} // namespace llzk
