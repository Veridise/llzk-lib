#include "zkir/Dialect/ZKIR/IR/Attrs.h"
#include "zkir/Dialect/ZKIR/IR/Types.h"

namespace zkir {

mlir::Type FeltConstAttr::getType() const { return FeltType::get(this->getContext()); }

} // namespace zkir
