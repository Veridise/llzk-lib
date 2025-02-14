#include "llzk/Dialect/LLZK/Util/SymbolHelper.h"

#include <mlir/IR/BuiltinAttributes.h>

#include <gtest/gtest.h>

using namespace llzk;
using namespace mlir;

TEST(SymbolHelperTests, test_getFlatSymbolRefAttr) {
  MLIRContext ctx;
  FlatSymbolRefAttr attr = getFlatSymbolRefAttr(&ctx, "name");
  ASSERT_EQ(attr.getValue(), "name");
}
