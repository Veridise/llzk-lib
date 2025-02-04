#include "llzk/Dialect/LLZK/IR/Ops.h"

#include <mlir/Dialect/Index/IR/IndexOps.h>

#include <gtest/gtest.h>

using namespace llzk;
using namespace mlir;

class AffineMapInstantiationTests : public ::testing::Test {
public:
  MLIRContext ctx;
  Location loc;

protected:
  AffineMapInstantiationTests() : ctx(), loc(UnknownLoc::get(&ctx)) {
    ctx.loadDialect<llzk::LLZKDialect>();
  }
};

TEST_F(AffineMapInstantiationTests, testElementInitEmpty) {
  OpBuilder bldr(&ctx);
  ArrayType arrTy = ArrayType::get(bldr.getIndexType(), {2, 2}); // !llzk.array<2,2 x index>
  CreateArrayOp op = bldr.create<CreateArrayOp>(loc, arrTy);
  ASSERT_TRUE(succeeded(op.verifyInvariants()));
}

TEST_F(AffineMapInstantiationTests, testElementInitNonEmpty) {
  OpBuilder bldr(&ctx);
  ArrayType arrTy = ArrayType::get(bldr.getIndexType(), {2}); // !llzk.array<2 x index>
  auto v1 = bldr.create<index::ConstantOp>(loc, 766);
  auto v2 = bldr.create<index::ConstantOp>(loc, 562);
  CreateArrayOp op = bldr.create<CreateArrayOp>(loc, arrTy, ValueRange({v1, v2}));
  ASSERT_TRUE(succeeded(op.verifyInvariants()));
}

TEST_F(AffineMapInstantiationTests, testElementInitTooFew) {
  EXPECT_DEATH(
      {
        OpBuilder bldr(&ctx);
        ArrayType arrTy = ArrayType::get(bldr.getIndexType(), {5}); // !llzk.array<5 x index>
        auto v1 = bldr.create<index::ConstantOp>(loc, 766);
        auto v2 = bldr.create<index::ConstantOp>(loc, 562);
        CreateArrayOp op = bldr.create<CreateArrayOp>(loc, arrTy, ValueRange({v1, v2}));
        assert(succeeded(op.verifyInvariants()));
      },
      "error: 'llzk.new_array' op failed to verify that operand types match result type"
  );
}

TEST_F(AffineMapInstantiationTests, testElementInitTooMany) {
  EXPECT_DEATH(
      {
        OpBuilder bldr(&ctx);
        ArrayType arrTy = ArrayType::get(bldr.getIndexType(), {1}); // !llzk.array<1 x index>
        auto v1 = bldr.create<index::ConstantOp>(loc, 766);
        auto v2 = bldr.create<index::ConstantOp>(loc, 562);
        CreateArrayOp op = bldr.create<CreateArrayOp>(loc, arrTy, ValueRange({v1, v2}));
        assert(succeeded(op.verifyInvariants()));
      },
      "error: 'llzk.new_array' op failed to verify that operand types match result type"
  );
}

TEST_F(AffineMapInstantiationTests, testElementInitWithAffineMapType) {
  EXPECT_DEATH(
      {
        OpBuilder bldr(&ctx);
        AffineMapAttr m = AffineMapAttr::get(bldr.getDimIdentityMap()); // (d0) -> (d0)
        ArrayType arrTy = ArrayType::get(bldr.getIndexType(), {m});     // !llzk.array<#m x index>
        CreateArrayOp op = bldr.create<CreateArrayOp>(loc, arrTy);
        assert(succeeded(op.verifyInvariants()));
      },
      "error: 'llzk.new_array' op map instantiation group count \\(0\\) does not match the number "
      "of affine map instantiations \\(1\\) required by the type"
  );
}

