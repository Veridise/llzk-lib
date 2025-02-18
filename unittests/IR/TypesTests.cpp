#include "llzk/Dialect/LLZK/IR/Ops.h"

#include <gtest/gtest.h>

using namespace llzk;
using namespace mlir;

class TypeTests : public ::testing::Test {
public:
  MLIRContext ctx;

protected:
  TypeTests() : ctx() { ctx.loadDialect<llzk::LLZKDialect>(); }
};

TEST_F(TypeTests, testCloneSuccessNewType) {
  IntegerType tyBool = IntegerType::get(&ctx, 1);
  IndexType tyIndex = IndexType::get(&ctx);
  ArrayType a = ArrayType::get(tyIndex, {2, 2});
  ArrayType b = a.cloneWith(std::nullopt, tyBool);
  ASSERT_EQ(b.getElementType(), tyBool);
  ASSERT_EQ(b.getShape(), ArrayRef(std::vector<int64_t>({2, 2})));
}

TEST_F(TypeTests, testCloneSuccessNewShape) {
  IndexType tyIndex = IndexType::get(&ctx);
  ArrayType a = ArrayType::get(tyIndex, {2, 2});
  std::vector<int64_t> newShapeVec({2, 3, 2});
  ArrayRef newShape(newShapeVec);
  ArrayType b = a.cloneWith(std::make_optional(newShape), tyIndex);
  ASSERT_EQ(b.getElementType(), tyIndex);
  ASSERT_EQ(b.getShape(), newShape);
}

TEST_F(TypeTests, testCloneWithEmptyShapeError) {
  EXPECT_DEATH(
      {
        IndexType tyIndex = IndexType::get(&ctx);
        ArrayType a = ArrayType::get(tyIndex, {2, 2});
        std::vector<int64_t> newShapeVec;
        ArrayRef newShape(newShapeVec);
        a.cloneWith(std::make_optional(newShape), tyIndex);
      },
      "error: array must have at least one dimension"
  );
}

TEST_F(TypeTests, testGetWithAttributeEmptyShapeError) {
  EXPECT_DEATH(
      {
        IndexType tyIndex = IndexType::get(&ctx);
        std::vector<Attribute> newDimsVec;
        ArrayRef<Attribute> dimensionSizes(newDimsVec);
        ArrayType::get(tyIndex, dimensionSizes);
      },
      "error: array must have at least one dimension"
  );
}

TEST_F(TypeTests, testGetWithAttributeWrongAttrKindError) {
  EXPECT_DEATH(
      {
        IndexType tyIndex = IndexType::get(&ctx);
        std::vector<Attribute> newDimsVec = {UnitAttr::get(&ctx)};
        ArrayRef<Attribute> dimensionSizes(newDimsVec);
        ArrayType::get(tyIndex, dimensionSizes);
      },
      "error: Array dimension must be one of .* but found 'builtin.unit'"
  );
}

TEST_F(TypeTests, testBriefString) {
  OpBuilder bldr(&ctx);
  EXPECT_EQ("b", shortString(bldr.getIntegerType(1)));
  EXPECT_EQ("i", shortString(bldr.getIndexType()));
  EXPECT_EQ("!v<@A>", shortString(TypeVarType::get(FlatSymbolRefAttr::get(&ctx, "A"))));
  EXPECT_EQ(
      "!a<b:4,235,123>",
      shortString(ArrayType::get(bldr.getIntegerType(1), ArrayRef<int64_t> {4, 235, 123}))
  );
  EXPECT_EQ("!s<@S1:>", shortString(StructType::get(FlatSymbolRefAttr::get(&ctx, "S1"))));
  EXPECT_EQ(
      "!s<@S1:43>",
      shortString(StructType::get(
          FlatSymbolRefAttr::get(&ctx, "S1"),
          ArrayAttr::get(&ctx, ArrayRef<Attribute> {bldr.getIntegerAttr(bldr.getIndexType(), 43)})
      ))
  );
  {
    auto innerStruct = StructType::get(
        FlatSymbolRefAttr::get(&ctx, "S1"),
        ArrayAttr::get(&ctx, ArrayRef<Attribute> {bldr.getIntegerAttr(bldr.getIndexType(), 43)})
    );
    auto params = ArrayAttr::get(
        &ctx,
        ArrayRef<Attribute> {
            bldr.getIntegerAttr(bldr.getIndexType(), 43), FlatSymbolRefAttr::get(&ctx, "ParamName"),
            TypeAttr::get(ArrayType::get(FeltType::get(&ctx), ArrayRef<int64_t> {3, 5, 1, 5, 7})),
            TypeAttr::get(innerStruct), AffineMapAttr::get(bldr.getDimIdentityMap())
        }
    );
    EXPECT_EQ(
        "!s<@Top:43,@ParamName,!a<f:3,5,1,5,7>,!s<@S1:43>,!m<(d0)->(d0)>>",
        shortString(StructType::get(FlatSymbolRefAttr::get(&ctx, "Top"), params))
    );
  }
}
