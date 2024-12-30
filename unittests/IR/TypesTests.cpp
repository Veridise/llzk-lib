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

#define TEST_F_WITH_DIAGNOSTIC(test_fixture, test_name, body_func, expected_diagnostic)            \
  TEST_F(test_fixture, test_name) {                                                                \
    std::string capturedDiagnostic;                                                                \
    auto h = ctx.getDiagEngine().registerHandler([&](Diagnostic &diag) {                           \
      llvm::raw_string_ostream stream(capturedDiagnostic);                                         \
      diag.print(stream);                                                                          \
      return success();                                                                            \
    });                                                                                            \
    (body_func);                                                                                   \
    EXPECT_EQ(capturedDiagnostic, expected_diagnostic);                                            \
    ctx.getDiagEngine().eraseHandler(h);                                                           \
  }

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

void testVerifyErrorEmptyShapeImpl(TypeTests *test) {
  IndexType tyIndex = IndexType::get(&test->ctx);
  ArrayType a = ArrayType::get(tyIndex, {2, 2});
  std::vector<int64_t> newShapeVec;
  ArrayRef newShape(newShapeVec);
  ArrayType b = a.cloneWith(std::make_optional(newShape), tyIndex);
  ASSERT_EQ(b, nullptr);
}

TEST_F_WITH_DIAGNOSTIC(
    TypeTests, testVerifyErrorEmptyShape, testVerifyErrorEmptyShapeImpl(this),
    "ArrayType::cloneWith() failed: array must have at least one dimension"
)
