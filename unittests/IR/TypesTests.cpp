#include "llzk/Dialect/LLZK/IR/Ops.h"

#include <gtest/gtest.h>

using namespace llzk;

class TypeTests : public ::testing::Test {
public:
  mlir::MLIRContext ctx;
  mlir::Builder bldr;

protected:
  TypeTests() : ctx(), bldr(&ctx) { ctx.loadDialect<llzk::LLZKDialect>(); }

  void SetUp() override { bldr = mlir::Builder(&ctx); }
};

#define TEST_F_WITH_DIAGNOSTIC(test_fixture, test_name, body_func, expected_diagnostic)            \
  TEST_F(test_fixture, test_name) {                                                                \
    std::string capturedDiagnostic;                                                                \
    ctx.getDiagEngine().registerHandler([&](mlir::Diagnostic &diag) {                              \
      llvm::raw_string_ostream stream(capturedDiagnostic);                                         \
      diag.print(stream);                                                                          \
      return mlir::success();                                                                      \
    });                                                                                            \
    (body_func);                                                                                   \
    EXPECT_EQ(capturedDiagnostic, expected_diagnostic);                                            \
  }

TEST_F(TypeTests, testCloneSuccessNewType) {
  auto tyBool = bldr.getI1Type();
  auto tyIndex = bldr.getIndexType();
  auto a = ArrayType::get(tyIndex, {2, 2});
  auto b = a.cloneWith(std::nullopt, tyBool);
  ASSERT_EQ(b.getElementType(), tyBool);
  ASSERT_EQ(b.getShape(), mlir::ArrayRef({2LL, 2LL}));
}

TEST_F(TypeTests, testCloneSuccessNewShape) {
  auto tyIndex = bldr.getIndexType();
  auto a = ArrayType::get(tyIndex, {2, 2});
  auto newShape = mlir::ArrayRef({2LL, 3LL, 2LL});
  auto b = a.cloneWith(std::make_optional(newShape), tyIndex);
  ASSERT_EQ(b.getElementType(), tyIndex);
  ASSERT_EQ(b.getShape(), newShape);
}

void testVerifyErrorEmptyShapeImpl(TypeTests *test) {
  auto tyIndex = test->bldr.getIndexType();
  auto a = ArrayType::get(tyIndex, {2, 2});
  std::vector<int64_t> newShapeVec;
  mlir::ArrayRef newShape(newShapeVec);
  auto b = a.cloneWith(std::make_optional(newShape), tyIndex);
  ASSERT_EQ(b, nullptr);
}

TEST_F_WITH_DIAGNOSTIC(
    TypeTests, testVerifyErrorEmptyShape, testVerifyErrorEmptyShapeImpl(this),
    "ArrayType::cloneWith() failed: array must have at least one dimension"
)
