#include <llzk/Dialect/LLZK/IR/Builders.h>

#include <gtest/gtest.h>

/* Tests for the ModuleBuilder */

using namespace llzk;

class ModuleBuilderTests : public ::testing::Test {
protected:
  mlir::MLIRContext context;
  ModuleBuilder builder;

  ModuleBuilderTests() : context(), builder(&context) { context.loadDialect<llzk::LLZKDialect>(); }

  void SetUp() override {
    // Create a new builder for each test.
    builder = ModuleBuilder(&context);
  }

  void TearDown() override { builder.getRootModule().erase(); }
};

TEST_F(ModuleBuilderTests, testModuleOpCreation) { ASSERT_NE(builder.getRootModule(), nullptr); }

TEST_F(ModuleBuilderTests, testStructDefInsertion) {
  auto structDef = builder.insertEmptyStruct("structOne");
  ASSERT_EQ(builder.getStruct("structOne"), structDef);
}

TEST_F(ModuleBuilderTests, testFnInsertion) {
  auto structOp = builder.insertFullStruct("structOne");

  auto computeFn = builder.getComputeFn(&structOp);
  ASSERT_EQ(computeFn.getBody().getArguments().size(), 0);

  auto constrainFn = builder.getConstrainFn(&structOp);
  ASSERT_EQ(constrainFn.getBody().getArguments().size(), 1);
}

TEST_F(ModuleBuilderTests, testReachabilitySimple) {
  auto a = builder.insertComputeOnlyStruct("structA");
  auto b = builder.insertComputeOnlyStruct("structB");
  builder.insertComputeCall(&a, &b);

  ASSERT_TRUE(builder.computeReachable(&a, &b));
  ASSERT_FALSE(builder.computeReachable(&b, &a));
}

TEST_F(ModuleBuilderTests, testReachabilityTransitive) {
  auto a = builder.insertComputeOnlyStruct("structA");
  auto b = builder.insertComputeOnlyStruct("structB");
  auto c = builder.insertComputeOnlyStruct("structC");
  builder.insertComputeCall(&a, &b);
  builder.insertComputeCall(&b, &c);

  ASSERT_TRUE(builder.computeReachable(&a, &b));
  ASSERT_TRUE(builder.computeReachable(&b, &c));
  ASSERT_TRUE(builder.computeReachable(&a, &c));
  ASSERT_FALSE(builder.computeReachable(&b, &a));
  ASSERT_FALSE(builder.computeReachable(&c, &a));
  ASSERT_TRUE(builder.computeReachable(&a, &a));
}

TEST_F(ModuleBuilderTests, testReachabilityComputeAndConstrain) {
  auto a = builder.insertFullStruct("structA");
  auto b = builder.insertComputeOnlyStruct("structB");
  auto c = builder.insertConstrainOnlyStruct("structC");
  builder.insertComputeCall(&a, &b);
  builder.insertConstrainCall(&a, &c);

  ASSERT_TRUE(builder.computeReachable(&a, &b));
  ASSERT_TRUE(builder.constrainReachable(&a, &c));
  ASSERT_FALSE(builder.constrainReachable(&a, &b));
  ASSERT_FALSE(builder.computeReachable(&a, &c));
}

TEST_F(ModuleBuilderTests, testConstruction) {
  auto a = builder.insertConstrainOnlyStruct("structA");
  auto b = builder.insertConstrainOnlyStruct("structB");
  builder.insertConstrainOnlyStruct("structC");
  builder.insertConstrainCall(&a, &b);

  size_t numStructs = 0;
  for (auto s : builder.getRootModule().getOps<llzk::StructDefOp>()) {
    numStructs++;
    size_t numFn = 0;
    for (auto fn : s.getOps<llzk::FuncOp>()) {
      numFn++;
      ASSERT_EQ(fn.getName(), llzk::FUNC_NAME_CONSTRAIN);
    }
    ASSERT_EQ(numFn, 1);
  }
  ASSERT_EQ(numStructs, 3);

  auto aFn = builder.getConstrainFn(&a);
  size_t numOps = 0;
  for (auto &_ : aFn.getOps()) {
    numOps++;
  }
  ASSERT_EQ(numOps, 2);
}