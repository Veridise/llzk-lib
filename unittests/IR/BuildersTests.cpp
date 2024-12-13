#include <llzk/Dialect/LLZK/IR/Builders.h>
#include <gtest/gtest.h>

/* Tests for the ModuleBuilder */

using namespace llzk;

TEST(LLZKTestModuleBuilderTests, testModuleOpCreation) {
  ModuleBuilder builder;

  ASSERT_NE(builder.getMod(), nullptr);
}

TEST(LLZKTestModuleBuilderTests, testStructDefInsertion) {
  ModuleBuilder builder;

  auto structDef = builder.insertEmptyStruct("structOne");
  ASSERT_EQ(builder.getStruct("structOne"), structDef);
}

TEST(LLZKTestModuleBuilderTests, testFnInsertion) {
  ModuleBuilder builder;

  auto structOp = builder.insertFullStruct("structOne");

  auto computeFn = builder.getComputeFn(&structOp);
  ASSERT_EQ(computeFn.getBody().getArguments().size(), 0);

  auto constrainFn = builder.getConstrainFn(&structOp);
  ASSERT_EQ(constrainFn.getBody().getArguments().size(), 1);
}

TEST(LLZKTestModuleBuilderTests, testReachabilitySimple) {
  ModuleBuilder builder;

  auto a = builder.insertComputeOnlyStruct("structA");
  auto b = builder.insertComputeOnlyStruct("structB");
  builder.insertComputeCall(&a, &b);

  ASSERT_TRUE(builder.computeReachable(&a, &b));
  ASSERT_FALSE(builder.computeReachable(&b, &a));
}

TEST(LLZKTestModuleBuilderTests, testReachabilityTransitive) {
  ModuleBuilder builder;

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

TEST(LLZKTestModuleBuilderTests, testReachabilityComputeAndConstrain) {
  ModuleBuilder builder;

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

TEST(LLZKTestModuleBuilderTests, testConstruction) {
  ModuleBuilder builder;

  auto a = builder.insertConstrainOnlyStruct("structA");
  auto b = builder.insertConstrainOnlyStruct("structB");
  builder.insertConstrainOnlyStruct("structC");
  builder.insertConstrainCall(&a, &b);

  size_t numStructs = 0;
  for (auto s : builder.getMod().getOps<llzk::StructDefOp>()) {
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