#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <mlir/IR/BuiltinOps.h>
#include <llzk/Dialect/LLZK/IR/Ops.h>

#include <llzk/Dialect/LLZK/Analysis/CallGraph.h>

#include "OpBuilders.h"


TEST(CallGraphTests, constructorTest) {
  LLZKTestModuleBuilder builder;
  builder.insertFullStruct("A");

  ASSERT_NO_THROW(llzk::CallGraph(builder.getMod()));
}

TEST(CallGraphTests, printTest) {
  LLZKTestModuleBuilder builder;
  builder.insertFullStruct("A");

  std::string s;
  llvm::raw_string_ostream sstream(s);

  llzk::CallGraph cgraph(builder.getMod());
  cgraph.print(sstream);

  ASSERT_FALSE(sstream.str().empty());
}


TEST(CallGraphTests, numFnTest) {
  LLZKTestModuleBuilder builder;
  builder.insertFullStruct("A");

  llzk::CallGraph cgraph(builder.getMod());

  // Size also include "nullptr" function, so it is number of real functions + 1
  ASSERT_EQ(cgraph.size(), 3);
}

TEST(CallGraphTests, reachabilityTest) {
  LLZKTestModuleBuilder builder;
  auto aOp = builder.insertFullStruct("A");
  auto bOp = builder.insertFullStruct("B");
  auto cOp = builder.insertFullStruct("C");

  builder.insertComputeCall(&aOp, &bOp);
  builder.insertComputeCall(&bOp, &cOp);
  builder.insertConstrainCall(&bOp, &aOp);
  builder.insertConstrainCall(&cOp, &aOp);

  llzk::CallGraph cgraph(builder.getMod());

  auto aComp = builder.getComputeFn(&aOp), bComp = builder.getComputeFn(&bOp), cComp = builder.getComputeFn(&cOp);
  auto aCons = builder.getConstrainFn(&aOp), bCons = builder.getConstrainFn(&bOp), cCons = builder.getConstrainFn(&cOp);

  ASSERT_TRUE(cgraph.isReachable(aComp, bComp));
  ASSERT_TRUE(cgraph.isReachable(bComp, cComp));
  ASSERT_TRUE(cgraph.isReachable(aComp, cComp));
  ASSERT_TRUE(cgraph.isReachable(aComp, cComp)); // test caching
  ASSERT_TRUE(cgraph.isReachable(bCons, aCons));
  ASSERT_TRUE(cgraph.isReachable(cCons, aCons));

  ASSERT_FALSE(cgraph.isReachable(cComp, bComp));
  ASSERT_FALSE(cgraph.isReachable(cComp, aCons));
  ASSERT_FALSE(cgraph.isReachable(aCons, bCons));
}

TEST(CallGraphTests, analysisConstructor) {
  LLZKTestModuleBuilder builder;
  builder.insertFullStruct("A");

  ASSERT_NO_THROW(llzk::CallGraphAnalysis(builder.getMod()));
}

TEST(CallGraphTests, analysisConstructorBadArg) {
  LLZKTestModuleBuilder builder;
  auto structOp = builder.insertFullStruct("A");

  ASSERT_DEATH(llzk::CallGraphAnalysis(structOp.getOperation()), "CallGraphAnalysis expects provided op to be a ModuleOp!");
}