#include <llzk/Dialect/LLZK/Analysis/CallGraphAnalyses.h>
#include <llzk/Dialect/LLZK/IR/Builders.h>
#include <llzk/Dialect/LLZK/IR/Ops.h>
#include <llzk/Dialect/LLZK/Util/SymbolHelper.h>

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/PassManager.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

using namespace llzk;

class CallGraphTests : public ::testing::Test {
protected:
  mlir::MLIRContext context;
  ModuleBuilder builder;

  CallGraphTests() : context(), builder(&context) { context.loadDialect<llzk::LLZKDialect>(); }

  void SetUp() override {
    // Create a new builder for each test.
    builder = ModuleBuilder(&context);
  }

  void TearDown() override { builder.getRootModule().erase(); }
};

TEST_F(CallGraphTests, constructorTest) {
  builder.insertFullStruct("A");

  ASSERT_NO_THROW(mlir::CallGraph(builder.getRootModule()));
}

TEST_F(CallGraphTests, printTest) {
  builder.insertFullStruct("A");

  std::string s;
  llvm::raw_string_ostream sstream(s);

  llzk::CallGraph cgraph(builder.getRootModule());
  cgraph.print(sstream);

  ASSERT_FALSE(sstream.str().empty());
}

TEST_F(CallGraphTests, numFnTest) {
  builder.insertFullStruct("A");

  llzk::CallGraph cgraph(builder.getRootModule());

  // Size also include "nullptr" function, so it is number of real functions + 1
  // ASSERT_EQ(cgraph.size(), 3);
}

TEST_F(CallGraphTests, reachabilityTest) {
  auto aOp = builder.insertFullStruct("A");
  auto bOp = builder.insertFullStruct("B");
  auto cOp = builder.insertFullStruct("C");

  builder.insertComputeCall(&aOp, &bOp);
  builder.insertComputeCall(&bOp, &cOp);
  builder.insertConstrainCall(&bOp, &aOp);
  builder.insertConstrainCall(&cOp, &aOp);

  auto aComp = builder.getComputeFn(&aOp), bComp = builder.getComputeFn(&bOp),
       cComp = builder.getComputeFn(&cOp);
  auto aCons = builder.getConstrainFn(&aOp), bCons = builder.getConstrainFn(&bOp),
       cCons = builder.getConstrainFn(&cOp);

  mlir::ModuleAnalysisManager mam(builder.getRootModule(), nullptr);
  mlir::AnalysisManager am = mam;
  llzk::CallGraphReachabilityAnalysis cgra(builder.getRootModule().getOperation(), am);

  ASSERT_TRUE(cgra.isReachable(aComp, bComp));
  ASSERT_TRUE(cgra.isReachable(bComp, cComp));
  ASSERT_TRUE(cgra.isReachable(aComp, cComp));
  ASSERT_TRUE(cgra.isReachable(bCons, aCons));
  ASSERT_TRUE(cgra.isReachable(cCons, aCons));

  ASSERT_FALSE(cgra.isReachable(cComp, bComp));
  ASSERT_FALSE(cgra.isReachable(cComp, aCons));
  ASSERT_FALSE(cgra.isReachable(aCons, bCons));
}

TEST_F(CallGraphTests, analysisConstructor) {
  builder.insertFullStruct("A");

  ASSERT_NO_THROW(llzk::CallGraphAnalysis(builder.getRootModule()));
}

TEST_F(CallGraphTests, analysisConstructorBadArg) {
  auto structOp = builder.insertFullStruct("A");

  ASSERT_DEATH(
      llzk::CallGraphAnalysis(structOp.getOperation()),
      "CallGraphAnalysis expects provided op to be a ModuleOp!"
  );
}

// TEST_F(CallGraphTests, removeTest) {
//   LLZKTestModuleBuilder builder;
//   auto structOp = builder.insertFullStruct("A");

//   mlir::CallGraph cgraph(builder.getRootModule());

//   auto removedComputeFn =
//   cgraph.removeFunctionFromModule(cgraph[builder.getComputeFn(&structOp)]);
//   ASSERT_NE(removedComputeFn, nullptr);
//   auto removedConstrainFn =
//       cgraph.removeFunctionFromModule(cgraph[builder.getConstrainFn(&structOp)]);
//   ASSERT_NE(removedConstrainFn, nullptr);

//   // Size also include "nullptr" function, so should just be 1
//   ASSERT_EQ(cgraph.size(), 1);
// }

TEST_F(CallGraphTests, lookupInSymbolTest) {
  auto structOp = builder.insertComputeOnlyStruct("A");
  auto computeFn = builder.getComputeFn(&structOp);

  // not nested
  auto computeOp = mlir::SymbolTable::lookupSymbolIn(structOp, computeFn.getName());
  ASSERT_EQ(computeOp, computeFn);

  // nested
  computeOp =
      mlir::SymbolTable::lookupSymbolIn(builder.getRootModule(), computeFn.getFullyQualifiedName());
  ASSERT_EQ(computeOp, computeFn);
}

TEST_F(CallGraphTests, lookupInSymbolFQNTest) {
  auto a = builder.insertComputeOnlyStruct("A");
  auto b = builder.insertComputeOnlyStruct("B");
  builder.insertComputeCall(&a, &b);
  auto computeFn = builder.getComputeFn(&b);

  // You should be able to find @compute in B
  ASSERT_EQ(computeFn, mlir::SymbolTable::lookupSymbolIn(b, computeFn.getName()));

  // You should be able to find B::@compute in the overall module
  ASSERT_EQ(
      computeFn,
      mlir::SymbolTable::lookupSymbolIn(builder.getRootModule(), computeFn.getFullyQualifiedName())
  );

  auto bSym = mlir::SymbolTable(b);
  auto modSym = mlir::SymbolTable(builder.getRootModule());

  // You should be able to find B::@compute in B
  // but we can't
  ASSERT_EQ(nullptr, mlir::SymbolTable::lookupSymbolIn(b, computeFn.getFullyQualifiedName()));

  // ... unless we use the symbol helpers
  mlir::SymbolTableCollection tables;
  auto res = llzk::lookupTopLevelSymbol<llzk::FuncOp>(
      tables, computeFn.getFullyQualifiedName(), computeFn.getOperation()
  );
  // ASSERT_EQ(computeFn, res.value().get());

  // Since A::compute calls B::compute, you should be able to find B::compute from A
  // auto computeOp = mlir::SymbolTable::lookupSymbolIn(a, computeFn.getFullyQualifiedName());
  // ASSERT_EQ(computeOp, computeFn);
}