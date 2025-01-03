#include <llzk/Dialect/LLZK/Analysis/CallGraphAnalyses.h>
#include <llzk/Dialect/LLZK/IR/Builders.h>
#include <llzk/Dialect/LLZK/IR/Ops.h>
#include <llzk/Dialect/LLZK/Util/SymbolHelper.h>

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/PassManager.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

using namespace llzk;

class ConstraintSummaryTests : public ::testing::Test {
protected:
  static constexpr auto structAName = "structA";
  static constexpr auto structBName = "structB";
  static constexpr auto structCName = "structC";

  mlir::MLIRContext context;
  mlir::OwningOpRef<mlir::ModuleOp> mod;
  ModuleBuilder builder;

  ConstraintSummaryTests() : context(), mod(createLLZKModule(&context)), builder(mod.get()) {
    context.loadDialect<llzk::LLZKDialect>();
  }

  void SetUp() override {
    // Create a new module and builder for each test.
    mod = createLLZKModule(&context);
    builder = ModuleBuilder(mod.get());
  }
};

TEST_F(ConstraintSummaryTests, constructorTest) {
  builder.insertFullStruct(structAName);

  ASSERT_NO_THROW(mlir::CallGraph(builder.getRootModule()));
}

TEST_F(ConstraintSummaryTests, printTest) {
  builder.insertFullStruct(structAName);

  std::string s;
  llvm::raw_string_ostream sstream(s);

  llzk::CallGraph cgraph(builder.getRootModule());
  cgraph.print(sstream);

  ASSERT_FALSE(sstream.str().empty());
}

TEST_F(ConstraintSummaryTests, numFnTest) {
  builder.insertFullStruct(structAName);

  llzk::CallGraph cgraph(builder.getRootModule());

  ASSERT_EQ(cgraph.size(), 2);
}

TEST_F(ConstraintSummaryTests, reachabilityTest) {
  builder.insertFullStruct(structAName)
      .insertFullStruct(structBName)
      .insertFullStruct(structCName)
      .insertComputeCall(structAName, structBName)
      .insertComputeCall(structBName, structCName)
      .insertConstrainCall(structBName, structAName)
      .insertConstrainCall(structCName, structAName);

  auto aComp = *builder.getComputeFn(structAName), bComp = *builder.getComputeFn(structBName),
       cComp = *builder.getComputeFn(structCName);
  auto aCons = *builder.getConstrainFn(structAName), bCons = *builder.getConstrainFn(structBName),
       cCons = *builder.getConstrainFn(structCName);

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

TEST_F(ConstraintSummaryTests, analysisConstructor) {
  builder.insertFullStruct(structAName);

  ASSERT_NO_THROW(llzk::CallGraphAnalysis(builder.getRootModule()));
}

TEST_F(ConstraintSummaryTests, analysisConstructorBadArg) {
  builder.insertFullStruct(structAName);

  auto s = builder.getStruct(structAName);
  ASSERT_TRUE(mlir::succeeded(s));
  ASSERT_DEATH(
      llzk::CallGraphAnalysis(s->getOperation()),
      "CallGraphAnalysis expects provided op to be a ModuleOp!"
  );
}

TEST_F(ConstraintSummaryTests, lookupInSymbolTest) {
  builder.insertComputeOnlyStruct(structAName);
  auto computeFn = builder.getComputeFn(structAName);
  ASSERT_TRUE(mlir::succeeded(computeFn));

  // not nested
  auto computeOp =
      mlir::SymbolTable::lookupSymbolIn(*builder.getStruct(structAName), computeFn->getName());
  ASSERT_EQ(computeOp, *computeFn);

  // nested
  computeOp = mlir::SymbolTable::lookupSymbolIn(
      builder.getRootModule(), computeFn->getFullyQualifiedName()
  );
  ASSERT_EQ(computeOp, *computeFn);
}

TEST_F(ConstraintSummaryTests, lookupInSymbolFQNTest) {
  builder.insertComputeOnlyStruct(structAName)
      .insertComputeOnlyStruct(structBName)
      .insertComputeCall(structAName, structBName);

  auto b = builder.getStruct(structBName);
  auto computeFn = builder.getComputeFn(structBName);
  // You should be able to find @compute in B
  ASSERT_EQ(*computeFn, mlir::SymbolTable::lookupSymbolIn(*b, computeFn->getName()));

  // You should be able to find B::@compute in the overall module
  ASSERT_EQ(
      *computeFn,
      mlir::SymbolTable::lookupSymbolIn(builder.getRootModule(), computeFn->getFullyQualifiedName())
  );

  auto bSym = mlir::SymbolTable(*b);
  auto modSym = mlir::SymbolTable(builder.getRootModule());

  // You should be able to find B::@compute in B, but we can't with built-in symbol tables
  ASSERT_EQ(nullptr, mlir::SymbolTable::lookupSymbolIn(*b, computeFn->getFullyQualifiedName()));

  // But we can find B::@compute in B with the symbol helpers
  mlir::SymbolTableCollection tables;
  auto res = llzk::lookupTopLevelSymbol<llzk::FuncOp>(
      tables, computeFn->getFullyQualifiedName(), computeFn->getOperation()
  );
  ASSERT_EQ(*computeFn, res.value().get());
}