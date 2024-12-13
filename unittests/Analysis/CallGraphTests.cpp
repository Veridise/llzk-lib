#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <llzk/Dialect/LLZK/IR/Builders.h>
#include <llzk/Dialect/LLZK/IR/Ops.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/PassManager.h>

#include <llzk/Dialect/LLZK/Analysis/CallGraphAnalyses.h>
#include <llzk/Dialect/LLZK/Util/SymbolHelper.h>

using namespace llzk;

TEST(CallGraphTests, constructorTest) {
  ModuleBuilder builder;
  builder.insertFullStruct("A");

  ASSERT_NO_THROW(mlir::CallGraph(builder.getMod()));
}

TEST(CallGraphTests, printTest) {
  ModuleBuilder builder;
  builder.insertFullStruct("A");

  std::string s;
  llvm::raw_string_ostream sstream(s);

  llzk::CallGraph cgraph(builder.getMod());
  cgraph.print(sstream);

  ASSERT_FALSE(sstream.str().empty());
}

TEST(CallGraphTests, numFnTest) {
  ModuleBuilder builder;
  builder.insertFullStruct("A");

  llzk::CallGraph cgraph(builder.getMod());

  // Size also include "nullptr" function, so it is number of real functions + 1
  // ASSERT_EQ(cgraph.size(), 3);
}

TEST(CallGraphTests, reachabilityTest) {
  ModuleBuilder builder;
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

  mlir::ModuleAnalysisManager mam(builder.getMod(), nullptr);
  mlir::AnalysisManager am = mam;
  llzk::CallGraphReachabilityAnalysis cgra(builder.getMod().getOperation(), am);

  ASSERT_TRUE(cgra.isReachable(aComp, bComp));
  ASSERT_TRUE(cgra.isReachable(bComp, cComp));
  ASSERT_TRUE(cgra.isReachable(aComp, cComp));
  ASSERT_TRUE(cgra.isReachable(bCons, aCons));
  ASSERT_TRUE(cgra.isReachable(cCons, aCons));

  ASSERT_FALSE(cgra.isReachable(cComp, bComp));
  ASSERT_FALSE(cgra.isReachable(cComp, aCons));
  ASSERT_FALSE(cgra.isReachable(aCons, bCons));
}

TEST(CallGraphTests, analysisConstructor) {
  ModuleBuilder builder;
  builder.insertFullStruct("A");

  ASSERT_NO_THROW(llzk::CallGraphAnalysis(builder.getMod()));
}

TEST(CallGraphTests, analysisConstructorBadArg) {
  ModuleBuilder builder;
  auto structOp = builder.insertFullStruct("A");

  ASSERT_DEATH(
      llzk::CallGraphAnalysis(structOp.getOperation()),
      "CallGraphAnalysis expects provided op to be a ModuleOp!"
  );
}

// TEST(CallGraphTests, removeTest) {
//   LLZKTestModuleBuilder builder;
//   auto structOp = builder.insertFullStruct("A");

//   mlir::CallGraph cgraph(builder.getMod());

//   auto removedComputeFn =
//   cgraph.removeFunctionFromModule(cgraph[builder.getComputeFn(&structOp)]);
//   ASSERT_NE(removedComputeFn, nullptr);
//   auto removedConstrainFn =
//       cgraph.removeFunctionFromModule(cgraph[builder.getConstrainFn(&structOp)]);
//   ASSERT_NE(removedConstrainFn, nullptr);

//   // Size also include "nullptr" function, so should just be 1
//   ASSERT_EQ(cgraph.size(), 1);
// }

TEST(SymbolTableTests, lookupInSymbolTest) {
  ModuleBuilder builder;
  auto structOp = builder.insertComputeOnlyStruct("A");
  auto computeFn = builder.getComputeFn(&structOp);

  // not nested
  auto computeOp = mlir::SymbolTable::lookupSymbolIn(structOp, computeFn.getName());
  ASSERT_EQ(computeOp, computeFn);
  // auto vis = mlir::SymbolTable::getSymbolVisibility(computeOp);
  // llvm::errs() << vis << "\n";

  // nested
  computeOp =
      mlir::SymbolTable::lookupSymbolIn(builder.getMod(), computeFn.getFullyQualifiedName());
  ASSERT_EQ(computeOp, computeFn);
}

TEST(SymbolTableTests, lookupInSymbolFQNTest) {
  ModuleBuilder builder;
  auto a = builder.insertComputeOnlyStruct("A");
  auto b = builder.insertComputeOnlyStruct("B");
  builder.insertComputeCall(&a, &b);
  auto computeFn = builder.getComputeFn(&b);

  // You should be able to find @compute in B
  ASSERT_EQ(computeFn, mlir::SymbolTable::lookupSymbolIn(b, computeFn.getName()));

  // You should be able to find B::@compute in the overall module
  ASSERT_EQ(
      computeFn,
      mlir::SymbolTable::lookupSymbolIn(builder.getMod(), computeFn.getFullyQualifiedName())
  );

  auto bSym = mlir::SymbolTable(b);
  auto modSym = mlir::SymbolTable(builder.getMod());
  // llvm::errs() << bSym << "\n";
  // llvm::err

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

TEST(SymbolTableTests, resolveCallableTest) {
  ModuleBuilder builder;
  auto a = builder.insertComputeOnlyStruct("A");
  auto b = builder.insertComputeOnlyStruct("B");
  builder.insertComputeCall(&a, &b);

  builder.getMod().walk([&](llzk::CallOp c) {
    auto callOp = mlir::dyn_cast<mlir::CallOpInterface>(c.getOperation());
    ASSERT_NE(callOp, nullptr);
    // llvm::errs() << *callOp << "\n";
    // auto op = callOp.resolveCallable(nullptr);
    // ASSERT_NE(op, nullptr);
    // auto op = callOp.resolveCallable(nullptr);
    auto res = llzk::resolveCallable<llzk::FuncOp>(callOp);
    ASSERT_TRUE(mlir::LogicalResult(res).succeeded());
    auto val = std::move(res.value());
    ASSERT_TRUE(val);
    auto op = val.get();
    ASSERT_NE(op, nullptr);
    ASSERT_EQ(builder.getComputeFn(&b), op);
  });
}