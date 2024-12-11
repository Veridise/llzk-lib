#include "OpBuilders.h"

#include "llzk/Dialect/LLZK/Util/SymbolHelper.h"

#include "gtest/gtest.h"
#include <cassert>

using namespace mlir;

/* LLZKTestModuleBuilder */

LLZKTestModuleBuilder::LLZKTestModuleBuilder() {
  auto dialect = context.getOrLoadDialect<llzk::LLZKDialect>();
  auto langAttr = StringAttr::get(&context, dialect->getNamespace());
  mod = ModuleOp::create(UnknownLoc::get(&context));
  mod->setAttr(llzk::LANG_ATTR_NAME, langAttr);
}

llzk::StructDefOp LLZKTestModuleBuilder::insertEmptyStruct(std::string_view structName) {
  assert(structMap.find(structName) == structMap.end());

  OpBuilder opBuilder(mod.getBody(), mod.getBody()->begin());
  auto structNameAtrr = StringAttr::get(&context, structName);
  auto structDef =
      opBuilder.create<llzk::StructDefOp>(UnknownLoc::get(&context), structNameAtrr, nullptr);
  // populate the initial region
  auto &region = structDef.getRegion();
  if (region.empty()) {
    region.push_back(new mlir::Block());
  }
  structMap[structName] = structDef;

  return structDef;
}

llzk::FuncOp LLZKTestModuleBuilder::insertComputeFn(llzk::StructDefOp *op) {
  OpBuilder opBuilder(op->getBody());
  assert(computeFnMap.find(op->getName()) == computeFnMap.end());

  auto structType = llzk::StructType::get(&context, SymbolRefAttr::get(*op));

  auto fnOp = opBuilder.create<llzk::FuncOp>(
      UnknownLoc::get(&context), StringAttr::get(&context, llzk::FUNC_NAME_COMPUTE),
      FunctionType::get(&context, {}, {structType})
  );
  fnOp.addEntryBlock();
  computeFnMap[op->getName()] = fnOp;
  return fnOp;
}

llzk::FuncOp LLZKTestModuleBuilder::insertConstrainFn(llzk::StructDefOp *op) {
  assert(constrainFnMap.find(op->getName()) == constrainFnMap.end());

  OpBuilder opBuilder(op->getBody());

  auto structType = llzk::StructType::get(&context, SymbolRefAttr::get(*op));

  auto fnOp = opBuilder.create<llzk::FuncOp>(
      UnknownLoc::get(&context), StringAttr::get(&context, llzk::FUNC_NAME_CONSTRAIN),
      FunctionType::get(&context, {structType}, {})
  );
  fnOp.addEntryBlock();

  constrainFnMap[op->getName()] = fnOp;
  return fnOp;
}

void LLZKTestModuleBuilder::insertComputeCall(
    llzk::StructDefOp *caller, llzk::StructDefOp *callee
) {
  auto callerFn = computeFnMap.at(caller->getName());
  auto calleeFn = computeFnMap.at(callee->getName());

  OpBuilder builder(callerFn.getBody());
  builder.create<llzk::CallOp>(
      UnknownLoc::get(&context),
      /*
        Note that using the FQN for the function call is required, simply using
        the FuncOp will only insert the function's name, omitting the struct.
      */
      getFullyQualifiedFuncSymbol(callee, calleeFn), mlir::ValueRange{}
  );
  updateComputeReachability(caller, callee);
}

void LLZKTestModuleBuilder::insertConstrainCall(
    llzk::StructDefOp *caller, llzk::StructDefOp *callee
) {
  auto callerFn = constrainFnMap.at(caller->getName());
  auto calleeFn = constrainFnMap.at(callee->getName());
  auto calleeTy = llzk::StructType::get(&context, SymbolRefAttr::get(*callee));

  size_t numOps = 0;
  for (auto it = caller->getBody().begin(); it != caller->getBody().end(); it++, numOps++)
    ;
  auto fieldName = StringAttr::get(&context, callee->getName().str() + std::to_string(numOps));

  // Insert the field declaration op
  {
    OpBuilder builder(caller->getBody());
    builder.create<llzk::FieldDefOp>(UnknownLoc::get(&context), fieldName, calleeTy);
  }

  // Insert the constrain function ops
  {
    OpBuilder builder(callerFn.getBody());

    auto field = builder.create<llzk::FieldReadOp>(
        UnknownLoc::get(&context), calleeTy,
        callerFn.getBody().getArgument(0), // first arg is self
        fieldName
    );
    builder.create<llzk::CallOp>(
        UnknownLoc::get(&context), getFullyQualifiedFuncSymbol(callee, calleeFn),
        mlir::ValueRange{field}
    );
  }
  updateConstrainReachability(caller, callee);
}

/* Tests for the LLZKTestModuleBuilder */

TEST(LLZKTestModuleBuilderTests, testModuleOpCreation) {
  LLZKTestModuleBuilder builder;

  ASSERT_NE(builder.getMod(), nullptr);
}

TEST(LLZKTestModuleBuilderTests, testStructDefInsertion) {
  LLZKTestModuleBuilder builder;

  auto structDef = builder.insertEmptyStruct("structOne");
  ASSERT_EQ(builder.getStruct("structOne"), structDef);
}

TEST(LLZKTestModuleBuilderTests, testFnInsertion) {
  LLZKTestModuleBuilder builder;

  auto structOp = builder.insertFullStruct("structOne");

  auto computeFn = builder.getComputeFn(&structOp);
  ASSERT_EQ(computeFn.getBody().getArguments().size(), 0);

  auto constrainFn = builder.getConstrainFn(&structOp);
  ASSERT_EQ(constrainFn.getBody().getArguments().size(), 1);
}

TEST(LLZKTestModuleBuilderTests, testReachabilitySimple) {
  LLZKTestModuleBuilder builder;

  auto a = builder.insertComputeOnlyStruct("structA");
  auto b = builder.insertComputeOnlyStruct("structB");
  builder.insertComputeCall(&a, &b);

  ASSERT_TRUE(builder.computeReachable(&a, &b));
  ASSERT_FALSE(builder.computeReachable(&b, &a));
}

TEST(LLZKTestModuleBuilderTests, testReachabilityTransitive) {
  LLZKTestModuleBuilder builder;

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
  LLZKTestModuleBuilder builder;

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
  LLZKTestModuleBuilder builder;

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