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