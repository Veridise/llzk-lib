#include "llzk/Dialect/LLZK/IR/Builders.h"
#include "llzk/Dialect/LLZK/IR/Ops.h"

#include <mlir/Dialect/Index/IR/IndexOps.h>

#include <gtest/gtest.h>

using namespace llzk;
using namespace mlir;

class AffineMapInstantiationTests : public ::testing::Test {
protected:
  static constexpr auto funcNameA = "FuncA";
  static constexpr auto funcNameB = "FuncB";
  static constexpr auto structNameA = "StructA";
  static constexpr auto structNameB = "StructB";
  static constexpr auto structNameC = "StructC";

  MLIRContext ctx;
  Location loc;
  OwningOpRef<ModuleOp> mod;

  AffineMapInstantiationTests() : ctx(), loc(getUnknownLoc(&ctx)), mod() {
    ctx.loadDialect<llzk::LLZKDialect>();
  }

  void SetUp() override {
    // Create a new module for each test
    mod = createLLZKModule(&ctx, loc);
  }

  void TearDown() override {
    // Allow existing module to be erased after each test
    mod = OwningOpRef<ModuleOp>();
  }

  ModuleBuilder newEmptyExample() { return ModuleBuilder {mod.get()}; }

  ModuleBuilder newBasicFunctionsExample(size_t numParams) {
    IndexType idxTy = IndexType::get(&ctx);
    SmallVector<Type> paramTypes(numParams, idxTy);
    FunctionType fTy = FunctionType::get(&ctx, TypeRange(paramTypes), TypeRange {idxTy});
    ModuleBuilder llzkBldr(mod.get());
    llzkBldr.insertGlobalFunc(funcNameB, fTy).insertGlobalFunc(funcNameA, fTy);
    return llzkBldr;
  }

  ModuleBuilder newBasicStructExample() {
    ModuleBuilder llzkBldr(mod.get());
    llzkBldr.insertFullStruct(structNameA)
        .insertFullStruct(structNameB)
        .insertComputeCall(structNameA, structNameB)
        .insertConstrainCall(structNameA, structNameB);
    return llzkBldr;
  }
};

template <typename ConcreteType> bool verify(Operation *op, bool verifySymbolUses = false) {
  // First, call the ODS-generated function for the Op to ensure that necessary attributes exist.
  if (failed(llvm::cast<ConcreteType>(op).verifyInvariants())) {
    return false;
  }
  // Second, verify all traits on the Op and call the custom verify() (if defined) via the
  // `verifyInvariants()` function in `OpDefinition.h`.
  if (failed(op->getName().verifyInvariants(op))) {
    return false;
  }
  // Finally, if applicable, call the ODS-generated `verifySymbolUses()` function.
  if (verifySymbolUses) {
    if (SymbolUserOpInterface userOp = llvm::dyn_cast<SymbolUserOpInterface>(op)) {
      SymbolTableCollection tables;
      if (failed(userOp.verifySymbolUses(tables))) {
        return false;
      }
    }
  }
  //
  return true;
}

template <typename ConcreteType>
inline bool verify(ConcreteType op, bool verifySymbolUses = false) {
  return verify<ConcreteType>(op.getOperation(), verifySymbolUses);
}

//===------------------------------------------------------------------===//
// CreateArrayOp::build(..., ArrayType, ValueRange)
//===------------------------------------------------------------------===//

TEST_F(AffineMapInstantiationTests, testElementInit_GoodEmpty) {
  OpBuilder bldr(mod->getRegion());
  ArrayType arrTy = ArrayType::get(bldr.getIndexType(), {2, 2}); // !llzk.array<2,2 x index>
  CreateArrayOp op = bldr.create<CreateArrayOp>(loc, arrTy);
  ASSERT_TRUE(verify(op));
}

TEST_F(AffineMapInstantiationTests, testElementInit_GoodNonEmpty) {
  OpBuilder bldr(mod->getRegion());
  ArrayType arrTy = ArrayType::get(bldr.getIndexType(), {2}); // !llzk.array<2 x index>
  auto v1 = bldr.create<index::ConstantOp>(loc, 766);
  auto v2 = bldr.create<index::ConstantOp>(loc, 562);
  CreateArrayOp op = bldr.create<CreateArrayOp>(loc, arrTy, ValueRange {v1, v2});
  ASSERT_TRUE(verify(op));
}

TEST_F(AffineMapInstantiationTests, testElementInit_TooFew) {
  OpBuilder bldr(mod->getRegion());
  ArrayType arrTy = ArrayType::get(bldr.getIndexType(), {5}); // !llzk.array<5 x index>
  auto v1 = bldr.create<index::ConstantOp>(loc, 766);
  auto v2 = bldr.create<index::ConstantOp>(loc, 562);
  EXPECT_DEATH(
      {
        CreateArrayOp op = bldr.create<CreateArrayOp>(loc, arrTy, ValueRange {v1, v2});
        assert(verify(op));
      },
      "error: 'llzk.new_array' op failed to verify that operand types match result type"
  );
}

TEST_F(AffineMapInstantiationTests, testElementInit_TooMany) {
  OpBuilder bldr(mod->getRegion());
  ArrayType arrTy = ArrayType::get(bldr.getIndexType(), {1}); // !llzk.array<1 x index>
  auto v1 = bldr.create<index::ConstantOp>(loc, 766);
  auto v2 = bldr.create<index::ConstantOp>(loc, 562);
  EXPECT_DEATH(
      {
        CreateArrayOp op = bldr.create<CreateArrayOp>(loc, arrTy, ValueRange {v1, v2});
        assert(verify(op));
      },
      "error: 'llzk.new_array' op failed to verify that operand types match result type"
  );
}

TEST_F(AffineMapInstantiationTests, testElementInit_WithAffineMapType) {
  OpBuilder bldr(mod->getRegion());
  AffineMapAttr m = AffineMapAttr::get(bldr.getDimIdentityMap()); // (d0) -> (d0)
  ArrayType arrTy = ArrayType::get(bldr.getIndexType(), {m});     // !llzk.array<#m x index>
  EXPECT_DEATH(
      {
        CreateArrayOp op = bldr.create<CreateArrayOp>(loc, arrTy);
        assert(verify(op));
      },
      "error: 'llzk.new_array' op map instantiation group count \\(0\\) does not match the number "
      "of affine map instantiations \\(1\\) required by the type"
  );
}

//===------------------------------------------------------------------===//
// CreateArrayOp::build(..., ArrayType, ArrayRef<ValueRange>, ArrayRef<int32_t>)
//===------------------------------------------------------------------===//

TEST_F(AffineMapInstantiationTests, testMapOpInit_Good) {
  OpBuilder bldr(mod->getRegion());
  AffineMapAttr m = AffineMapAttr::get(bldr.getDimIdentityMap()); // (d0) -> (d0)
  ArrayType arrTy = ArrayType::get(bldr.getIndexType(), {m, m});  // !llzk.array<#m,#m x index>

  SmallVector<ValueRange> mapOperands;
  auto v1 = bldr.create<index::ConstantOp>(loc, 10);
  mapOperands.push_back(ValueRange {v1});
  auto v2 = bldr.create<index::ConstantOp>(loc, 98);
  mapOperands.push_back(ValueRange {v2});
  SmallVector<int32_t> numDimsPerMap = {1, 1};
  CreateArrayOp op = bldr.create<CreateArrayOp>(loc, arrTy, mapOperands, numDimsPerMap);
  ASSERT_TRUE(verify(op));
}

TEST_F(AffineMapInstantiationTests, testMapOpInit_Op1_Dim1_Type2) {
  OpBuilder bldr(mod->getRegion());
  AffineMapAttr m = AffineMapAttr::get(bldr.getDimIdentityMap()); // (d0) -> (d0)
  ArrayType arrTy = ArrayType::get(bldr.getIndexType(), {m, m});  // !llzk.array<#m,#m x index>

  SmallVector<ValueRange> mapOperands;
  auto v1 = bldr.create<index::ConstantOp>(loc, 10);
  mapOperands.push_back(ValueRange {v1});
  SmallVector<int32_t> numDimsPerMap = {1};
  EXPECT_DEATH(
      {
        CreateArrayOp op = bldr.create<CreateArrayOp>(loc, arrTy, mapOperands, numDimsPerMap);
        assert(verify(op));
      },
      "error: 'llzk.new_array' op map instantiation group count \\(1\\) does not match the number "
      "of affine map instantiations \\(2\\) required by the type"
  );
}

TEST_F(AffineMapInstantiationTests, testMapOpInit_Op1_Dim2_Type2) {
  OpBuilder bldr(mod->getRegion());
  AffineMapAttr m = AffineMapAttr::get(bldr.getDimIdentityMap()); // (d0) -> (d0)
  ArrayType arrTy = ArrayType::get(bldr.getIndexType(), {m, m});  // !llzk.array<#m,#m x index>

  SmallVector<ValueRange> mapOperands;
  auto v1 = bldr.create<index::ConstantOp>(loc, 10);
  mapOperands.push_back(ValueRange {v1});
  SmallVector<int32_t> numDimsPerMap = {1, 0};
  EXPECT_DEATH(
      {
        CreateArrayOp op = bldr.create<CreateArrayOp>(loc, arrTy, mapOperands, numDimsPerMap);
        assert(verify(op));
      },
      "error: 'llzk.new_array' op map instantiation group count \\(1\\) does not match with length "
      "of 'mapOpGroupSizes' attribute \\(2\\)"
  );
}

TEST_F(AffineMapInstantiationTests, testMapOpInit_Op2_Dim1_Type2) {
  OpBuilder bldr(mod->getRegion());
  AffineMapAttr m = AffineMapAttr::get(bldr.getDimIdentityMap()); // (d0) -> (d0)
  ArrayType arrTy = ArrayType::get(bldr.getIndexType(), {m, m});  // !llzk.array<#m,#m x index>

  SmallVector<ValueRange> mapOperands;
  auto v1 = bldr.create<index::ConstantOp>(loc, 10);
  mapOperands.push_back(ValueRange {v1});
  auto v2 = bldr.create<index::ConstantOp>(loc, 98);
  mapOperands.push_back(ValueRange {v2});
  SmallVector<int32_t> numDimsPerMap = {1};
  EXPECT_DEATH(
      {
        CreateArrayOp op = bldr.create<CreateArrayOp>(loc, arrTy, mapOperands, numDimsPerMap);
        assert(verify(op));
      },
      "error: 'llzk.new_array' op map instantiation group count \\(2\\) does not match with length "
      "of 'mapOpGroupSizes' attribute \\(1\\)"
  );
}

TEST_F(AffineMapInstantiationTests, testMapOpInit_Op3_Dim3_Type1) {
  OpBuilder bldr(mod->getRegion());
  AffineMapAttr m = AffineMapAttr::get(bldr.getDimIdentityMap()); // (d0) -> (d0)
  ArrayType arrTy = ArrayType::get(bldr.getIndexType(), {m});     // !llzk.array<#m x index>

  SmallVector<ValueRange> mapOperands;
  auto v1 = bldr.create<index::ConstantOp>(loc, 10);
  mapOperands.push_back(ValueRange {v1});
  auto v2 = bldr.create<index::ConstantOp>(loc, 98);
  mapOperands.push_back(ValueRange {v2});
  auto v3 = bldr.create<index::ConstantOp>(loc, 4);
  mapOperands.push_back(ValueRange {v3});
  SmallVector<int32_t> numDimsPerMap = {1, 1, 1};
  EXPECT_DEATH(
      {
        CreateArrayOp op = bldr.create<CreateArrayOp>(loc, arrTy, mapOperands, numDimsPerMap);
        assert(verify(op));
      },
      "error: 'llzk.new_array' op map instantiation group count \\(3\\) does not match the number "
      "of affine map instantiations \\(1\\) required by the type"
  );
}

TEST_F(AffineMapInstantiationTests, testMapOpInit_Op3_Dim2_Type1) {
  OpBuilder bldr(mod->getRegion());
  AffineMapAttr m = AffineMapAttr::get(bldr.getDimIdentityMap()); // (d0) -> (d0)
  ArrayType arrTy = ArrayType::get(bldr.getIndexType(), {m});     // !llzk.array<#m x index>

  SmallVector<ValueRange> mapOperands;
  auto v1 = bldr.create<index::ConstantOp>(loc, 10);
  mapOperands.push_back(ValueRange {v1});
  auto v2 = bldr.create<index::ConstantOp>(loc, 98);
  mapOperands.push_back(ValueRange {v2});
  auto v3 = bldr.create<index::ConstantOp>(loc, 4);
  mapOperands.push_back(ValueRange {v3});
  SmallVector<int32_t> numDimsPerMap = {1, 1};
  EXPECT_DEATH(
      {
        CreateArrayOp op = bldr.create<CreateArrayOp>(loc, arrTy, mapOperands, numDimsPerMap);
        assert(verify(op));
      },
      "error: 'llzk.new_array' op map instantiation group count \\(3\\) does not match with length "
      "of 'mapOpGroupSizes' attribute \\(2\\)"
  );
}

TEST_F(AffineMapInstantiationTests, testMapOpInit_Op2_Dim3_Type1) {
  OpBuilder bldr(mod->getRegion());
  AffineMapAttr m = AffineMapAttr::get(bldr.getDimIdentityMap()); // (d0) -> (d0)
  ArrayType arrTy = ArrayType::get(bldr.getIndexType(), {m});     // !llzk.array<#m x index>

  SmallVector<ValueRange> mapOperands;
  auto v1 = bldr.create<index::ConstantOp>(loc, 10);
  mapOperands.push_back(ValueRange {v1});
  auto v2 = bldr.create<index::ConstantOp>(loc, 98);
  mapOperands.push_back(ValueRange {v2});
  SmallVector<int32_t> numDimsPerMap = {1, 1, 1};
  EXPECT_DEATH(
      {
        CreateArrayOp op = bldr.create<CreateArrayOp>(loc, arrTy, mapOperands, numDimsPerMap);
        assert(verify(op));
      },
      "error: 'llzk.new_array' op map instantiation group count \\(2\\) does not match with length "
      "of 'mapOpGroupSizes' attribute \\(3\\)"
  );
}

TEST_F(AffineMapInstantiationTests, testMapOpInit_NumDimsTooHigh) {
  OpBuilder bldr(mod->getRegion());
  AffineMapAttr m = AffineMapAttr::get(bldr.getDimIdentityMap()); // (d0) -> (d0)
  ArrayType arrTy = ArrayType::get(bldr.getIndexType(), {m});     // !llzk.array<#m x index>

  SmallVector<ValueRange> mapOperands;
  auto v1 = bldr.create<index::ConstantOp>(loc, 10);
  mapOperands.push_back(ValueRange {v1});
  SmallVector<int32_t> numDimsPerMap = {9};
  EXPECT_DEATH(
      {
        CreateArrayOp op = bldr.create<CreateArrayOp>(loc, arrTy, mapOperands, numDimsPerMap);
        assert(verify(op));
      },
      "error: 'llzk.new_array' op instantiation of map 0 expected 1 but found 9 dimension values "
      "in \\(\\)"
  );
}

TEST_F(AffineMapInstantiationTests, testMapOpInit_TooManyOpsForMap) {
  OpBuilder bldr(mod->getRegion());
  AffineMapAttr m = AffineMapAttr::get(bldr.getDimIdentityMap()); // (d0) -> (d0)
  ArrayType arrTy = ArrayType::get(bldr.getIndexType(), {m});     // !llzk.array<#m x index>

  SmallVector<ValueRange> mapOperands;
  auto v1 = bldr.create<index::ConstantOp>(loc, 10);
  auto v2 = bldr.create<index::ConstantOp>(loc, 23);
  mapOperands.push_back(ValueRange {v1, v2});
  SmallVector<int32_t> numDimsPerMap = {1};
  EXPECT_DEATH(
      {
        CreateArrayOp op = bldr.create<CreateArrayOp>(loc, arrTy, mapOperands, numDimsPerMap);
        assert(verify(op));
      },
      "error: 'llzk.new_array' op instantiation of map 0 expected 0 but found 1 symbol values in "
      "\\[\\]"
  );
}

TEST_F(AffineMapInstantiationTests, testMapOpInit_TooFewOpsForMap) {
  OpBuilder bldr(mod->getRegion());
  // (d0, d1) -> (d0 + d1)
  AffineMapAttr m = AffineMapAttr::get(AffineMap::get(
      /*dimCount=*/2, /*symbolCount=*/0, bldr.getAffineDimExpr(0) + bldr.getAffineDimExpr(1)
  ));
  ArrayType arrTy = ArrayType::get(bldr.getIndexType(), {m}); // !llzk.array<#m x index>
  SmallVector<ValueRange> mapOperands;
  auto v1 = bldr.create<index::ConstantOp>(loc, 10);
  mapOperands.push_back(ValueRange {v1});
  SmallVector<int32_t> numDimsPerMap = {1};
  EXPECT_DEATH(
      {
        CreateArrayOp op = bldr.create<CreateArrayOp>(loc, arrTy, mapOperands, numDimsPerMap);
        assert(verify(op));
      },
      "error: 'llzk.new_array' op instantiation of map 0 expected 2 but found 1 dimension values "
      "in \\(\\)"
  );
}

TEST_F(AffineMapInstantiationTests, testMapOpInit_WrongTypeForMapOperands) {
  OpBuilder bldr(mod->getRegion());
  AffineMapAttr m = AffineMapAttr::get(bldr.getDimIdentityMap()); // (d0) -> (d0)
  ArrayType arrTy = ArrayType::get(bldr.getIndexType(), {m});     // !llzk.array<#m x index>

  SmallVector<ValueRange> mapOperands;
  FeltConstAttr a = bldr.getAttr<FeltConstAttr>(APInt::getZero(64));
  auto v1 = bldr.create<FeltConstantOp>(loc, a);
  mapOperands.push_back(ValueRange {v1});
  SmallVector<int32_t> numDimsPerMap = {1};
  EXPECT_DEATH(
      {
        CreateArrayOp op = bldr.create<CreateArrayOp>(loc, arrTy, mapOperands, numDimsPerMap);
        assert(verify(op));
      },
      "error: 'llzk.new_array' op operand #0 must be variadic of index, but got '!llzk.felt'"
  );
}

//===------------------------------------------------------------------===//
// CallOp::build(..., TypeRange, SymbolRefAttr, ValueRange)
//===------------------------------------------------------------------===//

TEST_F(AffineMapInstantiationTests, testCallNoAffine_GoodNoArgs) {
  ModuleBuilder llzkBldr = newBasicFunctionsExample(0);

  auto funcA = llzkBldr.getGlobalFunc(funcNameA);
  ASSERT_TRUE(mlir::succeeded(funcA));
  auto funcB = llzkBldr.getGlobalFunc(funcNameB);
  ASSERT_TRUE(mlir::succeeded(funcB));

  OpBuilder bldr(funcA->getBody());
  CallOp op = bldr.create<CallOp>(
      loc, funcB->getResultTypes(), funcB->getFullyQualifiedName(), ValueRange {}
  );
  // module attributes {veridise.lang = "llzk"} {
  //   llzk.func @FuncA() -> index {
  //     %0 = call @FuncB() : () -> index
  //   }
  //   llzk.func @FuncB() -> index {
  //   }
  // }
  ASSERT_TRUE(verify(mod.get()));
  ASSERT_TRUE(verify(op, true));
}

TEST_F(AffineMapInstantiationTests, testCallNoAffine_GoodWithArgs) {
  ModuleBuilder llzkBldr = newBasicFunctionsExample(2);

  auto funcA = llzkBldr.getGlobalFunc(funcNameA);
  ASSERT_TRUE(mlir::succeeded(funcA));
  auto funcB = llzkBldr.getGlobalFunc(funcNameB);
  ASSERT_TRUE(mlir::succeeded(funcB));

  OpBuilder bldr(funcA->getBody());
  auto v1 = bldr.create<index::ConstantOp>(loc, 5);
  auto v2 = bldr.create<index::ConstantOp>(loc, 2);
  CallOp op = bldr.create<CallOp>(
      loc, funcB->getResultTypes(), funcB->getFullyQualifiedName(), ValueRange {v1, v2}
  );
  // module attributes {veridise.lang = "llzk"} {
  //   llzk.func @FuncA(%arg0: index, %arg1: index) -> index {
  //     %idx5 = index.constant 5
  //     %idx2 = index.constant 2
  //     %0 = call @FuncB(%idx5, %idx2) : (index, index) -> index
  //   }
  //   llzk.func @FuncB(%arg0: index, %arg1: index) -> index {
  //   }
  // }
  ASSERT_TRUE(verify(mod.get()));
  ASSERT_TRUE(verify(op, true));
}

TEST_F(AffineMapInstantiationTests, testCallNoAffine_TooFewValues) {
  ModuleBuilder llzkBldr = newBasicFunctionsExample(2);

  auto funcA = llzkBldr.getGlobalFunc(funcNameA);
  ASSERT_TRUE(mlir::succeeded(funcA));
  auto funcB = llzkBldr.getGlobalFunc(funcNameB);
  ASSERT_TRUE(mlir::succeeded(funcB));

  OpBuilder bldr(funcA->getBody());
  auto v1 = bldr.create<index::ConstantOp>(loc, 5);
  CallOp op = bldr.create<CallOp>(
      loc, funcB->getResultTypes(), funcB->getFullyQualifiedName(), ValueRange {v1}
  );
  // module attributes {veridise.lang = "llzk"} {
  //   llzk.func @FuncA(%arg0: index, %arg1: index) -> index {
  //     %idx5 = index.constant 5
  //     %0 = call @FuncB(%idx5) : (index) -> index
  //   }
  //   llzk.func @FuncB(%arg0: index, %arg1: index) -> index {
  //   }
  // }
  EXPECT_DEATH(
      {
        assert(verify(mod.get()));
        assert(verify(op, true));
      },
      "error: 'llzk.call' op incorrect number of operands for callee, expected 2"
  );
}

TEST_F(AffineMapInstantiationTests, testCallNoAffine_WrongRetTy) {
  ModuleBuilder llzkBldr = newBasicFunctionsExample(1);

  auto funcA = llzkBldr.getGlobalFunc(funcNameA);
  ASSERT_TRUE(mlir::succeeded(funcA));
  auto funcB = llzkBldr.getGlobalFunc(funcNameB);
  ASSERT_TRUE(mlir::succeeded(funcB));

  OpBuilder bldr(funcA->getBody());
  auto v1 = bldr.create<index::ConstantOp>(loc, 5);
  CallOp op = bldr.create<CallOp>(
      loc, TypeRange {bldr.getI1Type()}, funcB->getFullyQualifiedName(), ValueRange {v1}
  );
  // module attributes {veridise.lang = "llzk"} {
  //   llzk.func @FuncA(%arg0: index) -> index {
  //     %idx5 = index.constant 5
  //     %0 = call @FuncB(%idx5) : (index) -> i1
  //   }
  //   llzk.func @FuncB(%arg0: index) -> index {
  //   }
  // }
  EXPECT_DEATH(
      {
        assert(verify(mod.get()));
        assert(verify(op, true));
      },
      "error: 'llzk.call' op result type mismatch: expected type 'index', but found 'i1' for "
      "result number 0"
  );
}

TEST_F(AffineMapInstantiationTests, testCallNoAffine_InvalidCalleeName) {
  ModuleBuilder llzkBldr = newBasicFunctionsExample(0);

  auto funcA = llzkBldr.getGlobalFunc(funcNameA);
  ASSERT_TRUE(mlir::succeeded(funcA));

  OpBuilder bldr(funcA->getBody());
  CallOp op = bldr.create<CallOp>(
      loc, TypeRange {}, FlatSymbolRefAttr::get(&ctx, "invalidName"), ValueRange {}
  );
  // module attributes {veridise.lang = "llzk"} {
  //   llzk.func @FuncA() -> index {
  //     call @invalidName() : () -> ()
  //   }
  //   llzk.func @FuncB() -> index {
  //   }
  // }
  EXPECT_DEATH(
      {
        assert(verify(mod.get()));
        assert(verify(op, true));
      },
      "error: 'llzk.call' op references unknown symbol \"@invalidName\""
  );
}
