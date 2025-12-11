//===-- Felt.cpp ------------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk-c/Dialect/Felt.h"
#include "llzk/CAPI/Support.h"
#include "llzk/Dialect/Felt/IR/Attrs.h"

#include <mlir-c/BuiltinAttributes.h>
#include <mlir-c/BuiltinTypes.h>
#include <mlir-c/Support.h>

#include <llvm/ADT/APInt.h>

#include "../CAPITestBase.h"

// Include the auto-generated tests
#include "llzk/Dialect/Felt/IR/Attrs.capi.test.cpp.inc"
#include "llzk/Dialect/Felt/IR/Dialect.capi.test.cpp.inc"
#include "llzk/Dialect/Felt/IR/Ops.capi.test.cpp.inc"
#include "llzk/Dialect/Felt/IR/Types.capi.test.cpp.inc"

TEST_F(CAPITest, llzk_felt_const_attr_get) {
  auto attr = llzkFeltFeltConstAttrGet(context, 0);
  EXPECT_NE(attr.ptr, (void *)NULL);
}

TEST_F(CAPITest, llzkFeltFeltConstAttrGetWithBits) {
  constexpr auto BITS = 128;
  auto attr = llzkFeltFeltConstAttrGetWithBits(context, BITS, 0);
  EXPECT_NE(attr.ptr, (void *)NULL);
  auto cxx_attr = llvm::dyn_cast<llzk::felt::FeltConstAttr>(unwrap(attr));
  EXPECT_TRUE(cxx_attr);
  auto value = cxx_attr.getValue();
  EXPECT_EQ(value.getBitWidth(), BITS);
  EXPECT_EQ(value.getZExtValue(), 0);
}

TEST_F(CAPITest, llzkFeltFeltConstAttrGetFromString) {
  constexpr auto BITS = 64;
  auto str = MlirStringRef {.data = "123", .length = 3};
  auto attr = llzkFeltFeltConstAttrGetFromString(context, BITS, str);
  EXPECT_NE(attr.ptr, (void *)NULL);
  auto expected = llzk::felt::FeltConstAttr::get(
      unwrap(context), llvm::APInt(BITS, llvm::StringRef("123", 3), 10)
  );
  EXPECT_EQ(unwrap(attr), expected);
}

TEST_F(CAPITest, llzkFeltFeltConstAttrGetFromParts) {
  constexpr auto BITS = 254;
  const uint64_t parts[] = {10, 20, 30, 40};
  auto attr = llzkFeltFeltConstAttrGetFromParts(context, BITS, parts, 4);
  EXPECT_NE(attr.ptr, (void *)NULL);
  auto expected =
      llzk::felt::FeltConstAttr::get(unwrap(context), llvm::APInt(BITS, llvm::ArrayRef(parts, 4)));
  EXPECT_EQ(unwrap(attr), expected);
}

TEST_F(CAPITest, llzk_attribute_is_a_felt_const_attr_pass) {
  auto attr = llzkFeltFeltConstAttrGet(context, 0);
  EXPECT_TRUE(llzkAttributeIsAFeltFeltConstAttr(attr));
}

TEST_F(CAPITest, llzk_felt_type_get) {
  auto type = llzkFeltFeltTypeGet(context);
  EXPECT_NE(type.ptr, (void *)NULL);
}

TEST_F(CAPITest, llzk_type_is_a_felt_type_pass) {
  auto type = llzkFeltFeltTypeGet(context);
  EXPECT_TRUE(llzkTypeIsAFeltFeltType(type));
}

// Implementation for `FeltNonDetOp_build_pass` test
std::unique_ptr<FeltNonDetOpBuildFuncHelper> FeltNonDetOpBuildFuncHelper::get() {
  struct Impl : public FeltNonDetOpBuildFuncHelper {
    MlirOperation
    callBuild(const CAPITest &testClass, MlirOpBuilder builder, MlirLocation location) override {
      return llzkFeltFeltNonDetOpBuild(builder, location);
    }
  };
  return std::make_unique<Impl>();
}

// Implementation for `FeltConstantOp_build_pass` test
std::unique_ptr<FeltConstantOpBuildFuncHelper> FeltConstantOpBuildFuncHelper::get() {
  struct Impl : public FeltConstantOpBuildFuncHelper {
    MlirOperation
    callBuild(const CAPITest &testClass, MlirOpBuilder builder, MlirLocation location) override {
      // Use C++ API to avoid indirectly testing other LLZK C API functions here.
      auto attr = llzk::felt::FeltConstAttr::get(unwrap(testClass.context), llvm::APInt());
      return llzkFeltFeltConstantOpBuild(builder, location, wrap(attr));
    }
  };
  return std::make_unique<Impl>();
}

// Implementation for `OrFeltOp_build_pass` test
std::unique_ptr<OrFeltOpBuildFuncHelper> OrFeltOpBuildFuncHelper::get() {
  struct Impl : public OrFeltOpBuildFuncHelper {
    mlir::OwningOpRef<mlir::ModuleOp> parentModule;
    MlirOperation
    callBuild(const CAPITest &testClass, MlirOpBuilder builder, MlirLocation location) override {
      // Use "@compute" function as parent to avoid the following:
      // error: 'felt.bit_or' op only valid within a 'function.def' with 'function.allow_witness'
      this->parentModule = testClass.cppGenStructAndSetInsertionPoint(
          builder, location, llzk::function::FunctionKind::StructCompute
      );
      auto val = testClass.cppGenFeltConstant(builder, location);
      return llzkFeltOrFeltOpBuild(builder, location, wrap(val), wrap(val));
    }
  };
  return std::make_unique<Impl>();
}

// Implementation for `AndFeltOp_build_pass` test
std::unique_ptr<AndFeltOpBuildFuncHelper> AndFeltOpBuildFuncHelper::get() {
  struct Impl : public AndFeltOpBuildFuncHelper {
    mlir::OwningOpRef<mlir::ModuleOp> parentModule;
    MlirOperation
    callBuild(const CAPITest &testClass, MlirOpBuilder builder, MlirLocation location) override {
      // Use "@compute" function as parent to avoid the following:
      // error: 'felt.bit_and' op only valid within a 'function.def' with 'function.allow_witness'
      this->parentModule = testClass.cppGenStructAndSetInsertionPoint(
          builder, location, llzk::function::FunctionKind::StructCompute
      );
      auto val = testClass.cppGenFeltConstant(builder, location);
      return llzkFeltAndFeltOpBuild(builder, location, wrap(val), wrap(val));
    }
  };
  return std::make_unique<Impl>();
}

// Implementation for `XorFeltOp_build_pass` test
std::unique_ptr<XorFeltOpBuildFuncHelper> XorFeltOpBuildFuncHelper::get() {
  struct Impl : public XorFeltOpBuildFuncHelper {
    mlir::OwningOpRef<mlir::ModuleOp> parentModule;
    MlirOperation
    callBuild(const CAPITest &testClass, MlirOpBuilder builder, MlirLocation location) override {
      // Use "@compute" function as parent to avoid the following:
      // error: 'felt.bit_xor' op only valid within a 'function.def' with 'function.allow_witness'
      this->parentModule = testClass.cppGenStructAndSetInsertionPoint(
          builder, location, llzk::function::FunctionKind::StructCompute
      );
      auto val = testClass.cppGenFeltConstant(builder, location);
      return llzkFeltXorFeltOpBuild(builder, location, wrap(val), wrap(val));
    }
  };
  return std::make_unique<Impl>();
}

// Implementation for `NotFeltOp_build_pass` test
std::unique_ptr<NotFeltOpBuildFuncHelper> NotFeltOpBuildFuncHelper::get() {
  struct Impl : public NotFeltOpBuildFuncHelper {
    mlir::OwningOpRef<mlir::ModuleOp> parentModule;
    MlirOperation
    callBuild(const CAPITest &testClass, MlirOpBuilder builder, MlirLocation location) override {
      // Use "@compute" function as parent to avoid the following:
      // error: 'felt.bit_not' op only valid within a 'function.def' with 'function.allow_witness'
      this->parentModule = testClass.cppGenStructAndSetInsertionPoint(
          builder, location, llzk::function::FunctionKind::StructCompute
      );
      auto val = testClass.cppGenFeltConstant(builder, location);
      return llzkFeltNotFeltOpBuild(builder, location, wrap(val));
    }
  };
  return std::make_unique<Impl>();
}

// Implementation for `ShlFeltOp_build_pass` test
std::unique_ptr<ShlFeltOpBuildFuncHelper> ShlFeltOpBuildFuncHelper::get() {
  struct Impl : public ShlFeltOpBuildFuncHelper {
    mlir::OwningOpRef<mlir::ModuleOp> parentModule;
    MlirOperation
    callBuild(const CAPITest &testClass, MlirOpBuilder builder, MlirLocation location) override {
      // Use "@compute" function as parent to avoid the following:
      // error: 'felt.shl' op only valid within a 'function.def' with 'function.allow_witness'
      this->parentModule = testClass.cppGenStructAndSetInsertionPoint(
          builder, location, llzk::function::FunctionKind::StructCompute
      );
      auto val = testClass.cppGenFeltConstant(builder, location);
      return llzkFeltShlFeltOpBuild(builder, location, wrap(val), wrap(val));
    }
  };
  return std::make_unique<Impl>();
}

// Implementation for `ShrFeltOp_build_pass` test
std::unique_ptr<ShrFeltOpBuildFuncHelper> ShrFeltOpBuildFuncHelper::get() {
  struct Impl : public ShrFeltOpBuildFuncHelper {
    mlir::OwningOpRef<mlir::ModuleOp> parentModule;
    MlirOperation
    callBuild(const CAPITest &testClass, MlirOpBuilder builder, MlirLocation location) override {
      // Use "@compute" function as parent to avoid the following:
      // error: 'felt.shr' op only valid within a 'function.def' with 'function.allow_witness'
      this->parentModule = testClass.cppGenStructAndSetInsertionPoint(
          builder, location, llzk::function::FunctionKind::StructCompute
      );
      auto val = testClass.cppGenFeltConstant(builder, location);
      return llzkFeltShrFeltOpBuild(builder, location, wrap(val), wrap(val));
    }
  };
  return std::make_unique<Impl>();
}

// Implementation for `AddFeltOp_build_pass` test
std::unique_ptr<AddFeltOpBuildFuncHelper> AddFeltOpBuildFuncHelper::get() {
  struct Impl : public AddFeltOpBuildFuncHelper {
    MlirOperation
    callBuild(const CAPITest &testClass, MlirOpBuilder builder, MlirLocation location) override {
      auto val = testClass.cppGenFeltConstant(builder, location);
      return llzkFeltAddFeltOpBuild(builder, location, wrap(val), wrap(val));
    }
  };
  return std::make_unique<Impl>();
}

// Implementation for `SubFeltOp_build_pass` test
std::unique_ptr<SubFeltOpBuildFuncHelper> SubFeltOpBuildFuncHelper::get() {
  struct Impl : public SubFeltOpBuildFuncHelper {
    MlirOperation
    callBuild(const CAPITest &testClass, MlirOpBuilder builder, MlirLocation location) override {
      auto val = testClass.cppGenFeltConstant(builder, location);
      return llzkFeltSubFeltOpBuild(builder, location, wrap(val), wrap(val));
    }
  };
  return std::make_unique<Impl>();
}

// Implementation for `DivFeltOp_build_pass` test
std::unique_ptr<DivFeltOpBuildFuncHelper> DivFeltOpBuildFuncHelper::get() {
  struct Impl : public DivFeltOpBuildFuncHelper {
    MlirOperation
    callBuild(const CAPITest &testClass, MlirOpBuilder builder, MlirLocation location) override {
      auto val = testClass.cppGenFeltConstant(builder, location);
      return llzkFeltDivFeltOpBuild(builder, location, wrap(val), wrap(val));
    }
  };
  return std::make_unique<Impl>();
}

// Implementation for `InvFeltOp_build_pass` test
std::unique_ptr<InvFeltOpBuildFuncHelper> InvFeltOpBuildFuncHelper::get() {
  struct Impl : public InvFeltOpBuildFuncHelper {
    mlir::OwningOpRef<mlir::ModuleOp> parentModule;
    MlirOperation
    callBuild(const CAPITest &testClass, MlirOpBuilder builder, MlirLocation location) override {
      // Use "@compute" function as parent to avoid the following:
      // error: 'felt.inv' op only valid within a 'function.def' with 'function.allow_witness'
      this->parentModule = testClass.cppGenStructAndSetInsertionPoint(
          builder, location, llzk::function::FunctionKind::StructCompute
      );
      auto val = testClass.cppGenFeltConstant(builder, location);
      return llzkFeltInvFeltOpBuild(builder, location, wrap(val));
    }
  };
  return std::make_unique<Impl>();
}

// Implementation for `ModFeltOp_build_pass` test
std::unique_ptr<ModFeltOpBuildFuncHelper> ModFeltOpBuildFuncHelper::get() {
  struct Impl : public ModFeltOpBuildFuncHelper {
    MlirOperation
    callBuild(const CAPITest &testClass, MlirOpBuilder builder, MlirLocation location) override {
      auto val = testClass.cppGenFeltConstant(builder, location);
      return llzkFeltModFeltOpBuild(builder, location, wrap(val), wrap(val));
    }
  };
  return std::make_unique<Impl>();
}

// Implementation for `MulFeltOp_build_pass` test
std::unique_ptr<MulFeltOpBuildFuncHelper> MulFeltOpBuildFuncHelper::get() {
  struct Impl : public MulFeltOpBuildFuncHelper {
    MlirOperation
    callBuild(const CAPITest &testClass, MlirOpBuilder builder, MlirLocation location) override {
      auto val = testClass.cppGenFeltConstant(builder, location);
      return llzkFeltMulFeltOpBuild(builder, location, wrap(val), wrap(val));
    }
  };
  return std::make_unique<Impl>();
}

// Implementation for `NegFeltOp_build_pass` test
std::unique_ptr<NegFeltOpBuildFuncHelper> NegFeltOpBuildFuncHelper::get() {
  struct Impl : public NegFeltOpBuildFuncHelper {
    MlirOperation
    callBuild(const CAPITest &testClass, MlirOpBuilder builder, MlirLocation location) override {
      auto val = testClass.cppGenFeltConstant(builder, location);
      return llzkFeltNegFeltOpBuild(builder, location, wrap(val));
    }
  };
  return std::make_unique<Impl>();
}
