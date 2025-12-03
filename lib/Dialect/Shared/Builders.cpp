//===-- Builders.cpp - Operation builder implementations --------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/LLZK/IR/Dialect.h"
#include "llzk/Dialect/Shared/Builders.h"
#include "llzk/Util/SymbolHelper.h"

#include <llvm/Support/ErrorHandling.h>

namespace llzk {

using namespace mlir;
using namespace component;
using namespace function;

OwningOpRef<ModuleOp> createLLZKModule(MLIRContext *context, Location loc) {
  auto mod = ModuleOp::create(loc);
  addLangAttrForLLZKDialect(mod);
  return mod;
}

void addLangAttrForLLZKDialect(mlir::ModuleOp mod) {
  MLIRContext *ctx = mod.getContext();
  if (auto dialect = ctx->getOrLoadDialect<LLZKDialect>()) {
    mod->setAttr(LANG_ATTR_NAME, StringAttr::get(ctx, dialect->getNamespace()));
  } else {
    llvm::report_fatal_error("Could not load LLZK dialect!");
  }
}

/* ModuleBuilder */

void ModuleBuilder::ensureNoSuchFreeFunc(std::string_view funcName) {
  if (freeFuncMap.find(funcName) != freeFuncMap.end()) {
    auto error_message = "global function " + Twine(funcName) + " already exists!";
    llvm::report_fatal_error(error_message);
  }
}

void ModuleBuilder::ensureFreeFnExists(std::string_view funcName) {
  if (freeFuncMap.find(funcName) == freeFuncMap.end()) {
    auto error_message = "global function " + Twine(funcName) + " does not exist!";
    llvm::report_fatal_error(error_message);
  }
}

void ModuleBuilder::ensureNoSuchStruct(std::string_view structName) {
  if (structMap.find(structName) != structMap.end()) {
    auto error_message = "struct " + Twine(structName) + " already exists!";
    llvm::report_fatal_error(error_message);
  }
}

void ModuleBuilder::ensureNoSuchComputeFn(std::string_view structName) {
  if (computeFnMap.find(structName) != computeFnMap.end()) {
    auto error_message = "struct " + Twine(structName) + " already has a compute function!";
    llvm::report_fatal_error(error_message);
  }
}

void ModuleBuilder::ensureComputeFnExists(std::string_view structName) {
  if (computeFnMap.find(structName) == computeFnMap.end()) {
    auto error_message = "struct " + Twine(structName) + " has no compute function!";
    llvm::report_fatal_error(error_message);
  }
}

void ModuleBuilder::ensureNoSuchConstrainFn(std::string_view structName) {
  if (constrainFnMap.find(structName) != constrainFnMap.end()) {
    auto error_message = "struct " + Twine(structName) + " already has a constrain function!";
    llvm::report_fatal_error(error_message);
  }
}

void ModuleBuilder::ensureConstrainFnExists(std::string_view structName) {
  if (constrainFnMap.find(structName) == constrainFnMap.end()) {
    auto error_message = "struct " + Twine(structName) + " has no constrain function!";
    llvm::report_fatal_error(error_message);
  }
}

ModuleBuilder &
ModuleBuilder::insertEmptyStruct(std::string_view structName, Location loc, int numStructParams) {
  ensureNoSuchStruct(structName);

  OpBuilder opBuilder(rootModule.getBody(), rootModule.getBody()->begin());
  auto structNameAttr = StringAttr::get(context, structName);
  ArrayAttr structParams = nullptr;
  if (numStructParams >= 0) {
    SmallVector<Attribute> paramNames;
    for (int i = 0; i < numStructParams; ++i) {
      paramNames.push_back(FlatSymbolRefAttr::get(context, "T" + std::to_string(i)));
    }
    structParams = opBuilder.getArrayAttr(paramNames);
  }
  auto structDef = opBuilder.create<StructDefOp>(loc, structNameAttr, structParams);
  // populate the initial region
  auto &region = structDef.getRegion();
  (void)region.emplaceBlock();
  structMap[structName] = structDef;

  return *this;
}

ModuleBuilder &ModuleBuilder::insertComputeFn(StructDefOp op, Location loc) {
  ensureNoSuchComputeFn(op.getName());

  OpBuilder opBuilder(op.getBodyRegion());
  auto fnOp = opBuilder.create<FuncDefOp>(
      loc, StringAttr::get(context, FUNC_NAME_COMPUTE),
      FunctionType::get(context, {}, {op.getType()})
  );
  fnOp.setAllowWitnessAttr();
  fnOp.addEntryBlock();
  computeFnMap[op.getName()] = fnOp;
  return *this;
}

ModuleBuilder &ModuleBuilder::insertConstrainFn(StructDefOp op, Location loc) {
  ensureNoSuchConstrainFn(op.getName());

  OpBuilder opBuilder(op.getBodyRegion());
  auto fnOp = opBuilder.create<FuncDefOp>(
      loc, StringAttr::get(context, FUNC_NAME_CONSTRAIN),
      FunctionType::get(context, {op.getType()}, {})
  );
  fnOp.setAllowConstraintAttr();
  fnOp.addEntryBlock();
  constrainFnMap[op.getName()] = fnOp;
  return *this;
}

ModuleBuilder &
ModuleBuilder::insertComputeCall(StructDefOp caller, StructDefOp callee, Location callLoc) {
  ensureComputeFnExists(caller.getName());
  ensureComputeFnExists(callee.getName());

  auto callerFn = computeFnMap.at(caller.getName());
  auto calleeFn = computeFnMap.at(callee.getName());

  OpBuilder builder(callerFn.getBody());
  builder.create<CallOp>(callLoc, calleeFn);
  updateComputeReachability(caller, callee);
  return *this;
}

ModuleBuilder &ModuleBuilder::insertConstrainCall(
    StructDefOp caller, StructDefOp callee, Location callLoc, Location fieldDefLoc
) {
  ensureConstrainFnExists(caller.getName());
  ensureConstrainFnExists(callee.getName());

  FuncDefOp callerFn = constrainFnMap.at(caller.getName());
  FuncDefOp calleeFn = constrainFnMap.at(callee.getName());
  StructType calleeTy = callee.getType();

  size_t numOps = caller.getBody()->getOperations().size();
  auto fieldName = StringAttr::get(context, callee.getName().str() + std::to_string(numOps));

  // Insert the field declaration op
  {
    OpBuilder builder(caller.getBodyRegion());
    builder.create<FieldDefOp>(fieldDefLoc, fieldName, calleeTy);
  }

  // Insert the constrain function ops
  {
    OpBuilder builder(callerFn.getBody());

    auto field = builder.create<FieldReadOp>(
        callLoc, calleeTy, callerFn.getSelfValueFromConstrain(), fieldName
    );
    builder.create<CallOp>(
        callLoc, TypeRange {}, calleeFn.getFullyQualifiedName(), ValueRange {field}
    );
  }
  updateConstrainReachability(caller, callee);
  return *this;
}

ModuleBuilder &
ModuleBuilder::insertFreeFunc(std::string_view funcName, FunctionType type, Location loc) {
  ensureNoSuchFreeFunc(funcName);

  OpBuilder opBuilder(rootModule.getBody(), rootModule.getBody()->begin());
  auto funcDef = opBuilder.create<FuncDefOp>(loc, funcName, type);
  (void)funcDef.addEntryBlock();
  freeFuncMap[funcName] = funcDef;

  return *this;
}

ModuleBuilder &
ModuleBuilder::insertFreeCall(FuncDefOp caller, std::string_view callee, Location callLoc) {
  ensureFreeFnExists(callee);
  FuncDefOp calleeFn = freeFuncMap.at(callee);

  OpBuilder builder(caller.getBody());
  builder.create<CallOp>(callLoc, calleeFn);
  return *this;
}

} // namespace llzk
