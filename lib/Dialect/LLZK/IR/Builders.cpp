//===-- Builders.cpp - Operation builder implementations --------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/LLZK/IR/Builders.h"
#include "llzk/Dialect/LLZK/Util/SymbolHelper.h"
#include <llzk/Dialect/LLZK/Util/SymbolLookup.h>

#include <llvm/Support/ErrorHandling.h>

namespace llzk {

using namespace mlir;

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

bool isLLZKModule(mlir::ModuleOp mod) {
  MLIRContext *ctx = mod.getContext();
  if (auto dialect = ctx->getOrLoadDialect<LLZKDialect>()) {
    return mod->getAttr(LANG_ATTR_NAME) == StringAttr::get(ctx, dialect->getNamespace());
  } else {
    llvm::report_fatal_error("Could not load LLZK dialect!");
  }
}

/* ModuleBuilder */

void ModuleBuilder::ensureNoSuchGlobalFunc(std::string_view funcName) {
  if (globalFuncMap.find(funcName) != globalFuncMap.end()) {
    auto error_message = "global function " + Twine(funcName) + " already exists!";
    llvm::report_fatal_error(error_message);
  }
}

void ModuleBuilder::ensureGlobalFnExists(std::string_view funcName) {
  if (globalFuncMap.find(funcName) == globalFuncMap.end()) {
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
  auto structNameAtrr = StringAttr::get(context, structName);
  ArrayAttr structParams = nullptr;
  if (numStructParams >= 0) {
    SmallVector<Attribute> paramNames;
    for (int i = 0; i < numStructParams; ++i) {
      paramNames.push_back(FlatSymbolRefAttr::get(context, "T" + std::to_string(i)));
    }
    structParams = opBuilder.getArrayAttr(paramNames);
  }
  auto structDef = opBuilder.create<StructDefOp>(loc, structNameAtrr, structParams);
  // populate the initial region
  auto &region = structDef.getRegion();
  (void)region.emplaceBlock();
  structMap[structName] = structDef;

  return *this;
}

ModuleBuilder &ModuleBuilder::insertComputeFn(StructDefOp op, Location loc) {
  ensureNoSuchComputeFn(op.getName());

  OpBuilder opBuilder(op.getBody());

  auto fnOp = opBuilder.create<FuncOp>(
      loc, StringAttr::get(context, FUNC_NAME_COMPUTE),
      FunctionType::get(context, {}, {op.getType()})
  );
  fnOp.addEntryBlock();
  computeFnMap[op.getName()] = fnOp;
  return *this;
}

ModuleBuilder &ModuleBuilder::insertConstrainFn(StructDefOp op, Location loc) {
  ensureNoSuchConstrainFn(op.getName());

  OpBuilder opBuilder(op.getBody());

  auto fnOp = opBuilder.create<FuncOp>(
      loc, StringAttr::get(context, FUNC_NAME_CONSTRAIN),
      FunctionType::get(context, {op.getType()}, {})
  );
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
  builder.create<CallOp>(callLoc, calleeFn.getResultTypes(), calleeFn.getFullyQualifiedName());
  updateComputeReachability(caller, callee);
  return *this;
}

ModuleBuilder &ModuleBuilder::insertConstrainCall(
    StructDefOp caller, StructDefOp callee, Location callLoc, Location fieldDefLoc
) {
  ensureConstrainFnExists(caller.getName());
  ensureConstrainFnExists(callee.getName());

  auto callerFn = constrainFnMap.at(caller.getName());
  auto calleeFn = constrainFnMap.at(callee.getName());
  auto calleeTy = callee.getType();

  size_t numOps = 0;
  for (auto it = caller.getBody().begin(); it != caller.getBody().end(); it++, numOps++)
    ;
  auto fieldName = StringAttr::get(context, callee.getName().str() + std::to_string(numOps));

  // Insert the field declaration op
  {
    OpBuilder builder(caller.getBody());
    builder.create<FieldDefOp>(fieldDefLoc, fieldName, calleeTy);
  }

  // Insert the constrain function ops
  {
    OpBuilder builder(callerFn.getBody());

    auto field = builder.create<FieldReadOp>(
        callLoc, calleeTy,
        callerFn.getBody().getArgument(0), // first arg is self
        fieldName
    );
    builder.create<CallOp>(
        callLoc, TypeRange {}, calleeFn.getFullyQualifiedName(), ValueRange {field}
    );
  }
  updateConstrainReachability(caller, callee);
  return *this;
}

ModuleBuilder &
ModuleBuilder::insertGlobalFunc(std::string_view funcName, FunctionType type, Location loc) {
  ensureNoSuchGlobalFunc(funcName);

  OpBuilder opBuilder(rootModule.getBody(), rootModule.getBody()->begin());
  auto funcDef = opBuilder.create<FuncOp>(loc, funcName, type);
  (void)funcDef.addEntryBlock();
  globalFuncMap[funcName] = funcDef;

  return *this;
}

ModuleBuilder &
ModuleBuilder::insertGlobalCall(FuncOp caller, std::string_view callee, Location callLoc) {
  ensureGlobalFnExists(callee);
  FuncOp calleeFn = globalFuncMap.at(callee);

  OpBuilder builder(caller.getBody());
  builder.create<CallOp>(callLoc, calleeFn.getResultTypes(), calleeFn.getFullyQualifiedName());
  return *this;
}

} // namespace llzk
