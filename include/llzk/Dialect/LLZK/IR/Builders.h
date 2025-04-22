//===-- Builders.h ----------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llzk/Dialect/LLZK/IR/Ops.h"

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>

#include <deque>
#include <unordered_map>

namespace llzk {

inline mlir::Location getUnknownLoc(mlir::MLIRContext *context) {
  return mlir::UnknownLoc::get(context);
}

mlir::OwningOpRef<mlir::ModuleOp> createLLZKModule(mlir::MLIRContext *context, mlir::Location loc);

inline mlir::OwningOpRef<mlir::ModuleOp> createLLZKModule(mlir::MLIRContext *context) {
  return createLLZKModule(context, getUnknownLoc(context));
}

void addLangAttrForLLZKDialect(mlir::ModuleOp mod);

bool isLLZKModule(mlir::ModuleOp mod);

/// @brief Builds out a LLZK-compliant module and provides utilities for populating
/// that module. This class is designed to be used by front-ends looking to
/// generate LLZK IR programatically and is also a useful unit testing facility.
/// TODO: this is a WIP, flesh this class out as needed.
class ModuleBuilder {
public:
  ModuleBuilder(mlir::ModuleOp m) : context(m.getContext()), rootModule(m) {}

  /* Builder methods */

  inline mlir::Location getUnknownLoc() { return llzk::getUnknownLoc(context); }

  ModuleBuilder &
  insertEmptyStruct(std::string_view structName, mlir::Location loc, int numStructParams = -1);
  inline ModuleBuilder &insertEmptyStruct(std::string_view structName, int numStructParams = -1) {
    return insertEmptyStruct(structName, getUnknownLoc(), numStructParams);
  }

  ModuleBuilder &insertComputeOnlyStruct(
      std::string_view structName, mlir::Location structLoc, mlir::Location computeLoc
  ) {
    insertEmptyStruct(structName, structLoc);
    insertComputeFn(structName, computeLoc);
    return *this;
  }

  ModuleBuilder &insertComputeOnlyStruct(std::string_view structName) {
    auto unk = getUnknownLoc();
    return insertComputeOnlyStruct(structName, unk, unk);
  }

  ModuleBuilder &insertConstrainOnlyStruct(
      std::string_view structName, mlir::Location structLoc, mlir::Location constrainLoc
  ) {
    insertEmptyStruct(structName, structLoc);
    insertConstrainFn(structName, constrainLoc);
    return *this;
  }

  ModuleBuilder &insertConstrainOnlyStruct(std::string_view structName) {
    auto unk = getUnknownLoc();
    return insertConstrainOnlyStruct(structName, unk, unk);
  }

  ModuleBuilder &insertFullStruct(
      std::string_view structName, mlir::Location structLoc, mlir::Location computeLoc,
      mlir::Location constrainLoc, int numStructParams = -1
  ) {
    insertEmptyStruct(structName, structLoc, numStructParams);
    insertComputeFn(structName, computeLoc);
    insertConstrainFn(structName, constrainLoc);
    return *this;
  }

  /// Inserts a struct with both compute and constrain functions.
  ModuleBuilder &insertFullStruct(std::string_view structName, int numStructParams = -1) {
    auto unk = getUnknownLoc();
    return insertFullStruct(structName, unk, unk, unk, numStructParams);
  }

  /**
   * compute returns the type of the struct that defines it.
   * Since this is for testing, we accept no arguments.
   */
  ModuleBuilder &insertComputeFn(llzk::StructDefOp op, mlir::Location loc);
  ModuleBuilder &insertComputeFn(std::string_view structName, mlir::Location loc) {
    return insertComputeFn(*getStruct(structName), loc);
  }
  ModuleBuilder &insertComputeFn(std::string_view structName) {
    return insertComputeFn(structName, getUnknownLoc());
  }

  /**
   * constrain accepts the struct type as the first argument.
   */
  ModuleBuilder &insertConstrainFn(llzk::StructDefOp op, mlir::Location loc);
  ModuleBuilder &insertConstrainFn(std::string_view structName, mlir::Location loc) {
    return insertConstrainFn(*getStruct(structName), getUnknownLoc());
  }
  ModuleBuilder &insertConstrainFn(std::string_view structName) {
    return insertConstrainFn(structName, getUnknownLoc());
  }

  /**
   * Only requirement for compute is the call itself.
   * It should also initialize the internal member, but we can ignore those
   * ops for the sake of testing.
   */
  ModuleBuilder &
  insertComputeCall(llzk::StructDefOp caller, llzk::StructDefOp callee, mlir::Location callLoc);
  ModuleBuilder &
  insertComputeCall(std::string_view caller, std::string_view callee, mlir::Location callLoc) {
    return insertComputeCall(*getStruct(caller), *getStruct(callee), callLoc);
  }
  ModuleBuilder &insertComputeCall(std::string_view caller, std::string_view callee) {
    return insertComputeCall(caller, callee, getUnknownLoc());
  }

  /**
   * To call a constraint function, you must:
   * 1. Add the callee as an internal member of the caller,
   * 2. Read the callee in the caller's constraint function,
   * 3. Call the callee's constraint function.
   */
  ModuleBuilder &insertConstrainCall(
      llzk::StructDefOp caller, llzk::StructDefOp callee, mlir::Location callLoc,
      mlir::Location fieldDefLoc
  );
  ModuleBuilder &insertConstrainCall(
      std::string_view caller, std::string_view callee, mlir::Location callLoc,
      mlir::Location fieldDefLoc
  ) {
    return insertConstrainCall(*getStruct(caller), *getStruct(callee), callLoc, fieldDefLoc);
  }
  ModuleBuilder &insertConstrainCall(std::string_view caller, std::string_view callee) {
    return insertConstrainCall(caller, callee, getUnknownLoc(), getUnknownLoc());
  }

  ModuleBuilder &
  insertGlobalFunc(std::string_view funcName, ::mlir::FunctionType type, mlir::Location loc);
  inline ModuleBuilder &insertGlobalFunc(std::string_view funcName, ::mlir::FunctionType type) {
    return insertGlobalFunc(funcName, type, getUnknownLoc());
  }

  ModuleBuilder &
  insertGlobalCall(llzk::FuncOp caller, std::string_view callee, mlir::Location callLoc);
  ModuleBuilder &insertGlobalCall(llzk::FuncOp caller, std::string_view callee) {
    return insertGlobalCall(caller, callee, getUnknownLoc());
  }

  /* Getter methods */

  /// Get the top-level LLZK module.
  mlir::ModuleOp &getRootModule() { return rootModule; }

  mlir::FailureOr<llzk::StructDefOp> getStruct(std::string_view structName) const {
    if (structMap.find(structName) != structMap.end()) {
      return structMap.at(structName);
    }
    return mlir::failure();
  }

  mlir::FailureOr<llzk::FuncOp> getComputeFn(std::string_view structName) const {
    if (computeFnMap.find(structName) != computeFnMap.end()) {
      return computeFnMap.at(structName);
    }
    return mlir::failure();
  }
  inline mlir::FailureOr<llzk::FuncOp> getComputeFn(llzk::StructDefOp op) const {
    return getComputeFn(op.getName());
  }

  mlir::FailureOr<llzk::FuncOp> getConstrainFn(std::string_view structName) {
    if (constrainFnMap.find(structName) != constrainFnMap.end()) {
      return constrainFnMap.at(structName);
    }
    return mlir::failure();
  }
  inline mlir::FailureOr<llzk::FuncOp> getConstrainFn(llzk::StructDefOp op) {
    return getConstrainFn(op.getName());
  }

  mlir::FailureOr<llzk::FuncOp> getGlobalFunc(std::string_view funcName) const {
    if (globalFuncMap.find(funcName) != globalFuncMap.end()) {
      return globalFuncMap.at(funcName);
    }
    return mlir::failure();
  }

  /* Helper functions */

  /**
   * Returns if the callee compute function is reachable by the caller by construction.
   */
  bool computeReachable(llzk::StructDefOp caller, llzk::StructDefOp callee) {
    return isReachable(computeNodes, caller, callee);
  }
  bool computeReachable(std::string_view caller, std::string_view callee) {
    return computeReachable(*getStruct(caller), *getStruct(callee));
  }

  /**
   * Returns if the callee compute function is reachable by the caller by construction.
   */
  bool constrainReachable(llzk::StructDefOp caller, llzk::StructDefOp callee) {
    return isReachable(constrainNodes, caller, callee);
  }
  bool constrainReachable(std::string_view caller, std::string_view callee) {
    return constrainReachable(*getStruct(caller), *getStruct(callee));
  }

private:
  mlir::MLIRContext *context;
  mlir::ModuleOp rootModule;

  struct CallNode {
    mlir::DenseMap<llzk::StructDefOp, CallNode *> callees;
  };

  using Def2NodeMap = mlir::DenseMap<llzk::StructDefOp, CallNode>;
  using StructDefSet = mlir::DenseSet<llzk::StructDefOp>;

  Def2NodeMap computeNodes, constrainNodes;

  std::unordered_map<std::string_view, llzk::FuncOp> globalFuncMap;
  std::unordered_map<std::string_view, llzk::StructDefOp> structMap;
  std::unordered_map<std::string_view, llzk::FuncOp> computeFnMap;
  std::unordered_map<std::string_view, llzk::FuncOp> constrainFnMap;

  /// @brief Ensure that a global function with the given funcName has not been added,
  /// reporting a fatal error otherwise.
  /// @param funcName
  void ensureNoSuchGlobalFunc(std::string_view funcName);

  /// @brief Ensure that a global function with the given funcName has been added,
  /// reporting a fatal error otherwise.
  /// @param funcName
  void ensureGlobalFnExists(std::string_view funcName);

  /// @brief Ensure that a struct with the given structName has not been added,
  /// reporting a fatal error otherwise.
  /// @param structName
  void ensureNoSuchStruct(std::string_view structName);

  /// @brief Ensure that the given struct does not have a compute function,
  /// reporting a fatal error otherwise.
  /// @param structName
  void ensureNoSuchComputeFn(std::string_view structName);

  /// @brief Ensure that the given struct has a compute function,
  /// reporting a fatal error otherwise.
  /// @param structName
  void ensureComputeFnExists(std::string_view structName);

  /// @brief Ensure that the given struct does not have a constrain function,
  /// reporting a fatal error otherwise.
  /// @param structName
  void ensureNoSuchConstrainFn(std::string_view structName);

  /// @brief Ensure that the given struct has a constrain function,
  /// reporting a fatal error otherwise.
  /// @param structName
  void ensureConstrainFnExists(std::string_view structName);

  void updateComputeReachability(llzk::StructDefOp caller, llzk::StructDefOp callee) {
    updateReachability(computeNodes, caller, callee);
  }

  void updateConstrainReachability(llzk::StructDefOp caller, llzk::StructDefOp callee) {
    updateReachability(constrainNodes, caller, callee);
  }

  void updateReachability(Def2NodeMap &m, llzk::StructDefOp caller, llzk::StructDefOp callee) {
    auto &callerNode = m[caller];
    auto &calleeNode = m[callee];
    callerNode.callees[callee] = &calleeNode;
  }

  bool isReachable(Def2NodeMap &m, llzk::StructDefOp caller, llzk::StructDefOp callee) {
    StructDefSet visited;
    std::deque<llzk::StructDefOp> frontier;
    frontier.push_back(caller);

    while (!frontier.empty()) {
      auto s = frontier.front();
      frontier.pop_front();
      if (!visited.insert(s).second) {
        continue;
      }

      if (s == callee) {
        return true;
      }
      for (auto &[calleeStruct, _] : m[s].callees) {
        frontier.push_back(calleeStruct);
      }
    }
    return false;
  }
};

} // namespace llzk
