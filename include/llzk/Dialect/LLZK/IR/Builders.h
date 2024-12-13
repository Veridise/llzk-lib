#pragma once

#include "llzk/Dialect/LLZK/IR/Ops.h"
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>

#include <deque>
#include <unordered_map>
#include <unordered_set>

namespace llzk {

/// @brief Builds a LLZK-compliant module and provides utilities for populating
/// that module. This class is designed to be used by front-ends looking to
/// generate LLZK IR programatically and is also a useful unit testing facility.
/// TODO: this is a WIP, flesh this class out as needed.
class ModuleBuilder {
public:
  ModuleBuilder(mlir::MLIRContext *context);

  /* Builder methods */

  llzk::StructDefOp insertEmptyStruct(std::string_view structName);

  /* Getter methods */

  /// Get the top-level LLZK module.
  mlir::ModuleOp &getRootModule() { return rootModule; }

  llzk::StructDefOp insertComputeOnlyStruct(std::string_view structName) {
    auto s = insertEmptyStruct(structName);
    insertComputeFn(&s);
    return s;
  }

  llzk::StructDefOp insertConstrainOnlyStruct(std::string_view structName) {
    auto s = insertEmptyStruct(structName);
    insertConstrainFn(&s);
    return s;
  }

  llzk::StructDefOp insertFullStruct(std::string_view structName) {
    auto s = insertEmptyStruct(structName);
    insertComputeFn(&s);
    insertConstrainFn(&s);
    return s;
  }

  llzk::StructDefOp getStruct(std::string_view structName) { return structMap.at(structName); }

  /**
   * compute returns the type of the struct that defines it.
   * Since this is for testing, we accept no arguments.
   */
  llzk::FuncOp insertComputeFn(llzk::StructDefOp *op);

  llzk::FuncOp getComputeFn(llzk::StructDefOp *op) { return computeFnMap.at(op->getName()); }

  /**
   * constrain accepts the struct type as the first argument.
   */
  llzk::FuncOp insertConstrainFn(llzk::StructDefOp *op);

  llzk::FuncOp getConstrainFn(llzk::StructDefOp *op) { return constrainFnMap.at(op->getName()); }

  /**
   * Only requirement for compute is the call itself.
   * It should also initialize the internal member, but we can ignore those
   * ops for the sake of testing.
   */
  ModuleBuilder &insertComputeCall(llzk::StructDefOp *caller, llzk::StructDefOp *callee);

  /**
   * To call a constraint function, you must:
   * 1. Add the callee as an internal member of the caller,
   * 2. Read the callee in the caller's constraint function,
   * 3. Call the callee's constraint function.
   */
  ModuleBuilder &insertConstrainCall(llzk::StructDefOp *caller, llzk::StructDefOp *callee);

  /**
   * Returns if the callee compute function is reachable by the caller by construction.
   */
  bool computeReachable(llzk::StructDefOp *caller, llzk::StructDefOp *callee) {
    return isReachable(computeNodes, caller, callee);
  }

  /**
   * Returns if the callee compute function is reachable by the caller by construction.
   */
  bool constrainReachable(llzk::StructDefOp *caller, llzk::StructDefOp *callee) {
    return isReachable(constrainNodes, caller, callee);
  }

private:
  mlir::MLIRContext *context;
  mlir::ModuleOp rootModule;

  struct CallNode {
    std::unordered_map<llzk::StructDefOp *, CallNode *> callees;
  };

  std::unordered_map<llzk::StructDefOp *, CallNode> computeNodes, constrainNodes;

  std::unordered_map<std::string_view, llzk::StructDefOp> structMap;
  std::unordered_map<std::string_view, llzk::FuncOp> computeFnMap;
  std::unordered_map<std::string_view, llzk::FuncOp> constrainFnMap;

  void updateComputeReachability(llzk::StructDefOp *caller, llzk::StructDefOp *callee) {
    updateReachability(computeNodes, caller, callee);
  }

  void updateConstrainReachability(llzk::StructDefOp *caller, llzk::StructDefOp *callee) {
    updateReachability(constrainNodes, caller, callee);
  }

  void updateReachability(
      std::unordered_map<llzk::StructDefOp *, CallNode> &m, llzk::StructDefOp *caller,
      llzk::StructDefOp *callee
  ) {
    auto &callerNode = m[caller];
    auto &calleeNode = m[callee];
    callerNode.callees[callee] = &calleeNode;
  }

  bool isReachable(
      std::unordered_map<llzk::StructDefOp *, CallNode> &m, llzk::StructDefOp *caller,
      llzk::StructDefOp *callee
  ) {
    std::unordered_set<llzk::StructDefOp *> visited;
    std::deque<llzk::StructDefOp *> frontier;
    frontier.push_back(caller);

    while (!frontier.empty()) {
      auto *s = frontier.front();
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