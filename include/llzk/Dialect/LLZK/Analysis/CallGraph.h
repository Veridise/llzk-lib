/**
 * The contents of this file are adapted from llvm/include/llvm/Analysis/CallGraph.h.
 */

#pragma once

#include <llvm/ADT/SCCIterator.h>
#include <llvm/ADT/STLExtras.h>
#include <cassert>
#include <map>
#include <memory>
#include <unordered_set>
#include <utility>
#include <vector>

#include <mlir/Pass/Pass.h>

#include "llzk/Dialect/LLZK/IR/Ops.h"

namespace llvm {

template <class GraphType> struct GraphTraits;
class raw_ostream;

} // namespace llvm

namespace mlir {

class Operation;
class ModuleOp;

} // namespace mlir

namespace llzk {

class CallGraphNode;

/// The basic data container for the call graph of a Module of IR.
///
/// This class exposes both the interface to the call graph for a module of IR.
///
/// The core call graph itself can also be updated to reflect changes to the IR.
class CallGraph {
  mlir::ModuleOp M;

  using FunctionMapTy =
      std::map<FuncOp, std::unique_ptr<CallGraphNode>>;

  // A map from FuncOp* to CallGraphNode*.
  FunctionMapTy FunctionMap;

  /// This node has edges to all compute/constrain functions, as those are currently
  /// our entry points until main functions are supported.
  /// Points to a "null" function.
  CallGraphNode *EntryNode;

public:
  explicit CallGraph(mlir::ModuleOp M);
  CallGraph(CallGraph &&Arg);
  ~CallGraph();

  void print(llvm::raw_ostream &OS) const;
  void dump() const;

  using iterator = FunctionMapTy::iterator;
  using const_iterator = FunctionMapTy::const_iterator;

  /// Returns the module the call graph corresponds to.
  mlir::ModuleOp &getModule() { return M; }
  const mlir::ModuleOp &getModule() const { return M; }

  inline iterator begin() { return FunctionMap.begin(); }
  inline iterator end() { return FunctionMap.end(); }
  inline const_iterator begin() const { return FunctionMap.begin(); }
  inline const_iterator end() const { return FunctionMap.end(); }
  inline size_t size() const { return FunctionMap.size(); }

  /// Returns the call graph node for the provided function.
  inline const CallGraphNode *operator[](const FuncOp &F) const {
    const_iterator I = FunctionMap.find(F);
    assert(I != FunctionMap.end() && "Function not in callgraph!");
    return I->second.get();
  }

  /// Returns the call graph node for the provided function.
  inline CallGraphNode *operator[](const FuncOp &F) {
    const_iterator I = FunctionMap.find(F);
    assert(I != FunctionMap.end() && "Function not in callgraph!");
    return I->second.get();
  }

  /**
   * Returns the node that points to all possible entry functions.
   */
  CallGraphNode *getEntryNode() const { return EntryNode; }

  //===---------------------------------------------------------------------
  // Functions to keep a call graph up to date with a function that has been
  // modified.
  //

  /// Unlink the function from this module, returning it.
  ///
  /// Because this removes the function from the module, the call graph node is
  /// destroyed.  This is only valid if the function does not call any other
  /// functions (ie, there are no edges in it's CGN).  The easiest way to do
  /// this is to dropAllReferences before calling this.
  FuncOp removeFunctionFromModule(CallGraphNode *CGN);

  /// Similar to operator[], but this will insert a new CallGraphNode for
  /// \c F if one does not already exist.
  CallGraphNode *getOrInsertFunction(FuncOp F);

  /// Populate \p CGN based on the calls inside the associated function.
  void populateCallGraphNode(CallGraphNode *CGN);

  /// Add a function to the call graph, and link the node to all of the
  /// functions that it calls.
  void addToCallGraph(FuncOp &F);
};

/// A node in the call graph for a module.
///
/// Typically represents a function in the call graph. There are also special
/// "null" nodes used to represent theoretical entries in the call graph.
class CallGraphNode {
public:
  /// A pair of the calling instruction and the call graph node being called.
  using CallRecord = std::pair<CallOp *, CallGraphNode *>;

public:
  using CalledFunctionsVector = std::vector<CallRecord>;

  /// Creates a node for the specified function.
  inline CallGraphNode(CallGraph *CG, FuncOp &F) : CG(CG), F(F) {}

  CallGraphNode(const CallGraphNode &) = delete;
  CallGraphNode &operator=(const CallGraphNode &) = delete;

  ~CallGraphNode() {
    assert(NumReferences == 0 && "Node deleted while references remain");
  }

  using iterator = std::vector<CallRecord>::iterator;
  using const_iterator = std::vector<CallRecord>::const_iterator;

  /// Returns the function that this call graph node represents.
  FuncOp &getFunction() { return F; }
  const FuncOp &getFunction() const { return F; }

  inline iterator begin() { return CalledFunctions.begin(); }
  inline iterator end() { return CalledFunctions.end(); }
  inline const_iterator begin() const { return CalledFunctions.begin(); }
  inline const_iterator end() const { return CalledFunctions.end(); }
  inline bool empty() const { return CalledFunctions.empty(); }
  inline unsigned size() const { return (unsigned)CalledFunctions.size(); }

  /// Returns the number of other CallGraphNodes in this CallGraph that
  /// reference this node in their callee list.
  unsigned getNumReferences() const { return NumReferences; }

  /// Returns the i'th called function.
  CallGraphNode *operator[](unsigned i) const {
    assert(i < CalledFunctions.size() && "Invalid index");
    return CalledFunctions[i].second;
  }

  /// Print out this call graph node.
  void dump() const;
  void print(llvm::raw_ostream &OS) const;

  //===---------------------------------------------------------------------
  // Methods to keep a call graph up to date with a function that has been
  // modified
  //

  /// Removes all edges from this CallGraphNode to any functions it calls.
  void removeAllCalledFunctions() {
    while (!CalledFunctions.empty()) {
      CalledFunctions.back().second->DropRef();
      CalledFunctions.pop_back();
    }
  }

  /// Moves all the callee information from N to this node.
  void stealCalledFunctionsFrom(CallGraphNode *N) {
    assert(CalledFunctions.empty() &&
           "Cannot steal callsite information if I already have some");
    std::swap(CalledFunctions, N->CalledFunctions);
  }

  /// Adds a function to the list of functions called by this one.
  void addCalledFunction(CallOp *Call, CallGraphNode *M) {
    CalledFunctions.emplace_back(Call, M);
    M->AddRef();
  }

  void removeCallEdge(iterator I) {
    I->second->DropRef();
    *I = CalledFunctions.back();
    CalledFunctions.pop_back();
  }

  /// Removes the edge in the node for the specified call site.
  ///
  /// Note that this method takes linear time, so it should be used sparingly.
  void removeCallEdgeFor(CallOp *Call);

  /// Removes all call edges from this node to the specified callee
  /// function.
  ///
  /// This takes more time to execute than removeCallEdgeTo, so it should not
  /// be used unless necessary.
  void removeAnyCallEdgeTo(CallGraphNode *Callee);

  /// Removes one edge associated with a null callsite from this node to
  /// the specified callee function.
  void removeOneAbstractEdgeTo(CallGraphNode *Callee);

  /// Replaces the edge in the node for the specified call site with a
  /// new one.
  ///
  /// Note that this method takes linear time, so it should be used sparingly.
  void replaceCallEdge(CallOp *Call, CallOp *NewCall,
                       CallGraphNode *NewNode);

private:
  friend class CallGraph;

  CallGraph *CG;
  FuncOp F;

  std::vector<CallRecord> CalledFunctions;

  /// The number of times that this CallGraphNode occurs in the
  /// CalledFunctions array of this or other CallGraphNodes.
  unsigned NumReferences = 0;

  void DropRef() { --NumReferences; }
  void AddRef() { ++NumReferences; }

  /// A special function that should only be used by the CallGraph class.
  void allReferencesDropped() { NumReferences = 0; }
};

/// An analysis wrapper to compute the \c CallGraph for a \c Module.
///
/// This class implements the concept of an analysis pass used by the \c
/// ModuleAnalysisManager to run an analysis over a module and cache the
/// resulting data.
class CallGraphAnalysis {
  std::unique_ptr<CallGraph> cg;

public:
  CallGraphAnalysis(mlir::Operation *op);

  CallGraph &getCallGraph() { return *cg.get(); }
  const CallGraph &getCallGraph() const { return *cg.get(); }
};

/// Pre-constructed all-pairs reachability analysis.
class CallGraphReachabilityAnalysis {

  struct FuncOpHash {
    size_t operator()(const FuncOp &op) const {
      return std::hash<mlir::Operation *>{}(const_cast<FuncOp&>(op).getOperation());
    }
  };

  // Maps function -> callees
  using CalleeMapTy =
    std::unordered_map<FuncOp, std::unordered_set<FuncOp, FuncOpHash>, FuncOpHash>;

  CalleeMapTy reachabilityMap;

  std::unordered_set<const CallGraphNode *> dfsNodes(const CallGraphNode *currNode, std::unordered_set<const CallGraphNode *> visited);

public:
  CallGraphReachabilityAnalysis(mlir::Operation *op, mlir::AnalysisManager &am);

  bool isInvalidated(const mlir::AnalysisManager::PreservedAnalyses &pa) {
    return !pa.isPreserved<CallGraphReachabilityAnalysis>() ||
           !pa.isPreserved<CallGraphAnalysis>();
  }

  /**
   * Returns whether B is reachable from A.
   */
  bool isReachable(FuncOp &A, FuncOp &B) const {
    auto it = reachabilityMap.find(A);
    return it != reachabilityMap.end() && it->second.find(B) != it->second.end();
  }
};

} // namespace llzk

//===----------------------------------------------------------------------===//
// GraphTraits specializations for call graphs so that they can be treated as
// graphs by the generic graph algorithms.
//

namespace llvm {

// Provide graph traits for traversing call graphs using standard graph
// traversals.
template <> struct GraphTraits<llzk::CallGraphNode *> {
  using NodeRef = llzk::CallGraphNode *;
  using CGNPairTy = llzk::CallGraphNode::CallRecord;

  static NodeRef getEntryNode(llzk::CallGraphNode *CGN) { return CGN; }
  static llzk::CallGraphNode *CGNGetValue(CGNPairTy P) { return P.second; }

  using ChildIteratorType =
      llvm::mapped_iterator<llzk::CallGraphNode::iterator, decltype(&CGNGetValue)>;

  static ChildIteratorType child_begin(NodeRef N) {
    return ChildIteratorType(N->begin(), &CGNGetValue);
  }

  static ChildIteratorType child_end(NodeRef N) {
    return ChildIteratorType(N->end(), &CGNGetValue);
  }
};

template <> struct llvm::GraphTraits<const llzk::CallGraphNode *> {
  using NodeRef = const llzk::CallGraphNode *;
  using CGNPairTy = llzk::CallGraphNode::CallRecord;
  using EdgeRef = const llzk::CallGraphNode::CallRecord &;

  static NodeRef getEntryNode(const llzk::CallGraphNode *CGN) { return CGN; }
  static const llzk::CallGraphNode *CGNGetValue(CGNPairTy P) { return P.second; }

  using ChildIteratorType =
      mapped_iterator<llzk::CallGraphNode::const_iterator, decltype(&CGNGetValue)>;
  using ChildEdgeIteratorType = llzk::CallGraphNode::const_iterator;

  static ChildIteratorType child_begin(NodeRef N) {
    return ChildIteratorType(N->begin(), &CGNGetValue);
  }

  static ChildIteratorType child_end(NodeRef N) {
    return ChildIteratorType(N->end(), &CGNGetValue);
  }

  static ChildEdgeIteratorType child_edge_begin(NodeRef N) {
    return N->begin();
  }
  static ChildEdgeIteratorType child_edge_end(NodeRef N) { return N->end(); }

  static NodeRef edge_dest(EdgeRef E) { return E.second; }
};

template <>
struct GraphTraits<llzk::CallGraph *> : public GraphTraits<llzk::CallGraphNode *> {
  using PairTy =
      std::pair<const llzk::FuncOp, std::unique_ptr<llzk::CallGraphNode>>;

  static NodeRef getEntryNode(llzk::CallGraph *CGN) {
    return CGN->getEntryNode();
  }

  static llzk::CallGraphNode *CGGetValuePtr(const PairTy &P) {
    return P.second.get();
  }

  // nodes_iterator/begin/end - Allow iteration over all nodes in the graph
  using nodes_iterator =
      mapped_iterator<llzk::CallGraph::iterator, decltype(&CGGetValuePtr)>;

  static nodes_iterator nodes_begin(llzk::CallGraph *CG) {
    return nodes_iterator(CG->begin(), &CGGetValuePtr);
  }

  static nodes_iterator nodes_end(llzk::CallGraph *CG) {
    return nodes_iterator(CG->end(), &CGGetValuePtr);
  }
};

template <>
struct GraphTraits<const llzk::CallGraph *> : public GraphTraits<
                                            const llzk::CallGraphNode *> {
  using PairTy =
      std::pair<const llzk::FuncOp, std::unique_ptr<llzk::CallGraphNode>>;

  static NodeRef getEntryNode(const llzk::CallGraph *CGN) {
    return CGN->getEntryNode();
  }

  static const llzk::CallGraphNode *CGGetValuePtr(const PairTy &P) {
    return P.second.get();
  }

  // nodes_iterator/begin/end - Allow iteration over all nodes in the graph
  using nodes_iterator =
      mapped_iterator<llzk::CallGraph::const_iterator, decltype(&CGGetValuePtr)>;

  static nodes_iterator nodes_begin(const llzk::CallGraph *CG) {
    return nodes_iterator(CG->begin(), &CGGetValuePtr);
  }

  static nodes_iterator nodes_end(const llzk::CallGraph *CG) {
    return nodes_iterator(CG->end(), &CGGetValuePtr);
  }
};

} // namespace llvm
