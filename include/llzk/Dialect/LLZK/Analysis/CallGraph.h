/**
 * The contents of this file are adapted from llvm/include/llvm/Analysis/CallGraph.h.
 */

#pragma once

#include "llvm/ADT/STLExtras.h"
#include <cassert>
#include <map>
#include <memory>
#include <utility>
#include <vector>

namespace llvm {

template <class GraphType> struct GraphTraits;
class raw_ostream;

} // namespace llvm

namespace mlir {

class Operation;
class ModuleOp;

} // namespace mlir

namespace llzk {

class FuncOp;
class CallOp;
class CallGraphNode;

/// The basic data container for the call graph of a Module of IR.
///
/// This class exposes both the interface to the call graph for a module of IR.
///
/// The core call graph itself can also be updated to reflect changes to the IR.
class CallGraph {
  mlir::ModuleOp &M;

  using FunctionMapTy =
      std::map<const FuncOp *, std::unique_ptr<CallGraphNode>>;

  // A map from FuncOp* to CallGraphNode*.
  FunctionMapTy FunctionMap;

  /// This node has edges to all external functions and those internal
  /// functions that have their address taken.
  CallGraphNode *ExternalCallingNode;

  /// This node has edges to it from all functions making indirect calls
  /// or calling an external function.
  std::unique_ptr<CallGraphNode> CallsExternalNode;

public:
  explicit CallGraph(mlir::ModuleOp &M);
  CallGraph(CallGraph &&Arg);
  ~CallGraph();

  void print(llvm::raw_ostream &OS) const;
  void dump() const;

  using iterator = FunctionMapTy::iterator;
  using const_iterator = FunctionMapTy::const_iterator;

  /// Returns the module the call graph corresponds to.
  mlir::ModuleOp &getModule() const { return M; }

//   bool invalidate(mlir::ModuleOp &, const PreservedAnalyses &PA,
//                   ModuleAnalysisManager::Invalidator &);

  inline iterator begin() { return FunctionMap.begin(); }
  inline iterator end() { return FunctionMap.end(); }
  inline const_iterator begin() const { return FunctionMap.begin(); }
  inline const_iterator end() const { return FunctionMap.end(); }

  /// Returns the call graph node for the provided function.
  inline const CallGraphNode *operator[](const FuncOp *F) const {
    const_iterator I = FunctionMap.find(F);
    assert(I != FunctionMap.end() && "Function not in callgraph!");
    return I->second.get();
  }

  /// Returns the call graph node for the provided function.
  inline CallGraphNode *operator[](const FuncOp *F) {
    const_iterator I = FunctionMap.find(F);
    assert(I != FunctionMap.end() && "Function not in callgraph!");
    return I->second.get();
  }

  /// Returns the \c CallGraphNode which is used to represent
  /// undetermined calls into the callgraph.
  CallGraphNode *getExternalCallingNode() const { return ExternalCallingNode; }

  CallGraphNode *getCallsExternalNode() const {
    return CallsExternalNode.get();
  }

  /// Old node has been deleted, and New is to be used in its place, update the
  /// ExternalCallingNode.
  void ReplaceExternalCallEdge(CallGraphNode *Old, CallGraphNode *New);

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
  FuncOp *removeFunctionFromModule(CallGraphNode *CGN);

  /// Similar to operator[], but this will insert a new CallGraphNode for
  /// \c F if one does not already exist.
  CallGraphNode *getOrInsertFunction(const FuncOp *F);

  /// Populate \p CGN based on the calls inside the associated function.
  void populateCallGraphNode(CallGraphNode *CGN);

  /// Add a function to the call graph, and link the node to all of the
  /// functions that it calls.
  void addToCallGraph(FuncOp *F);
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
  inline CallGraphNode(CallGraph *CG, FuncOp *F) : CG(CG), F(F) {}

  CallGraphNode(const CallGraphNode &) = delete;
  CallGraphNode &operator=(const CallGraphNode &) = delete;

  ~CallGraphNode() {
    assert(NumReferences == 0 && "Node deleted while references remain");
  }

  using iterator = std::vector<CallRecord>::iterator;
  using const_iterator = std::vector<CallRecord>::const_iterator;

  /// Returns the function that this call graph node represents.
  FuncOp *getFunction() const { return F; }

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
  FuncOp *F;

  std::vector<CallRecord> CalledFunctions;

  /// The number of times that this CallGraphNode occurs in the
  /// CalledFunctions array of this or other CallGraphNodes.
  unsigned NumReferences = 0;

  void DropRef() { --NumReferences; }
  void AddRef() { ++NumReferences; }

  /// A special function that should only be used by the CallGraph class.
  void allReferencesDropped() { NumReferences = 0; }
};

/// TODO: Enable passes

#if 0

/// An analysis pass to compute the \c CallGraph for a \c Module.
///
/// This class implements the concept of an analysis pass used by the \c
/// ModuleAnalysisManager to run an analysis over a module and cache the
/// resulting data.
class CallGraphAnalysis : public AnalysisInfoMixin<CallGraphAnalysis> {
  friend AnalysisInfoMixin<CallGraphAnalysis>;

  static AnalysisKey Key;

public:
  /// A formulaic type to inform clients of the result type.
  using Result = CallGraph;

  /// Compute the \c CallGraph for the module \c M.
  ///
  /// The real work here is done in the \c CallGraph constructor.
  CallGraph run(Module &M, ModuleAnalysisManager &) { return CallGraph(M); }
};

/// Printer pass for the \c CallGraphAnalysis results.
class CallGraphPrinterPass : public PassInfoMixin<CallGraphPrinterPass> {
  raw_ostream &OS;

public:
  explicit CallGraphPrinterPass(raw_ostream &OS) : OS(OS) {}

  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);

  static bool isRequired() { return true; }
};

/// Printer pass for the summarized \c CallGraphAnalysis results.
class CallGraphSCCsPrinterPass
    : public PassInfoMixin<CallGraphSCCsPrinterPass> {
  raw_ostream &OS;

public:
  explicit CallGraphSCCsPrinterPass(raw_ostream &OS) : OS(OS) {}

  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);

  static bool isRequired() { return true; }
};

/// The \c ModulePass which wraps up a \c CallGraph and the logic to
/// build it.
///
/// This class exposes both the interface to the call graph container and the
/// module pass which runs over a module of IR and produces the call graph. The
/// call graph interface is entirelly a wrapper around a \c CallGraph object
/// which is stored internally for each module.
class CallGraphWrapperPass : public ModulePass {
  std::unique_ptr<CallGraph> G;

public:
  static char ID; // Class identification, replacement for typeinfo

  CallGraphWrapperPass();
  ~CallGraphWrapperPass() override;

  /// The internal \c CallGraph around which the rest of this interface
  /// is wrapped.
  const CallGraph &getCallGraph() const { return *G; }
  CallGraph &getCallGraph() { return *G; }

  using iterator = CallGraph::iterator;
  using const_iterator = CallGraph::const_iterator;

  /// Returns the module the call graph corresponds to.
  Module &getModule() const { return G->getModule(); }

  inline iterator begin() { return G->begin(); }
  inline iterator end() { return G->end(); }
  inline const_iterator begin() const { return G->begin(); }
  inline const_iterator end() const { return G->end(); }

  /// Returns the call graph node for the provided function.
  inline const CallGraphNode *operator[](const Function *F) const {
    return (*G)[F];
  }

  /// Returns the call graph node for the provided function.
  inline CallGraphNode *operator[](const Function *F) { return (*G)[F]; }

  /// Returns the \c CallGraphNode which is used to represent
  /// undetermined calls into the callgraph.
  CallGraphNode *getExternalCallingNode() const {
    return G->getExternalCallingNode();
  }

  CallGraphNode *getCallsExternalNode() const {
    return G->getCallsExternalNode();
  }

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
  FuncOp *removeFunctionFromModule(CallGraphNode *CGN) {
    return G->removeFunctionFromModule(CGN);
  }

  /// Similar to operator[], but this will insert a new CallGraphNode for
  /// \c F if one does not already exist.
  CallGraphNode *getOrInsertFunction(const Function *F) {
    return G->getOrInsertFunction(F);
  }

  //===---------------------------------------------------------------------
  // Implementation of the ModulePass interface needed here.
  //

  void getAnalysisUsage(AnalysisUsage &AU) const override;
  bool runOnModule(Module &M) override;
  void releaseMemory() override;

  void print(raw_ostream &o, const Module *) const override;
  void dump() const;
};

#endif

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
      std::pair<const llzk::FuncOp *const, std::unique_ptr<llzk::CallGraphNode>>;

  static NodeRef getEntryNode(llzk::CallGraph *CGN) {
    return CGN->getExternalCallingNode(); // Start at the external node!
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
      std::pair<const llzk::FuncOp *const, std::unique_ptr<llzk::CallGraphNode>>;

  static NodeRef getEntryNode(const llzk::CallGraph *CGN) {
    return CGN->getExternalCallingNode(); // Start at the external node!
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
