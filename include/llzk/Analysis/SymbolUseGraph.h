//===-- SymbolUseGraph.h ----------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/SymbolTable.h>

#include <llvm/ADT/GraphTraits.h>
#include <llvm/ADT/MapVector.h>
#include <llvm/ADT/SetVector.h>
#include <llvm/Support/DOTGraphTraits.h>

#include <utility>

namespace llzk {

class SymbolUseGraphNode {
  mlir::ModuleOp symbolPathRoot;
  mlir::SymbolRefAttr symbolPath;
  bool isStructConstParam;

  /* Tree structure. The SymbolUseGraph owns the nodes so just pointers here. */
  /// Predecessor: Symbol that uses the current Symbol with its defining Operation.
  mlir::SetVector<SymbolUseGraphNode *> predecessors;
  /// Successor: Symbol that is used within the current Symbol defining Operation.
  mlir::SetVector<SymbolUseGraphNode *> successors;

  SymbolUseGraphNode(mlir::ModuleOp pathRoot, mlir::SymbolRefAttr path)
      : symbolPathRoot(pathRoot), symbolPath(path), isStructConstParam(false) {
    assert(pathRoot && "'pathRoot' cannot be nullptr");
    assert(path && "'path' cannot be nullptr");
  }

  // Used only for creating the symbolic root node in the graph.
  SymbolUseGraphNode() : symbolPathRoot(nullptr), symbolPath(nullptr), isStructConstParam(false) {}
  static bool isNotSymbolicRoot(const SymbolUseGraphNode *node) {
    return node->symbolPath != nullptr;
  }

  /// Add a successor node.
  void addSuccessor(SymbolUseGraphNode *node);

  /// Remove a successor node.
  void removeSuccessor(SymbolUseGraphNode *node);

  // Provide access to private members.
  friend class SymbolUseGraph;

public:
  /// Return the root ModuleOp for the path.
  mlir::ModuleOp getSymbolPathRoot() const { return symbolPathRoot; }

  /// The symbol path+name relative to the closest root ModuleOp.
  mlir::SymbolRefAttr getSymbolPath() const { return symbolPath; }

  /// Return `true` iff the symbol is a struct constant parameter name.
  bool isStructParam() const { return isStructConstParam; }

  /// Return true if this node has any predecessors.
  bool hasPredecessor() const {
    return llvm::find_if(predecessors, isNotSymbolicRoot) != predecessors.end();
  }
  size_t numPredecessors() const { return llvm::count_if(predecessors, isNotSymbolicRoot); }

  /// Return true if this node has any successors.
  bool hasSuccessor() const {
    assert(
        llvm::find_if_not(successors, isNotSymbolicRoot) == successors.end() &&
        "symbolic root has no predecessors so it cannot appear in a successor list"
    );
    return !successors.empty();
  }
  size_t numSuccessors() const {
    assert(
        llvm::find_if_not(successors, isNotSymbolicRoot) == successors.end() &&
        "symbolic root has no predecessors so it cannot appear in a successor list"
    );
    return successors.size();
  }

  /// Iterator over predecessors/successors.
  using iterator = llvm::filter_iterator<
      mlir::SetVector<SymbolUseGraphNode *>::const_iterator, bool (*)(const SymbolUseGraphNode *)>;

  iterator predecessors_begin() const { return predecessorIter().begin(); }
  iterator predecessors_end() const { return predecessorIter().end(); }
  iterator successors_begin() const { return successorIter().begin(); }
  iterator successors_end() const { return successorIter().end(); }

  /// Range over predecessor nodes.
  inline llvm::iterator_range<iterator> predecessorIter() const {
    return llvm::make_filter_range(predecessors, isNotSymbolicRoot);
  }

  /// Range over successor nodes.
  inline llvm::iterator_range<iterator> successorIter() const {
    return llvm::make_filter_range(successors, isNotSymbolicRoot);
  }

  /// Print the node in a human readable format.
  std::string toString() const;
  void print(llvm::raw_ostream &os) const;
};

/// Builds a graph structure representing the relationships between symbols and their uses. There is
/// a node for each SymbolRef and the successors are the Symbols uses within this Symbol's defining
/// Operation.
class SymbolUseGraph {
  using NodeMapKeyT = std::pair<mlir::ModuleOp, mlir::SymbolRefAttr>;
  /// Maps Symbol operation to the (owned) SymbolUseGraphNode for that op
  using NodeMapT = llvm::MapVector<NodeMapKeyT, std::unique_ptr<SymbolUseGraphNode>>;

  /// The set of nodes within the graph.
  NodeMapT nodes;

  /// The singleton symbolic (i.e. no associated op) root node of the graph. Every newly created
  /// SymbolUseGraphNode is initially a successor of this node until a real successor (if any) is
  /// added.
  // Implementation note: An actual SymbolUseGraphNode is used instead of a list of root nodes
  // because the GraphTraits implementation requires a single root node. This node is not added to
  // the 'nodes' set and should be transparent to users of this graph.
  SymbolUseGraphNode root;

  /// An iterator over the internal graph nodes. Unwraps the map iterator to access the node.
  class NodeIterator final
      : public llvm::mapped_iterator<
            NodeMapT::const_iterator, SymbolUseGraphNode *(*)(const NodeMapT::value_type &)> {
    static SymbolUseGraphNode *unwrap(const NodeMapT::value_type &value) {
      return value.second.get();
    }

  public:
    /// Initializes the result type iterator to the specified result iterator.
    NodeIterator(NodeMapT::const_iterator it)
        : llvm::mapped_iterator<
              NodeMapT::const_iterator, SymbolUseGraphNode *(*)(const NodeMapT::value_type &)>(
              it, &unwrap
          ) {}
  };

  /// Get or add a graph node for the given symbol reference relative to the given root ModuleOp.
  SymbolUseGraphNode *getOrAddNode(
      mlir::ModuleOp pathRoot, mlir::SymbolRefAttr path, SymbolUseGraphNode *predecessorNode
  );

  SymbolUseGraphNode *getSymbolUserNode(const mlir::SymbolTable::SymbolUse &u);
  void buildGraph(mlir::SymbolOpInterface symbolOp);

  // Friend declaration for the specialization of GraphTraits
  friend struct llvm::GraphTraits<const llzk::SymbolUseGraph *>;

public:
  SymbolUseGraph(mlir::SymbolOpInterface rootSymbolOp);

  /// Return the existing node for the symbol reference relative to the given module, else nullptr.
  const SymbolUseGraphNode *lookupNode(mlir::ModuleOp pathRoot, mlir::SymbolRefAttr path) const;

  /// Return the existing node for the symbol definition op, else nullptr.
  const SymbolUseGraphNode *lookupNode(mlir::SymbolOpInterface symbolDef) const;

  /// Return the total number of nodes in the graph.
  size_t size() const { return nodes.size(); }

  /// Iterator over the root nodes (i.e. nodes that have no predecessors).
  using roots_iterator = SymbolUseGraphNode::iterator;
  roots_iterator roots_begin() const { return root.successors_begin(); }
  roots_iterator roots_end() const { return root.successors_end(); }

  /// Range over root nodes (i.e. nodes that have no predecessors).
  inline llvm::iterator_range<roots_iterator> rootsIter() const {
    return llvm::make_range(roots_begin(), roots_end());
  }

  /// An iterator over the nodes of the graph.
  using iterator = NodeIterator;
  iterator begin() const { return nodes.begin(); }
  iterator end() const { return nodes.end(); }

  /// Range over all nodes in the graph.
  inline llvm::iterator_range<iterator> nodesIter() const {
    return llvm::make_range(begin(), end());
  }

  /// Dump the graph in a human readable format.
  inline void dump() const { print(llvm::errs()); }
  void print(llvm::raw_ostream &os) const;

  /// Dump the graph to file in dot graph format.
  void dumpToDotFile(std::string filename = "") const;
};

} // namespace llzk

namespace llvm {

// Provide graph traits for traversing SymbolUseGraph using standard graph traversals.
template <> struct GraphTraits<const llzk::SymbolUseGraphNode *> {
  using NodeRef = const llzk::SymbolUseGraphNode *;
  static NodeRef getEntryNode(NodeRef node) { return node; }

  /// ChildIteratorType/begin/end - Allow iteration over all nodes in the graph.
  using ChildIteratorType = llzk::SymbolUseGraphNode::iterator;
  static ChildIteratorType child_begin(NodeRef node) { return node->successors_begin(); }
  static ChildIteratorType child_end(NodeRef node) { return node->successors_end(); }
};

template <>
struct GraphTraits<const llzk::SymbolUseGraph *>
    : public GraphTraits<const llzk::SymbolUseGraphNode *> {
  using GraphType = const llzk::SymbolUseGraph *;

  /// The entry node into the graph is the root node.
  static NodeRef getEntryNode(GraphType g) { return &g->root; }

  /// nodes_iterator/begin/end - Allow iteration over all nodes in the graph.
  using nodes_iterator = llzk::SymbolUseGraph::iterator;
  static nodes_iterator nodes_begin(GraphType g) { return g->begin(); }
  static nodes_iterator nodes_end(GraphType g) { return g->end(); }

  /// Return total number of nodes in the graph.
  static unsigned size(GraphType g) { return g->size(); }
};

// Provide graph traits for printing SymbolUseGraph using dot graph printer.
template <> struct DOTGraphTraits<const llzk::SymbolUseGraphNode *> : public DefaultDOTGraphTraits {
  using NodeRef = const llzk::SymbolUseGraphNode *;
  using GraphType = const llzk::SymbolUseGraph *;

  DOTGraphTraits(bool isSimple = false) : DefaultDOTGraphTraits(isSimple) {}

  std::string getNodeLabel(NodeRef n, GraphType) { return n->toString(); }
};

template <>
struct DOTGraphTraits<const llzk::SymbolUseGraph *>
    : public DOTGraphTraits<const llzk::SymbolUseGraphNode *> {

  DOTGraphTraits(bool isSimple = false) : DOTGraphTraits<NodeRef>(isSimple) {}

  static std::string getGraphName(GraphType) { return "Symbol Use Graph"; }

  std::string getNodeLabel(NodeRef n, GraphType g) {
    return DOTGraphTraits<NodeRef>::getNodeLabel(n, g);
  }
};

} // namespace llvm
