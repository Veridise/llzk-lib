//===-- SymbolDefTree.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include <mlir/IR/SymbolTable.h>

#include <llvm/ADT/GraphTraits.h>
#include <llvm/ADT/MapVector.h>
#include <llvm/ADT/SetVector.h>
#include <llvm/Support/DOTGraphTraits.h>

namespace llzk {

class SymbolDefTreeNode {
  // The Symbol operation referenced by this node.
  mlir::SymbolOpInterface symbolDef;

  /* Tree structure. The SymbolDefTree owns the nodes so just pointers here. */
  SymbolDefTreeNode *parent;
  mlir::SetVector<SymbolDefTreeNode *> children;

  SymbolDefTreeNode(mlir::SymbolOpInterface symbolDefOp) : symbolDef(symbolDefOp), parent(nullptr) {
    assert(symbolDef && "must have a SymbolOpInterface node");
  }

  // Used only for creating the root node in the tree.
  SymbolDefTreeNode() : symbolDef(nullptr), parent(nullptr) {}

  // Provide access to private members.
  friend class SymbolDefTree;

public:
  /// Returns the Symbol operation referenced by this node.
  mlir::SymbolOpInterface getOp() const { return symbolDef; }

  /// Returns true if this node is a root for symbol lookup. See `llzk::getRootModule()`.
  bool isLookupRoot() const;

  /// Add a child node, i.e. a Symbol defined in `this` SymbolTable.
  void addChild(SymbolDefTreeNode *node);

  /// Returns true if this node has any child edges.
  bool hasChildren() const { return !children.empty(); }

  /// Iterator over the children of this node.
  using child_iterator = mlir::SetVector<SymbolDefTreeNode *>::const_iterator;
  child_iterator begin() const { return children.begin(); }
  child_iterator end() const { return children.end(); }

  /// Range over child nodes.
  inline llvm::iterator_range<child_iterator> childIter() const {
    return llvm::make_range(begin(), end());
  }

  /// Print the node in a human readable format.
  std::string toString() const;
  void print(llvm::raw_ostream &os) const;
};

/// Builds a tree structure representing the symbol table structure. There is a node for each Symbol
/// Operation and the parent is the SymbolTable that defines the Symbol.
class SymbolDefTree {
  /// Maps Symbol operation to the (owned) SymbolDefTreeNode for that op
  using NodeMapT = llvm::MapVector<mlir::SymbolOpInterface, std::unique_ptr<SymbolDefTreeNode>>;

  /// The set of nodes within the tree.
  NodeMapT nodes;

  /// The singleton symbolic (i.e. no associated op) root node of the tree.
  SymbolDefTreeNode root;

  /// An iterator over the internal tree nodes. Unwraps the map iterator to access the node.
  class NodeIterator final
      : public llvm::mapped_iterator<
            NodeMapT::const_iterator, SymbolDefTreeNode *(*)(const NodeMapT::value_type &)> {
    static SymbolDefTreeNode *unwrap(const NodeMapT::value_type &value) {
      return value.second.get();
    }

  public:
    /// Initializes the result type iterator to the specified result iterator.
    NodeIterator(NodeMapT::const_iterator it)
        : llvm::mapped_iterator<
              NodeMapT::const_iterator, SymbolDefTreeNode *(*)(const NodeMapT::value_type &)>(
              it, &unwrap
          ) {}
  };

  /// Get or add a tree node for the given symbol def op. `parentNode` is the node containing the
  /// SymbolTable for the given symbol, or nullptr if there is no parent node.
  SymbolDefTreeNode *getOrAddNode(mlir::SymbolOpInterface symbolDef, SymbolDefTreeNode *parentNode);

  void buildTree(mlir::SymbolOpInterface symbolOp, SymbolDefTreeNode *parentNode);

public:
  SymbolDefTree(mlir::SymbolOpInterface root);

  /// Lookup the node for the given symbol Op, or nullptr if none exists.
  const SymbolDefTreeNode *lookupNode(mlir::SymbolOpInterface symbolOp) const;

  /// Returns the symbolic (i.e. no associated op) root node of the tree.
  const SymbolDefTreeNode *getRoot() const { return &root; }

  /// Return total number of nodes in the tree.
  size_t size() const { return nodes.size(); }

  /// An iterator over the nodes of the tree.
  using iterator = NodeIterator;
  iterator begin() const { return nodes.begin(); }
  iterator end() const { return nodes.end(); }

  inline llvm::iterator_range<iterator> nodeIter() const {
    return llvm::make_range(begin(), end());
  }

  /// Dump the tree in a human readable format.
  inline void dump() const { print(llvm::errs()); }
  void print(llvm::raw_ostream &os) const;
};

} // namespace llzk

namespace llvm {
// Provide graph traits for traversing SymbolDefTree using standard graph traversals.

template <> struct GraphTraits<const llzk::SymbolDefTreeNode *> {
  using NodeRef = const llzk::SymbolDefTreeNode *;
  static NodeRef getEntryNode(NodeRef node) { return node; }

  /// ChildIteratorType/begin/end - Allow iteration over all nodes in the graph.
  using ChildIteratorType = llzk::SymbolDefTreeNode::child_iterator;
  static ChildIteratorType child_begin(NodeRef node) { return node->begin(); }
  static ChildIteratorType child_end(NodeRef node) { return node->end(); }
};

template <>
struct GraphTraits<const llzk::SymbolDefTree *>
    : public GraphTraits<const llzk::SymbolDefTreeNode *> {

  /// The entry node into the graph is the external node.
  static NodeRef getEntryNode(const llzk::SymbolDefTree *g) { return g->getRoot(); }

  /// nodes_iterator/begin/end - Allow iteration over all nodes in the graph.
  using nodes_iterator = llzk::SymbolDefTree::iterator;
  static nodes_iterator nodes_begin(const llzk::SymbolDefTree *g) { return g->begin(); }
  static nodes_iterator nodes_end(const llzk::SymbolDefTree *g) { return g->end(); }

  /// Return total number of nodes in the graph.
  static unsigned size(const llzk::SymbolDefTree *g) { return g->size(); }
};

// Provide graph traits for printing SymbolDefTree using dot graph printer.
template <> struct DOTGraphTraits<const llzk::SymbolDefTreeNode *> : public DefaultDOTGraphTraits {

  DOTGraphTraits(bool isSimple = false) : DefaultDOTGraphTraits(isSimple) {}

  std::string getNodeLabel(const llzk::SymbolDefTreeNode *n, const llzk::SymbolDefTreeNode *) {
    return n->toString();
  }
};

template <>
struct DOTGraphTraits<const llzk::SymbolDefTree *>
    : public DOTGraphTraits<const llzk::SymbolDefTreeNode *> {

  DOTGraphTraits(bool isSimple = false)
      : DOTGraphTraits<const llzk::SymbolDefTreeNode *>(isSimple) {}

  static std::string getGraphName(const llzk::SymbolDefTree *) { return "Symbol Def Tree"; }

  std::string getNodeLabel(const llzk::SymbolDefTreeNode *n, const llzk::SymbolDefTree *g) {
    return DOTGraphTraits<const llzk::SymbolDefTreeNode *>::getNodeLabel(n, g->getRoot());
  }
};

} // namespace llvm
