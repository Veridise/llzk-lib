//===-- SymbolDefTree.cpp ---------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Analysis/SymbolDefTree.h"
#include "llzk/Util/Constants.h"
#include "llzk/Util/SymbolTableLLZK.h"

#include <mlir/IR/BuiltinOps.h>

using namespace mlir;

namespace llzk {

//===----------------------------------------------------------------------===//
// SymbolDefTreeNode
//===----------------------------------------------------------------------===//

bool SymbolDefTreeNode::isLookupRoot() const {
  return llvm::isa<ModuleOp>(symbolDef) && symbolDef->hasAttr(LANG_ATTR_NAME);
}

void SymbolDefTreeNode::addChild(SymbolDefTreeNode *node) {
  assert(!node->parent && "def cannot be in more than one symbol table");
  node->parent = this;
  children.insert(node);
}

//===----------------------------------------------------------------------===//
// SymbolDefTree
//===----------------------------------------------------------------------===//

SymbolDefTree::SymbolDefTree(SymbolOpInterface root) {
  assert(root->hasTrait<OpTrait::SymbolTable>());
  buildTree(root, /*parentNode=*/nullptr);
}

void SymbolDefTree::buildTree(SymbolOpInterface symbolOp, SymbolDefTreeNode *parentNode) {
  // Add node for the current symbol
  parentNode = getOrAddNode(symbolOp, parentNode);
  // If this symbol is also its own SymbolTable, recursively add child symbols
  if (symbolOp->hasTrait<OpTrait::SymbolTable>()) {
    for (Operation &op : symbolOp->getRegion(0).front()) {
      if (SymbolOpInterface childSym = llvm::dyn_cast<SymbolOpInterface>(&op)) {
        buildTree(childSym, parentNode);
      }
    }
  }
}

SymbolDefTreeNode *
SymbolDefTree::getOrAddNode(SymbolOpInterface symbolDef, SymbolDefTreeNode *parentNode) {
  std::unique_ptr<SymbolDefTreeNode> &node = nodes[symbolDef];
  if (!node) {
    node.reset(new SymbolDefTreeNode(symbolDef));
    // Add this node to the given parent node if given, else the root node.
    if (parentNode) {
      parentNode->addChild(node.get());
    } else {
      root.addChild(node.get());
    }
  }
  return node.get();
}

const SymbolDefTreeNode *SymbolDefTree::lookupNode(SymbolOpInterface region) const {
  const auto *it = nodes.find(region);
  return it == nodes.end() ? nullptr : it->second.get();
}

//===----------------------------------------------------------------------===//
// Printing

void SymbolDefTree::print(llvm::raw_ostream &os) const {
  std::function<void(SymbolDefTreeNode *)> printNode = [&os, &printNode](SymbolDefTreeNode *node) {
    // Print the current node
    SymbolOpInterface sym = node->symbolDef;
    os << "// - Node : [" << node << "] '" << sym->getName() << "' ";
    if (StringAttr name = llzk::getSymbolName(sym)) {
      os << "named " << name << '\n';
    } else {
      os << "without a name\n";
    }
    // Print list of IDs for the children
    os << "// --- Children : [";
    llvm::interleaveComma(node->children, os);
    os << "]\n";
    // Recursively print the children
    for (SymbolDefTreeNode *c : node->children) {
      printNode(c);
    }
  };

  os << "// ---- SymbolDefTree ----\n";
  for (SymbolDefTreeNode *r : root.children) {
    printNode(r);
  }
  os << "// -------------------\n";
}

} // namespace llzk
