#include "llzk/Dialect/LLZK/IR/Ops.h"
#include "llzk/Dialect/LLZK/Transforms/LLZKTransformationPasses.h"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseMapInfo.h>
#include <memory>
#include <deque>

/// Include the generated base pass class definitions.
namespace llzk {
#define GEN_PASS_DEF_REDUNDANTREADANDWRITEELIMINATIONPASS
#include "llzk/Dialect/LLZK/Transforms/LLZKTransformationPasses.h.inc"
} // namespace llzk

using namespace mlir;
using namespace llzk;

namespace {

/// @brief An reference to a value, represented either by an SSA value, a
/// symbol reference (e.g., a field name), or an int (e.g., a constant array index).
class ReferenceID {
public:
  explicit ReferenceID(Value v) {
    // reserved special pointer values for DenseMapInfo
    if (v.getImpl() == reinterpret_cast<mlir::detail::ValueImpl*>(1) || v.getImpl() == reinterpret_cast<mlir::detail::ValueImpl*>(2)) {
      identifier = v;
    } else if (auto constVal = dyn_cast_if_present<FeltConstantOp>(v.getDefiningOp())) {
      identifier = constVal.getValue().getValue();
    } else if (auto constIdxVal = dyn_cast_if_present<arith::ConstantIndexOp>(v.getDefiningOp())) {
      identifier = APInt(64, constIdxVal.value());
    } else {
      identifier = v;
    }
  }
  explicit ReferenceID(FlatSymbolRefAttr s) : identifier(s) {}
  explicit ReferenceID(llvm::APInt i) : identifier(i) {}
  explicit ReferenceID(unsigned i) : identifier(llvm::APInt(64, i)) {}

  bool isValue() const {
    return std::holds_alternative<Value>(identifier);
  }
  bool isSymbol() const {
    return std::holds_alternative<FlatSymbolRefAttr>(identifier);
  }
  bool isConst() const {
    return std::holds_alternative<APInt>(identifier);
  }

  Value getValue() const {
    ensure(isValue(), "does not hold Value");
    return std::get<Value>(identifier);
  }

  FlatSymbolRefAttr getSymbol() const {
    ensure(isSymbol(), "does not hold symbol");
    return std::get<FlatSymbolRefAttr>(identifier);
  }

  APInt getConst() const {
    ensure(isConst(), "does not hold const");
    return std::get<APInt>(identifier);
  }

  void print(raw_ostream &os) const {
    if (auto v = std::get_if<Value>(&identifier)) {
      if (auto opres = dyn_cast<OpResult>(*v)) {
        os << '%' << opres.getResultNumber();
      } else {
        os << *v;
      }
    } else if (auto s = std::get_if<FlatSymbolRefAttr>(&identifier)) {
      os << *s;
    } else {
      os << std::get<APInt>(identifier);
    }
  }

  friend bool operator==(const ReferenceID &lhs, const ReferenceID &rhs) {
    return lhs.identifier == rhs.identifier;
  }

  friend raw_ostream &operator<<(raw_ostream &os, const ReferenceID &id) {
    id.print(os);
    return os;
  }

private:
  std::variant<Value, FlatSymbolRefAttr, APInt> identifier;
};

}

namespace llvm {

template <> struct DenseMapInfo<ReferenceID> {
  static ReferenceID getEmptyKey() {
    return ReferenceID(Value(reinterpret_cast<mlir::detail::ValueImpl*>(1)));
  }
  static inline ReferenceID getTombstoneKey() {
    return ReferenceID(Value(reinterpret_cast<mlir::detail::ValueImpl*>(2)));
  }
  static unsigned getHashValue(const ReferenceID &r) {
    if (r.isValue()) {
      return hash_value(r.getValue());
    } else if (r.isSymbol()) {
      return hash_value(r.getSymbol());
    }
    return hash_value(r.getConst());
  }
  static bool isEqual(const ReferenceID &lhs, const ReferenceID &rhs) {
    return lhs == rhs;
  }
};

}

namespace {

/// @brief
/// Does not allow mixing of constant and non-constant child indices, as we
/// do not know if they alias.
class ReferenceNode {
public:
  template<typename IdType>
  static std::shared_ptr<ReferenceNode> create(IdType id, Value v, ReferenceNode *parent = nullptr) {
    return std::make_shared<ReferenceNode>(parent, id, v);
  }

  template<typename IdType>
  ReferenceNode(ReferenceNode *parentNode, IdType id, Value initialVal)
  : identifier(id), storedValue(initialVal), lastWrite(nullptr), parent(parentNode), children(), childrenContainValueRefs(false) {}

  template<typename IdType>
  std::shared_ptr<ReferenceNode> createChild(IdType id, Value storedVal, std::shared_ptr<ReferenceNode> valTree = nullptr) {
    auto child = create(id, storedVal, this);
    child->setCurrentValue(storedVal, valTree);
    if (child->identifier.isValue()) {
      childrenContainValueRefs = true;
    }
    children[child->identifier] = child;
    return child;
  }

  template<typename IdType>
  std::shared_ptr<ReferenceNode> getChild(IdType id) const {
    auto it = children.find(ReferenceID(id));
    if (it != children.end()) {
      return it->second;
    }
    return nullptr;
  }

  template<typename IdType>
  std::shared_ptr<ReferenceNode> getOrCreateChild(IdType id, Value storedVal = nullptr) {
    auto it = children.find(ReferenceID(id));
    if (it != children.end()) {
      return it->second;
    }
    return createChild(id, storedVal);
  }

  Operation *updateLastWrite(Operation *writeOp) {
    auto old = lastWrite;
    lastWrite = writeOp;
    return old;
  }

  void setCurrentValue(Value v, std::shared_ptr<ReferenceNode> valTree = nullptr) {
    storedValue = v;
    if (valTree != nullptr) {
      // Overwrite our current set of children with new children, since we overwrote
      // the stored value
      children = valTree->children;
      childrenContainValueRefs = valTree->childrenContainValueRefs;
    }
  }

  bool isLeaf() const {
    return children.empty();
  }

  Value getStoredValue() const { return storedValue; }

  bool hasStoredValue() const { return storedValue != nullptr; }

  bool childrenMayAlias() const { return childrenContainValueRefs; }

  void print(raw_ostream &os) const {
    if (parent != nullptr) {
      parent->print(os);
    }
    os << '[' << identifier << " => " << storedValue << ']';
  }

  friend raw_ostream &operator<<(raw_ostream &os, const ReferenceNode &r) {
    r.print(os);
    return os;
  }

private:
  ReferenceID identifier;
  mlir::Value storedValue;
  Operation *lastWrite;
  ReferenceNode *parent;
  DenseMap<ReferenceID, std::shared_ptr<ReferenceNode>> children;
  bool childrenContainValueRefs;
};

using ValueMap = DenseMap<mlir::Value, std::shared_ptr<ReferenceNode>>;

class RedundantReadAndWriteEliminationPass : public llzk::impl::RedundantReadAndWriteEliminationPassBase<RedundantReadAndWriteEliminationPass> {
  void runOnOperation() override {
    getOperation().walk([&](FuncOp fn) {
      runOnFunc(fn);
    });
  }

  void runOnFunc(FuncOp fn) {
    if (fn.getCallableRegion() == nullptr) {
      return;
    }

    llvm::errs() << "Running on " << fn.getName() << "\n";

    ValueMap state;
    // Initialize the state to the function arguments.
    for (auto arg : fn.getArguments()) {
      state[arg] = ReferenceNode::create(arg, arg);
      llvm::errs() << "init value " << arg << "\n";
    }

    DenseMap<Block*, ValueMap> endStates;
    endStates[nullptr] = state;
    auto getBlockState = [&endStates](Block *blockPtr) {
      auto it = endStates.find(blockPtr);
      ensure(it != endStates.end(), "unknown end state means we have an unsupported backedge");
      return it->second;
    };
    std::deque<Block*> frontier;
    frontier.push_back(&fn.getCallableRegion()->front());
    DenseSet<Block*> visited;

    while (!frontier.empty()) {
      auto currentBlock = frontier.front();
      frontier.pop_front();
      visited.insert(currentBlock);

      // get predecessors
      ValueMap currentState;
      auto it = currentBlock->pred_begin();
      auto itEnd = currentBlock->pred_end();
      if (it == itEnd) {
        currentState = endStates[nullptr];
      } else {
        currentState = getBlockState(*it);
        for (it++; it != itEnd; it++) {
          llvm::report_fatal_error("todo! intersect state maps!");
        }
      }

      // Run this block
      runOnBlock(*currentBlock, currentState);

      // Update the end states
      ensure(endStates.find(currentBlock) == endStates.end(), "backedge");
      endStates[currentBlock] = currentState;

      // add successors to frontier
      for (auto *succ : currentBlock->getSuccessors()) {
        if (visited.find(succ) == visited.end()) {
          frontier.push_back(succ);
        }
      }
    }
  }

  void runOnBlock(Block &b, ValueMap &state) {
    DenseMap<Value, Value> replacementMap;
    SmallVector<Value> readVals;
    SmallVector<Operation*> redundantWrites;
    for (Operation &op : b) {
      runOperation(&op, state, replacementMap, readVals, redundantWrites);
    }
    for (auto &[orig, replace] : replacementMap) {
      orig.replaceAllUsesWith(replace);
      // We save the deletion to the readVals loop to prevent double-free.
    }
    // Remove redundant writes now that it is safe to do so.
    for (auto *writeOp : redundantWrites) {
      llvm::errs() << "erase write: " << *writeOp << "\n";
      writeOp->erase();
    }
    // Now we do a pass over read values to see if any are now unused.
    // We do this in reverse order to free up early reads if their users would
    // be removed.
    for (auto it = readVals.rbegin(); it != readVals.rend(); it++) {
      auto readVal = *it;
      if (readVal.use_empty()) {
        llvm::errs() << "erase read: " << readVal << "\n";
        readVal.getDefiningOp()->erase();
      }
    }
  }

  void runOperation(Operation *op, ValueMap &state, DenseMap<Value, Value> &replacementMap, SmallVector<Value> &readVals, SmallVector<Operation*> &redundantWrites) {
    auto translate = [&replacementMap](Value v) {
      if (auto it = replacementMap.find(v); it != replacementMap.end()) {
        return it->second;
      }
      return v;
    };

    auto tryGetValTree = [&state](Value v) -> std::shared_ptr<ReferenceNode> {
      if (auto it = state.find(v); it != state.end()) {
        return it->second;
      }
      return nullptr;
    };

    // struct ops
    if (auto newStruct = dyn_cast<CreateStructOp>(op)) {
      // For new values, the "stored value" of the reference is the creation site.
      auto structVal = ReferenceNode::create(newStruct, newStruct);
      state[newStruct] = structVal;
      llvm::errs() << "new_struct: " << *state[newStruct] << "\n";
      // adding this to readVals
      readVals.push_back(newStruct);
    }
    else if (auto readf = dyn_cast<FieldReadOp>(op)) {
      auto structVal = state.at(translate(readf.getComponent()));
      auto symbol = readf.getFieldNameAttr();
      auto resVal = translate(readf.getVal());
      // Check if such a child already exists.
      if (auto child = structVal->getChild(symbol)) {
        llvm::errs() << "replace " << resVal << " with " << child->getStoredValue() << "\n";
        replacementMap[resVal] = child->getStoredValue();
      } else {
        state[readf] = structVal->createChild(symbol, resVal);
        llvm::errs() << "readf: " << *state[readf] << "\n";
      }
      // specifically add the untranslated value back for removal checks
      readVals.push_back(readf.getVal());
    } else if (auto writef = dyn_cast<FieldWriteOp>(op)) {
      auto structVal = state.at(translate(writef.getComponent()));
      auto writeVal = translate(writef.getVal());
      auto symbol = writef.getFieldNameAttr();
      auto valTree = tryGetValTree(writeVal);

      auto child = structVal->getOrCreateChild(symbol);
      if (child->getStoredValue() == writeVal) {
        redundantWrites.push_back(writef);
      } else {
        if (auto *lastWrite = child->updateLastWrite(writef)) {
          llvm::errs() << "eliminate redundant write " << *lastWrite << "\n";
          redundantWrites.push_back(lastWrite);
        }
        child->setCurrentValue(writeVal, valTree);
      }
    }
    // array ops
    else if (auto newArray = dyn_cast<CreateArrayOp>(op)) {
      auto arrayVal = ReferenceNode::create(newArray, newArray);
      state[newArray] = arrayVal;

      unsigned idx = 0;
      for (auto elem : newArray.getElements()) {
        auto elemVal = translate(elem);
        auto valTree = tryGetValTree(elemVal);
        auto elemChild = arrayVal->createChild(idx, elemVal, valTree);
        llvm::errs() << "new_array[" << idx << "]: " << *elemChild << "\n";
        idx++;
      }

      readVals.push_back(newArray);
    } else if (auto readarr = dyn_cast<ReadArrayOp>(op)) {
      auto arrVal = state.at(translate(readarr.getArrRef()));

      auto currVal = arrVal;
      bool mayAlias = false;
      for (auto idx : readarr.getIndices()) {
        auto idxVal = translate(idx);
        mayAlias |= currVal->childrenMayAlias();
        currVal = currVal->getOrCreateChild(idxVal);
      }

      auto resVal = readarr.getResult();
      if (currVal->getStoredValue() != resVal && !mayAlias) {
        llvm::errs() << "readarr: replace " << resVal << " with " << currVal->getStoredValue() << '\n';
        replacementMap[resVal] = currVal->getStoredValue();
      } else {
        state[resVal] = currVal;
        llvm::errs() << "readarr: " << resVal << " => " << *currVal << "\n";
      }

      readVals.push_back(resVal);
    } else if (auto writearr = dyn_cast<WriteArrayOp>(op)) {
      auto arrayVal = state.at(translate(writearr.getArrRef()));
      auto newVal = translate(writearr.getRvalue());
      auto valTree = tryGetValTree(newVal);

      auto currentArrVal = arrayVal;
      bool mayAlias = false;
      for (auto idx : writearr.getIndices()) {
        auto idxVal = translate(idx);
        mayAlias |= currentArrVal->childrenMayAlias();
        currentArrVal = currentArrVal->getOrCreateChild(idxVal);
      }

      if (currentArrVal->getStoredValue() == newVal && !mayAlias) {
        llvm::errs() << "writearr: subsequent " << writearr << " is redundant\n";
        redundantWrites.push_back(writearr);
      } else {
        auto *lastWrite = currentArrVal->updateLastWrite(writearr);
        if (lastWrite && !mayAlias) {
          llvm::errs() << "writearr: replacing " << lastWrite << " with prior write " << *lastWrite << "\n";
          redundantWrites.push_back(lastWrite);
        }
        currentArrVal->setCurrentValue(newVal, valTree);
      }
    } else if (auto extractarr = dyn_cast<ExtractArrayOp>(op)) {
      // Logic is essentially the same as readarr
      auto arrVal = state.at(translate(extractarr.getArrRef()));

      auto currVal = arrVal;
      bool mayAlias = false;
      for (auto idx : extractarr.getIndices()) {
        auto idxVal = translate(idx);
        mayAlias |= currVal->childrenMayAlias();
        currVal = currVal->getOrCreateChild(idxVal);
      }

      auto resVal = extractarr.getResult();
      if (!currVal->hasStoredValue()) {
        currVal->setCurrentValue(resVal);
      } else if (currVal->getStoredValue() != resVal && !mayAlias) {
        llvm::errs() << "extractarr: replace " << resVal << " with " << currVal->getStoredValue() << '\n';
        replacementMap[resVal] = currVal->getStoredValue();
      } else {
        state[resVal] = currVal;
        llvm::errs() << "extractarr: " << resVal << " => " << *currVal << "\n";
      }

      readVals.push_back(resVal);

    } else if (auto insertarr = dyn_cast<InsertArrayOp>(op)) {
      // Logic is essentially the same as writearr
      auto arrayVal = state.at(translate(insertarr.getArrRef()));
      auto newVal = translate(insertarr.getRvalue());
      auto valTree = tryGetValTree(newVal);

      auto currentArrVal = arrayVal;
      bool mayAlias = false;
      for (auto idx : insertarr.getIndices()) {
        auto idxVal = translate(idx);
        mayAlias |= currentArrVal->childrenMayAlias();
        currentArrVal = currentArrVal->getOrCreateChild(idxVal);
      }

      if (currentArrVal->getStoredValue() == newVal && !mayAlias) {
        llvm::errs() << "insertarr: subsequent " << insertarr << " is redundant\n";
        redundantWrites.push_back(insertarr);
      } else {
        auto *lastWrite = currentArrVal->updateLastWrite(insertarr);
        if (lastWrite && !mayAlias) {
          llvm::errs() << "insertarr: replacing " << lastWrite << " with prior write " << *lastWrite << "\n";
          redundantWrites.push_back(lastWrite);
        }
        currentArrVal->setCurrentValue(newVal, valTree);
      }
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> llzk::createRedundantReadAndWriteEliminationPass() {
  return std::make_unique<RedundantReadAndWriteEliminationPass>();
};
