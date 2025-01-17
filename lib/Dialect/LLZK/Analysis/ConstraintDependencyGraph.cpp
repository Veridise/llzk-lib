#include "llzk/Dialect/LLZK/Analysis/ConstraintDependencyGraph.h"
#include "llzk/Dialect/LLZK/Analysis/DenseAnalysis.h"
#include "llzk/Dialect/LLZK/Util/Hash.h"
#include "llzk/Dialect/LLZK/Util/SymbolHelper.h"

#include <mlir/Analysis/DataFlow/DeadCodeAnalysis.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Index/IR/IndexOps.h>
#include <mlir/IR/Value.h>

#include <numeric>
#include <unordered_set>

namespace llzk {

/*
Private Utilities:

These classes are defined here and not in the header as they are not designed
for use outside of this specific ConstraintDependencyGraph analysis.
*/

class ConstrainRefLatticeValue;

using ConstantMap = mlir::DenseMap<mlir::Value, mlir::APInt>;
using ConstrainRefSetMap = mlir::DenseMap<mlir::Value, ConstrainRefSet>;
using ConstrainRefSetVec = std::vector<ConstrainRefSet>;
using ArgumentMap = mlir::DenseMap<mlir::BlockArgument, ConstrainRefSet>;
using TranslationMap =
    std::unordered_map<ConstrainRef, ConstrainRefLatticeValue, ConstrainRef::Hash>;

/// @brief A value at a given point of the ConstrainRefLattice.
class ConstrainRefLatticeValue {
  /// For scalar values.
  using ScalarTy = ConstrainRefSet;
  /// For arrays of values created by, e.g., the LLZK new_array op. A recursive
  /// definition to support arrays of arbitrary dimensions.
  /// Unique pointers are used as each value must be self contained for the
  /// sake of consistent translations.
  /// This array is flattened.
  using ArrayTy = std::vector<std::unique_ptr<ConstrainRefLatticeValue>>;

  static ArrayTy constructArrayTy(const mlir::ArrayRef<int64_t> &shape) {
    size_t totalElem = 1;
    for (auto dim : shape) {
      totalElem *= dim;
    }
    ArrayTy arr(totalElem);
    for (auto it = arr.begin(); it != arr.end(); it++) {
      *it = std::make_unique<ConstrainRefLatticeValue>();
    }
    return arr;
  }

public:
  explicit ConstrainRefLatticeValue(ScalarTy s) : value(s), arrayShape(std::nullopt) {}
  explicit ConstrainRefLatticeValue(ConstrainRef r) : ConstrainRefLatticeValue(ScalarTy {r}) {}
  ConstrainRefLatticeValue() : ConstrainRefLatticeValue(ScalarTy {}) {}

  // Create an empty array of the given shape.
  explicit ConstrainRefLatticeValue(mlir::ArrayRef<int64_t> shape)
      : value(constructArrayTy(shape)), arrayShape(shape) {}

  // Enable copying by duplicating unique_ptrs
  ConstrainRefLatticeValue(const ConstrainRefLatticeValue &rhs) { *this = rhs; }

  ConstrainRefLatticeValue &operator=(const ConstrainRefLatticeValue &rhs) {
    arrayShape = rhs.arrayShape;
    if (rhs.isScalar()) {
      value = rhs.getScalarValue();
    } else {
      // create an empty array of the same size
      value = constructArrayTy(rhs.arrayShape.value());
      auto &lhsArr = getArrayValue();
      auto &rhsArr = rhs.getArrayValue();
      for (unsigned i = 0; i < lhsArr.size(); i++) {
        // Recursive copy assignment of lattice values
        *lhsArr[i] = *rhsArr[i];
      }
    }
    return *this;
  }

  bool isScalar() const { return std::holds_alternative<ScalarTy>(value); }
  bool isSingleValue() const { return isScalar() && getScalarValue().size() == 1; }
  bool isArray() const { return std::holds_alternative<ArrayTy>(value); }

  const ScalarTy &getScalarValue() const {
    debug::ensure(isScalar(), "not a scalar value");
    return std::get<ScalarTy>(value);
  }

  ScalarTy &getScalarValue() {
    debug::ensure(isScalar(), "not a scalar value");
    return std::get<ScalarTy>(value);
  }

  const ConstrainRef &getSingleValue() const {
    debug::ensure(isSingleValue(), "not a single value");
    return *getScalarValue().begin();
  }

  const ArrayTy &getArrayValue() const {
    debug::ensure(isArray(), "not an array value");
    return std::get<ArrayTy>(value);
  }

  size_t getArraySize() const { return getArrayValue().size(); }

  ArrayTy &getArrayValue() {
    debug::ensure(isArray(), "not an array value");
    return std::get<ArrayTy>(value);
  }

  const ConstrainRefLatticeValue &getElemFlatIdx(unsigned i) const {
    debug::ensure(isArray(), "not an array value");
    auto &arr = getArrayValue();
    debug::ensure(i < arr.size(), "index out of range");
    return *arr.at(i);
  }

  ConstrainRefLatticeValue &getElemFlatIdx(unsigned i) {
    debug::ensure(isArray(), "not an array value");
    auto &arr = getArrayValue();
    debug::ensure(i < arr.size(), "index out of range");
    return *arr.at(i);
  }

  /// @brief Sets this value to be equal to `rhs`.
  /// Like the assignment operator, but returns a mlir::ChangeResult if an update
  /// is created,
  mlir::ChangeResult setValue(const ConstrainRefLatticeValue &rhs) {
    if (*this == rhs) {
      return mlir::ChangeResult::NoChange;
    }
    *this = rhs;
    return mlir::ChangeResult::Change;
  }

  /// @brief Union this value with that of rhs.
  mlir::ChangeResult update(const ConstrainRefLatticeValue &rhs) {
    if (isScalar() && rhs.isScalar()) {
      return updateScalar(rhs.getScalarValue());
    } else if (isArray() && rhs.isArray() && getArraySize() == rhs.getArraySize()) {
      return updateArray(rhs.getArrayValue());
    } else {
      return foldAndUpdate(rhs);
    }
  }

  mlir::ChangeResult insert(const ConstrainRef &rhs) {
    auto rhsVal = ConstrainRefLatticeValue(rhs);
    if (isScalar()) {
      return updateScalar(rhsVal.getScalarValue());
    } else {
      return foldAndUpdate(rhsVal);
    }
  }

  /// Translate
  std::pair<ConstrainRefLatticeValue, mlir::ChangeResult>
  translate(const TranslationMap &translation) const {
    auto newVal = *this;
    auto res = mlir::ChangeResult::NoChange;
    if (newVal.isScalar()) {
      res = newVal.translateScalar(translation);
    } else {
      for (auto &elem : newVal.getArrayValue()) {
        auto [newElem, elemRes] = elem->translate(translation);
        (*elem) = newElem;
        res |= elemRes;
      }
    }
    return {newVal, res};
  }

  std::pair<ConstrainRefLatticeValue, mlir::ChangeResult>
  index(const std::vector<ConstrainRefIndex> &indices) const {
    if (isArray()) {
      debug::ensure(indices.size() == arrayShape->size(), "dimension mismatch");
      // if all the indices are concrete, then we can resolve to a specific element. Otherwise,
      // we'll fold the values together.

      // the set of indices to fold together
      std::vector<size_t> currIdxs {0};
      for (unsigned i = 0; i < indices.size(); i++) {
        auto &idx = indices[i];
        auto currDim = arrayShape.value()[i];

        std::vector<size_t> newIdxs;
        debug::ensure(idx.isIndex() || idx.isIndexRange(), "wrong type of index for array");
        if (idx.isIndex()) {
          auto idxVal = idx.getIndex().getZExtValue();
          std::transform(
              currIdxs.begin(), currIdxs.end(), std::back_inserter(newIdxs),
              [&currDim, &idxVal](size_t i) { return i * currDim + idxVal; }
          );
        } else {
          auto [low, high] = idx.getIndexRange();
          for (auto idxVal = low.getZExtValue(); idxVal < high.getZExtValue(); idxVal++) {
            std::transform(
                currIdxs.begin(), currIdxs.end(), std::back_inserter(newIdxs),
                [&currDim, &idxVal](size_t i) { return i * currDim + idxVal; }
            );
          }
        }
        currIdxs = newIdxs;
      }

      // Now, get and fold all the values together.
      auto &arr = getArrayValue();
      auto currLatticeVal = *arr.at(currIdxs[0]);
      for (unsigned i = 1; i < currIdxs.size(); i++) {
        (void)currLatticeVal.update(*arr.at(currIdxs[i]));
      }
      return {currLatticeVal, mlir::ChangeResult::Change};
    } else {
      auto currVal = *this;
      auto res = mlir::ChangeResult::NoChange;
      for (auto &idx : indices) {
        auto transform = [&idx](const ConstrainRef &r) -> ConstrainRef {
          return r.createChild(idx);
        };
        auto [newVal, transformRes] = currVal.elementwiseTransform(transform);
        currVal = std::move(newVal);
        res |= transformRes;
      }
      return {currVal, res};
    }
  }

  std::pair<ConstrainRefLatticeValue, mlir::ChangeResult> index(const ConstrainRefIndex &idx
  ) const {
    auto transform = [&idx](const ConstrainRef &r) -> ConstrainRef { return r.createChild(idx); };
    return elementwiseTransform(transform);
  }

  std::pair<ConstrainRefLatticeValue, mlir::ChangeResult> index(const ConstrainRef &fieldRef
  ) const {
    auto transform = [&fieldRef](const ConstrainRef &r) -> ConstrainRef {
      return r.createChild(fieldRef);
    };
    return elementwiseTransform(transform);
  }

  /// Perform an extract_arr operation
  std::pair<ConstrainRefLatticeValue, mlir::ChangeResult>
  extract(const std::vector<ConstrainRefIndex> &indices) const {
    if (isArray()) {
      debug::ensure(indices.size() < arrayShape->size(), "invalid extract array operands");

      // First, compute what chunk(s) to index
      std::vector<size_t> currIdxs {0};
      for (unsigned i = 0; i < indices.size(); i++) {
        auto &idx = indices[i];
        auto currDim = arrayShape.value()[i];

        std::vector<size_t> newIdxs;
        debug::ensure(idx.isIndex() || idx.isIndexRange(), "wrong type of index for array");
        if (idx.isIndex()) {
          auto idxVal = idx.getIndex().getZExtValue();
          std::transform(
              currIdxs.begin(), currIdxs.end(), std::back_inserter(newIdxs),
              [&currDim, &idxVal](size_t i) { return i * currDim + idxVal; }
          );
        } else {
          auto [low, high] = idx.getIndexRange();
          for (auto idxVal = low.getZExtValue(); idxVal < high.getZExtValue(); idxVal++) {
            std::transform(
                currIdxs.begin(), currIdxs.end(), std::back_inserter(newIdxs),
                [&currDim, &idxVal](size_t i) { return i * currDim + idxVal; }
            );
          }
        }

        currIdxs = newIdxs;
      }
      std::vector<int64_t> newArrayDims;
      size_t chunkSz = 1;
      for (unsigned i = indices.size(); i < arrayShape->size(); i++) {
        auto dim = arrayShape->at(i);
        newArrayDims.push_back(dim);
        chunkSz *= dim;
      }
      auto extractedVal = ConstrainRefLatticeValue(newArrayDims);
      for (auto chunkStart : currIdxs) {
        for (size_t i = 0; i < chunkSz; i++) {
          (void)extractedVal.getElemFlatIdx(i).update(getElemFlatIdx(chunkStart + i));
        }
      }

      return {extractedVal, mlir::ChangeResult::Change};
    } else {
      auto currVal = *this;
      auto res = mlir::ChangeResult::NoChange;
      for (auto &idx : indices) {
        auto transform = [&idx](const ConstrainRef &r) -> ConstrainRef {
          return r.createChild(idx);
        };
        auto [newVal, transformRes] = currVal.elementwiseTransform(transform);
        currVal = std::move(newVal);
        res |= transformRes;
      }
      return {currVal, res};
    }
  }

  /// A convenience wrapper to either extract a sub array or fully index the array.
  std::pair<ConstrainRefLatticeValue, mlir::ChangeResult>
  extractOrIndex(const std::vector<ConstrainRefIndex> &indices) const {
    if (isArray() && indices.size() < arrayShape->size()) {
      return extract(indices);
    }
    return index(indices);
  }

  ScalarTy foldToScalar() const {
    if (isScalar()) {
      return getScalarValue();
    }

    ScalarTy res;
    for (auto &val : getArrayValue()) {
      auto rhs = val->foldToScalar();
      res.insert(rhs.begin(), rhs.end());
    }
    return res;
  }

  void print(mlir::raw_ostream &os) const {
    if (isScalar()) {
      os << getScalarValue();
    } else {
      os << "[ ";
      const auto &arr = getArrayValue();
      for (auto it = arr.begin(); it != arr.end();) {
        (*it)->print(os);
        it++;
        if (it != arr.end()) {
          os << ", ";
        } else {
          os << ' ';
        }
      }
      os << ']';
    }
  }

  bool operator==(const ConstrainRefLatticeValue &rhs) const {
    if (isScalar() && rhs.isScalar()) {
      return getScalarValue() == rhs.getScalarValue();
    } else if (isArray() && rhs.isArray() && getArraySize() == rhs.getArraySize()) {
      for (size_t i = 0; i < getArraySize(); i++) {
        if (getElemFlatIdx(i) != rhs.getElemFlatIdx(i)) {
          return false;
        }
      }
      return true;
    }
    return false;
  }

private:
  std::variant<ScalarTy, ArrayTy> value;
  std::optional<std::vector<int64_t>> arrayShape;

  mlir::ChangeResult updateScalar(const ScalarTy &rhs) {
    mlir::ChangeResult res = mlir::ChangeResult::NoChange;
    auto &lhs = getScalarValue();
    for (auto &ref : rhs) {
      auto [_, inserted] = lhs.insert(ref);
      res |= inserted ? mlir::ChangeResult::Change : mlir::ChangeResult::NoChange;
    }
    return res;
  }

  mlir::ChangeResult updateArray(const ArrayTy &rhs) {
    mlir::ChangeResult res = mlir::ChangeResult::NoChange;
    auto &lhs = getArrayValue();
    for (size_t i = 0; i < getArraySize(); i++) {
      res |= lhs[i]->update(*rhs.at(i));
    }
    return res;
  }

  mlir::ChangeResult foldAndUpdate(const ConstrainRefLatticeValue &rhs) {
    auto folded = foldToScalar();
    auto rhsScalar = rhs.foldToScalar();
    folded.insert(rhsScalar.begin(), rhsScalar.end());
    if (isScalar() && getScalarValue() == folded) {
      return mlir::ChangeResult::NoChange;
    }
    value = folded;
    return mlir::ChangeResult::Change;
  }

  mlir::ChangeResult translateScalar(const TranslationMap &translation) {
    auto res = mlir::ChangeResult::NoChange;
    // copy the current value
    auto currVal = getScalarValue();
    // reset this value
    value = ScalarTy();
    for (auto &[ref, val] : translation) {
      auto it = currVal.find(ref);
      if (it != currVal.end()) {
        res |= update(val);
      }
    }
    return res;
  }

  std::pair<ConstrainRefLatticeValue, mlir::ChangeResult>
  elementwiseTransform(std::function<ConstrainRef(const ConstrainRef &)> transform) const {
    auto newVal = *this;
    auto res = mlir::ChangeResult::NoChange;
    if (newVal.isScalar()) {
      ScalarTy indexed;
      for (auto &ref : newVal.getScalarValue()) {
        auto [_, inserted] = indexed.insert(transform(ref));
        if (inserted) {
          res |= mlir::ChangeResult::Change;
        }
      }
      newVal.getScalarValue() = indexed;
    } else {
      for (auto &elem : newVal.getArrayValue()) {
        auto [newElem, elemRes] = elem->elementwiseTransform(transform);
        (*elem) = newElem;
        res |= elemRes;
      }
    }
    return {newVal, res};
  }
};

mlir::raw_ostream &operator<<(mlir::raw_ostream &os, const ConstrainRefLatticeValue &v) {
  v.print(os);
  return os;
}

/// A lattice for use in dense analysis.
class ConstrainRefLattice : public dataflow::AbstractDenseLattice {
public:
  using ValueMap = mlir::DenseMap<mlir::Value, ConstrainRefLatticeValue>;
  using AbstractDenseLattice::AbstractDenseLattice;

  /* Static utilities */

  /// If val is the source of other values (i.e., a block argument from the function
  /// args or a constant), create the base reference to the val. Otherwise,
  /// return failure.
  /// Our lattice values must originate from somewhere.
  static mlir::FailureOr<ConstrainRef> getSourceRef(mlir::Value val) {
    if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(val)) {
      return ConstrainRef(blockArg);
    } else if (auto defOp = val.getDefiningOp()) {
      if (auto constFelt = mlir::dyn_cast<FeltConstantOp>(defOp)) {
        return ConstrainRef(constFelt);
      } else if (auto constIdx = mlir::dyn_cast<mlir::index::ConstantOp>(defOp)) {
        return ConstrainRef(constIdx);
      } else if (auto readConst = mlir::dyn_cast<ConstReadOp>(defOp)) {
        return ConstrainRef(readConst);
      }
    }
    return mlir::failure();
  }

  /* Required methods */

  /// Maximum upper bound
  mlir::ChangeResult join(const AbstractDenseLattice &rhs) override {
    if (auto *r = dynamic_cast<const ConstrainRefLattice *>(&rhs)) {
      return setValues(r->valMap);
    }
    llvm::report_fatal_error("invalid join lattice type");
    return mlir::ChangeResult::NoChange;
  }

  /// Minimum lower bound
  virtual mlir::ChangeResult meet(const AbstractDenseLattice &rhs) override {
    llvm::report_fatal_error("meet operation is not supported for ConstrainRefLattice");
    return mlir::ChangeResult::NoChange;
  }

  void print(mlir::raw_ostream &os) const override {
    os << "ConstrainRefLattice { ";
    for (auto mit = valMap.begin(); mit != valMap.end();) {
      auto &[val, latticeVal] = *mit;
      os << "\n    (" << val << ") => " << latticeVal;
      mit++;
      if (mit != valMap.end()) {
        os << ',';
      } else {
        os << '\n';
      }
    }
    os << "}\n";
  }

  /* Update utility methods */

  mlir::ChangeResult setValues(const ValueMap &rhs) {
    auto res = mlir::ChangeResult::NoChange;

    for (auto &[v, s] : rhs) {
      res |= setValue(v, s);
    }
    return res;
  }

  mlir::ChangeResult setValue(mlir::Value v, const ConstrainRefLatticeValue &rhs) {
    return valMap[v].setValue(rhs);
  }

  mlir::ChangeResult setValue(mlir::Value v, const ConstrainRef &ref) {
    return valMap[v].setValue(ConstrainRefLatticeValue(ref));
  }

  ConstrainRefLatticeValue getOrDefault(mlir::Value v) const {
    auto it = valMap.find(v);
    if (it == valMap.end()) {
      auto sourceRef = getSourceRef(v);
      if (mlir::succeeded(sourceRef)) {
        return ConstrainRefLatticeValue(sourceRef.value());
      }
      return ConstrainRefLatticeValue();
    }
    return it->second;
  }

  ConstrainRefLatticeValue getReturnValue(unsigned i) const {
    auto op = this->getPoint().get<mlir::Operation *>();
    if (auto retOp = mlir::dyn_cast<ReturnOp>(op)) {
      if (i >= retOp.getNumOperands()) {
        llvm::report_fatal_error("return value requested is out of range");
      }
      return this->getOrDefault(retOp.getOperand(i));
    }
    return ConstrainRefLatticeValue();
  }

private:
  ValueMap valMap;
};

mlir::raw_ostream &operator<<(mlir::raw_ostream &os, const ConstrainRefLattice &v) {
  v.print(os);
  return os;
}

/// @brief The dataflow analysis that computes the set of references that
/// LLZK operations use and produce. The analysis is simple: any operation will
/// simply output a union of its input references, regardless of what type of
/// operation it performs, as the analysis is operator-insensitive.
class ConstrainRefAnalysis : public dataflow::DenseForwardDataFlowAnalysis<ConstrainRefLattice> {
public:
  using dataflow::DenseForwardDataFlowAnalysis<ConstrainRefLattice>::DenseForwardDataFlowAnalysis;

  void visitCallControlFlowTransfer(
      mlir::CallOpInterface call, dataflow::CallControlFlowAction action,
      const ConstrainRefLattice &before, ConstrainRefLattice *after
  ) override {

    auto fnOpRes = resolveCallable<FuncOp>(tables, call);
    debug::ensure(succeeded(fnOpRes), "could not resolve called function");

    auto fnOp = fnOpRes->get();
    if (fnOp.getName() == FUNC_NAME_CONSTRAIN || fnOp.getName() == FUNC_NAME_COMPUTE) {
      // Do nothing special.
      join(after, before);
      return;
    }
    /// `action == CallControlFlowAction::Enter` indicates that:
    ///   - `before` is the state before the call operation;
    ///   - `after` is the state at the beginning of the callee entry block;
    else if (action == dataflow::CallControlFlowAction::EnterCallee) {
      // Add all of the argument values to the lattice.
      auto calledFnRes = resolveCallable<FuncOp>(tables, call);
      debug::ensure(mlir::succeeded(calledFnRes), "could not resolve function call");
      auto calledFn = calledFnRes->get();

      auto updated = after->join(before);
      for (auto arg : calledFn->getRegion(0).getArguments()) {
        auto sourceRef = ConstrainRefLattice::getSourceRef(arg);
        debug::ensure(mlir::succeeded(sourceRef), "failed to get source ref");
        updated |= after->setValue(arg, sourceRef.value());
      }
      propagateIfChanged(after, updated);
    }
    /// `action == CallControlFlowAction::Exit` indicates that:
    ///   - `before` is the state at the end of a callee exit block;
    ///   - `after` is the state after the call operation.
    else if (action == dataflow::CallControlFlowAction::ExitCallee) {
      // Translate argument values based on the operands given at the call site.
      std::unordered_map<ConstrainRef, ConstrainRefLatticeValue, ConstrainRef::Hash> translation;
      auto funcOpRes = resolveCallable<FuncOp>(tables, call);
      debug::ensure(mlir::succeeded(funcOpRes), "could not lookup called function");
      auto funcOp = funcOpRes->get();

      auto callOp = mlir::dyn_cast<CallOp>(call.getOperation());
      debug::ensure(callOp, "call is not a llzk::CallOp");

      for (unsigned i = 0; i < funcOp.getNumArguments(); i++) {
        auto key = ConstrainRef(funcOp.getArgument(i));
        auto val = before.getOrDefault(callOp.getOperand(i));
        translation[key] = val;
        llvm::errs() << "Translating " << key << " to " << val << "\n";
      }

      mlir::ChangeResult updated = after->join(before);
      for (unsigned i = 0; i < callOp.getNumResults(); i++) {
        auto retVal = before.getReturnValue(i);
        auto [translatedVal, _] = retVal.translate(translation);
        updated |= after->setValue(callOp->getResult(i), translatedVal);
      }
      propagateIfChanged(after, updated);
    }
    // Note that `setToEntryState` may be a "partial fixpoint" for some
    // lattices, e.g., lattices that are lists of maps of other lattices will
    // only set fixpoint for "known" lattices.
    else if (action == mlir::dataflow::CallControlFlowAction::ExternalCallee) {
      setToEntryState(after);
    }
  }

  /// @brief Propagate constrain reference lattice values from operands to results.
  /// @param op
  /// @param before
  /// @param after
  void visitOperation(
      mlir::Operation *op, const ConstrainRefLattice &before, ConstrainRefLattice *after
  ) override {
    // Collect the references that are made by the operands to `op`.
    ConstrainRefLattice::ValueMap operandVals;
    for (auto &operand : op->getOpOperands()) {
      operandVals[operand.get()] = before.getOrDefault(operand.get());
    }

    // Propagate existing state.
    join(after, before);

    // We will now join the the operand refs based on the type of operand.
    if (auto fieldRead = mlir::dyn_cast<FieldReadOp>(op)) {
      // In the readf case, the operand is indexed into by the read's fielddefop.
      assert(operandVals.size() == 1);
      assert(fieldRead->getNumResults() == 1);

      auto fieldOpRes = fieldRead.getFieldDefOp(tables);
      debug::ensure(mlir::succeeded(fieldOpRes), "could not find field read");

      auto res = fieldRead->getResult(0);
      auto idx = ConstrainRefIndex(fieldOpRes.value());
      const auto &ops = operandVals.at(fieldRead->getOpOperand(0).get());
      auto [fieldVals, _] = ops.index(idx);

      propagateIfChanged(after, after->setValue(res, fieldVals));
    } else if (auto arrayRead = mlir::dyn_cast<ReadArrayOp>(op)) {
      // In the readarr case, we index the first operand by all remaining indices
      assert(arrayRead->getNumResults() == 1);
      auto res = arrayRead->getResult(0);

      auto array = arrayRead.getOperand(0);
      auto currVals = operandVals[array];

      // read_arr must fully index the array. We'll accumulate all indices
      // and index the value all at once.
      std::vector<ConstrainRefIndex> indices;

      for (size_t i = 1; i < arrayRead.getNumOperands(); i++) {
        auto currentOp = arrayRead.getOperand(i);
        auto &idxVals = operandVals[currentOp];

        if (idxVals.isSingleValue() && idxVals.getSingleValue().isConstantIndex()) {
          ConstrainRefIndex idx(idxVals.getSingleValue().getConstantIndexValue());
          indices.push_back(idx);
        } else {
          // Otherwise, assume any range is valid.
          auto arrayType = mlir::dyn_cast<ArrayType>(array.getType());
          auto lower = mlir::APInt::getZero(64);
          mlir::APInt upper(64, arrayType.getDimSize(i - 1));
          auto idxRange = ConstrainRefIndex(lower, upper);
          indices.push_back(idxRange);
        }
      }

      auto [newVals, _] = currVals.index(indices);

      propagateIfChanged(after, after->setValue(res, newVals));
    } else if (auto createArray = mlir::dyn_cast<CreateArrayOp>(op)) {
      // Create an array using the operand values, if they exist.
      // Currently, the new array must either be fully initialized or uninitialized.

      auto newArrayVal = ConstrainRefLatticeValue(createArray.getType().getShape());
      // If the array is initialized, iterate through all operands and initialize the array value.
      for (unsigned i = 0; i < createArray.getNumOperands(); i++) {
        auto currentOp = createArray.getOperand(i);
        auto &opVals = operandVals[currentOp];
        (void)newArrayVal.getElemFlatIdx(i).setValue(opVals);
      }

      assert(createArray->getNumResults() == 1);
      auto res = createArray->getResult(0);

      propagateIfChanged(after, after->setValue(res, newArrayVal));
    } else if (auto extractArray = mlir::dyn_cast<ExtractArrayOp>(op)) {
      // Pretty similar to the readarr case
      // In the extract case, we index the first operand by all remaining indices
      assert(extractArray->getNumResults() == 1);
      auto res = extractArray->getResult(0);

      auto array = extractArray.getOperand(0);
      auto currVals = operandVals[array];

      // extractarr must partially index the array. We'll accumulate all indices
      // and index the value all at once.
      std::vector<ConstrainRefIndex> indices;

      for (size_t i = 1; i < extractArray.getNumOperands(); i++) {
        auto currentOp = extractArray.getOperand(i);
        auto &idxVals = operandVals[currentOp];

        if (idxVals.isSingleValue() && idxVals.getSingleValue().isConstantIndex()) {
          ConstrainRefIndex idx(idxVals.getSingleValue().getConstantIndexValue());
          indices.push_back(idx);
        } else {
          // Otherwise, assume any range is valid.
          auto arrayType = mlir::dyn_cast<ArrayType>(array.getType());
          auto lower = mlir::APInt::getZero(64);
          mlir::APInt upper(64, arrayType.getDimSize(i - 1));
          auto idxRange = ConstrainRefIndex(lower, upper);
          indices.push_back(idxRange);
        }
      }

      auto [newVals, _] = currVals.extract(indices);

      propagateIfChanged(after, after->setValue(res, newVals));
    } else {
      // Standard union of operands into the results value.
      // TODO: Could perform constant computation/propagation here for, e.g., arithmetic
      // over constants, but such analysis may be better suited for a dedicated pass.
      propagateIfChanged(after, fallbackOpUpdate(op, operandVals, before, after));
    }
  }

protected:
  void setToEntryState(ConstrainRefLattice *lattice) override {
    // the entry state is empty, so do nothing.
  }

  // Perform a standard union of operands into the results value.
  mlir::ChangeResult fallbackOpUpdate(
      mlir::Operation *op, const ConstrainRefLattice::ValueMap &operandVals,
      const ConstrainRefLattice &before, ConstrainRefLattice *after
  ) {
    auto updated = mlir::ChangeResult::NoChange;
    for (auto res : op->getResults()) {
      auto cur = before.getOrDefault(res);

      for (auto &[_, opVal] : operandVals) {
        (void)cur.update(opVal);
      }
      updated |= after->setValue(res, cur);
    }
    return updated;
  }

private:
  mlir::SymbolTableCollection tables;
};

/*
ConstraintDependencyGraphAnalysis

Needs to be declared before implementing the ConstraintDependencyGraph functions, as
they reference the ConstraintDependencyGraphAnalysis.
*/

/// @brief An analysis wrapper around the ConstraintDependencyGraph for a given struct.
/// This analysis is a StructDefOp-level analysis that should not be directly
/// interacted with---rather, it is a utility used by the ConstraintDependencyGraphModuleAnalysis
/// that helps use MLIR's AnalysisManager to cache dependencies for sub-components.
class ConstraintDependencyGraphAnalysis {
public:
  ConstraintDependencyGraphAnalysis(mlir::Operation *op) {
    structDefOp = mlir::dyn_cast<StructDefOp>(op);
    if (!structDefOp) {
      auto error_message =
          "ConstraintDependencyGraphAnalysis expects provided op to be a StructDefOp!";
      op->emitError(error_message);
      llvm::report_fatal_error(error_message);
    }
    auto maybeModOp = getRootModule(op);
    if (mlir::failed(maybeModOp)) {
      auto error_message =
          "ConstraintDependencyGraphAnalysis could not find root module from StructDefOp!";
      op->emitError(error_message);
      llvm::report_fatal_error(error_message);
    }
    modOp = *maybeModOp;
  }

  /// @brief Construct a CDG, using the module's analysis manager to query
  /// ConstraintDependencyGraph objects for nested components.
  mlir::LogicalResult
  constructCDG(mlir::DataFlowSolver &solver, mlir::AnalysisManager &moduleAnalysisManager) {
    auto res =
        ConstraintDependencyGraph::compute(modOp, structDefOp, solver, moduleAnalysisManager);
    if (mlir::failed(res)) {
      return mlir::failure();
    }
    cdg = std::make_shared<ConstraintDependencyGraph>(*res);
    return mlir::success();
  }

  /// @brief Return true iff the CDG has been constructed
  bool constructed() const { return cdg != nullptr; }

  ConstraintDependencyGraph &getCDG() {
    ensureCDGCreated();
    return *cdg;
  }

  const ConstraintDependencyGraph &getCDG() const {
    ensureCDGCreated();
    return *cdg;
  }

private:
  mlir::ModuleOp modOp;
  StructDefOp structDefOp;
  std::shared_ptr<ConstraintDependencyGraph> cdg;

  void ensureCDGCreated() const {
    debug::ensure(cdg != nullptr, "CDG does not exist; must invoke constructCDG");
  }

  friend class ConstraintDependencyGraphModuleAnalysis;
};

/* ConstraintDependencyGraph */

mlir::FailureOr<ConstraintDependencyGraph> ConstraintDependencyGraph::compute(
    mlir::ModuleOp m, StructDefOp s, mlir::DataFlowSolver &solver, mlir::AnalysisManager &am
) {
  ConstraintDependencyGraph cdg(m, s);
  if (cdg.computeConstraints(solver, am).failed()) {
    return mlir::failure();
  }
  return cdg;
}

void ConstraintDependencyGraph::dump() const { print(llvm::errs()); }

/// Print all constraints. Any element that is unconstrained is omitted.
void ConstraintDependencyGraph::print(llvm::raw_ostream &os) const {
  // the EquivalenceClasses::iterator is sorted, but the EquivalenceClasses::member_iterator is
  // not guaranteed to be sorted. So, we will sort members before printing them.
  // We also want to add the constant values into the printing.
  std::set<std::set<ConstrainRef>> sortedSets;
  for (auto it = signalSets.begin(); it != signalSets.end(); it++) {
    if (!it->isLeader()) {
      continue;
    }

    std::set<ConstrainRef> sortedMembers;
    for (auto mit = signalSets.member_begin(it); mit != signalSets.member_end(); mit++) {
      sortedMembers.insert(*mit);
    }

    // We only want to print sets with a size > 1, because size == 1 means the
    // signal is not in a constraint.
    if (sortedMembers.size() > 1) {
      sortedSets.insert(sortedMembers);
    }
  }
  // Add the constants in separately.
  for (auto &[ref, constSet] : constantSets) {
    if (constSet.empty()) {
      continue;
    }
    std::set<ConstrainRef> sortedMembers(constSet.begin(), constSet.end());
    sortedMembers.insert(ref);
    sortedSets.insert(sortedMembers);
  }

  os << "ConstraintDependencyGraph { ";

  for (auto it = sortedSets.begin(); it != sortedSets.end();) {
    os << "\n    { ";
    for (auto mit = it->begin(); mit != it->end();) {
      os << *mit;
      mit++;
      if (mit != it->end()) {
        os << ", ";
      }
    }

    it++;
    if (it == sortedSets.end()) {
      os << " }\n";
    } else {
      os << " },";
    }
  }

  os << "}\n";
}

mlir::LogicalResult ConstraintDependencyGraph::computeConstraints(
    mlir::DataFlowSolver &solver, mlir::AnalysisManager &am
) {
  // Fetch the constrain function. This is a required feature for all LLZK structs.
  auto constrainFnOp = structDef.getConstrainFuncOp();
  debug::ensure(
      constrainFnOp,
      "malformed struct " + mlir::Twine(structDef.getName()) + " must define a constrain function"
  );

  /**
   * Now, given the analysis, construct the CDG:
   * - Union all references based on solver results.
   * - Union all references based on nested dependencies.
   */

  // - Union all constraints from the analysis
  // This requires iterating over all of the emit operations
  constrainFnOp.walk([this, &solver](EmitEqualityOp emitOp) {
    this->walkConstrainOp(solver, emitOp);
  });

  constrainFnOp.walk([this, &solver](EmitContainmentOp emitOp) {
    this->walkConstrainOp(solver, emitOp);
  });

  /**
   * Step two of the analysis is to traverse all of the constrain calls.
   * This is the nested analysis, basically.
   * Constrain functions don't return, so we don't need to compute "values" from
   * the call. We just need to see what constraints are generated here, and
   * add them to the transitive closures.
   */
  constrainFnOp.walk([this, &solver, &am](CallOp fnCall) mutable {
    auto res = resolveCallable<FuncOp>(tables, fnCall);
    debug::ensure(mlir::succeeded(res), "could not resolve constrain call");

    auto fn = res->get();
    if (fn.getName() != FUNC_NAME_CONSTRAIN) {
      return;
    }
    // Nested
    auto calledStruct = fn.getOperation()->getParentOfType<StructDefOp>();
    ConstrainRefRemappings translations;

    auto lattice = solver.lookupState<ConstrainRefLattice>(fnCall.getOperation());
    debug::ensure(lattice, "could not find lattice for call operation");

    // Map fn parameters to args in the call op
    for (unsigned i = 0; i < fn.getNumArguments(); i++) {
      auto prefix = ConstrainRef(fn.getArgument(i));
      auto val = lattice->getOrDefault(fnCall.getOperand(i));
      translations.push_back({prefix, val});
    }
    auto &childAnalysis = am.getChildAnalysis<ConstraintDependencyGraphAnalysis>(calledStruct);
    if (!childAnalysis.constructed()) {
      debug::ensure(
          mlir::succeeded(childAnalysis.constructCDG(solver, am)),
          "could not construct CDG for child struct"
      );
    }
    auto translatedCDG = childAnalysis.getCDG().translate(translations);

    // Now, union sets based on the translation
    // We should be able to just merge what is in the translatedCDG to the current CDG
    auto &tSets = translatedCDG.signalSets;
    for (auto lit = tSets.begin(); lit != tSets.end(); lit++) {
      if (!lit->isLeader()) {
        continue;
      }
      auto leader = lit->getData();
      for (auto mit = tSets.member_begin(lit); mit != tSets.member_end(); mit++) {
        signalSets.unionSets(leader, *mit);
      }
    }
    // And update the constant sets
    for (auto &[ref, constSet] : translatedCDG.constantSets) {
      constantSets[ref].insert(constSet.begin(), constSet.end());
    }
  });

  return mlir::success();
}

void ConstraintDependencyGraph::walkConstrainOp(
    mlir::DataFlowSolver &solver, mlir::Operation *emitOp
) {
  std::vector<ConstrainRef> signalUsages, constUsages;
  auto lattice = solver.lookupState<ConstrainRefLattice>(emitOp);
  debug::ensure(lattice, "failed to get lattice for emit operation");

  for (auto operand : emitOp->getOperands()) {
    auto latticeVal = lattice->getOrDefault(operand);
    for (auto &ref : latticeVal.foldToScalar()) {
      if (ref.isConstant()) {
        constUsages.push_back(ref);
      } else {
        signalUsages.push_back(ref);
      }
    }
  }

  // Compute a transitive closure over the signals.
  if (!signalUsages.empty()) {
    auto it = signalUsages.begin();
    auto leader = signalSets.getOrInsertLeaderValue(*it);
    for (it++; it != signalUsages.end(); it++) {
      signalSets.unionSets(leader, *it);
    }
  }
  // Also update constant references for each value.
  for (auto &sig : signalUsages) {
    constantSets[sig].insert(constUsages.begin(), constUsages.end());
  }
}

ConstraintDependencyGraph ConstraintDependencyGraph::translate(ConstrainRefRemappings translation) {
  ConstraintDependencyGraph res(mod, structDef);
  auto translate = [&translation](const ConstrainRef &elem
                   ) -> mlir::FailureOr<std::vector<ConstrainRef>> {
    std::vector<ConstrainRef> refs;
    for (auto &[prefix, vals] : translation) {
      if (!elem.isValidPrefix(prefix)) {
        continue;
      }

      if (vals.isArray()) {
        // Try to index into the array
        auto suffix = elem.getSuffix(prefix);
        debug::ensure(
            mlir::succeeded(suffix), "failure is nonsensical, we already checked for valid prefix"
        );

        auto [resolvedVals, _] = vals.extractOrIndex(suffix.value());
        auto folded = resolvedVals.foldToScalar();
        refs.insert(refs.end(), folded.begin(), folded.end());
      } else {
        for (auto &replacement : vals.getScalarValue()) {
          auto translated = elem.translate(prefix, replacement);
          if (mlir::succeeded(translated)) {
            refs.push_back(translated.value());
          }
        }
      }
    }
    if (refs.empty()) {
      return mlir::failure();
    }
    return refs;
  };

  for (auto leaderIt = signalSets.begin(); leaderIt != signalSets.end(); leaderIt++) {
    if (!leaderIt->isLeader()) {
      continue;
    }
    // translate everything in this set first
    std::vector<ConstrainRef> translatedSignals, translatedConsts;
    for (auto mit = signalSets.member_begin(leaderIt); mit != signalSets.member_end(); mit++) {
      auto member = translate(*mit);
      if (mlir::failed(member)) {
        continue;
      }
      for (auto &ref : *member) {
        if (ref.isConstant()) {
          translatedConsts.push_back(ref);
        } else {
          translatedSignals.push_back(ref);
        }
      }
      // Also add the constants from the original CDG
      auto &origConstSet = constantSets[*mit];
      translatedConsts.insert(translatedConsts.end(), origConstSet.begin(), origConstSet.end());
    }

    if (translatedSignals.empty()) {
      continue;
    }

    // Now we can insert the translated signals
    auto it = translatedSignals.begin();
    auto leader = *it;
    res.signalSets.insert(leader);
    for (it++; it != translatedSignals.end(); it++) {
      res.signalSets.insert(*it);
      res.signalSets.unionSets(leader, *it);
    }

    // And update the constant references
    for (auto &ref : translatedSignals) {
      res.constantSets[ref].insert(translatedConsts.begin(), translatedConsts.end());
    }
  }
  return res;
}

ConstrainRefSet ConstraintDependencyGraph::getConstrainingValues(const ConstrainRef &ref) const {
  ConstrainRefSet res;
  auto currRef = mlir::FailureOr<ConstrainRef>(ref);
  while (mlir::succeeded(currRef)) {
    // Add signals
    auto it = signalSets.findLeader(currRef.value());
    for (; it != signalSets.member_end(); it++) {
      if (currRef.value() != *it) {
        res.insert(*it);
      }
    }
    // Add constants
    auto constIt = constantSets.find(*currRef);
    if (constIt != constantSets.end()) {
      res.insert(constIt->second.begin(), constIt->second.end());
    }
    // Go to parent
    currRef = currRef->getParentPrefix();
  }
  return res;
}

/* ConstraintDependencyGraphModuleAnalysis */

ConstraintDependencyGraphModuleAnalysis::ConstraintDependencyGraphModuleAnalysis(
    mlir::Operation *op, mlir::AnalysisManager &am
) {
  if (auto modOp = mlir::dyn_cast<mlir::ModuleOp>(op)) {
    mlir::DataFlowConfig config;
    mlir::DataFlowSolver solver(config);
    dataflow::markAllOpsAsLive(solver, modOp);

    // The analysis is run at the module level so that lattices are computed
    // for global functions as well.
    solver.load<ConstrainRefAnalysis>();
    auto res = solver.initializeAndRun(modOp);
    debug::ensure(res.succeeded(), "solver failed to run on module!");

    modOp.walk([this, &solver, &am](StructDefOp s) {
      auto &csa = am.getChildAnalysis<ConstraintDependencyGraphAnalysis>(s);
      if (mlir::failed(csa.constructCDG(solver, am))) {
        auto error_message = "ConstraintDependencyGraphAnalysis failed to compute CDG for " +
                             mlir::Twine(s.getName());
        s->emitError(error_message);
        llvm::report_fatal_error(error_message);
      }
      dependencies[s] = csa.cdg;
    });
  } else {
    auto error_message =
        "ConstraintDependencyGraphModuleAnalysis expects provided op to be an mlir::ModuleOp!";
    op->emitError(error_message);
    llvm::report_fatal_error(error_message);
  }
}

} // namespace llzk
