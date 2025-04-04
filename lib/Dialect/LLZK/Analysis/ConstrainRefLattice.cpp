//===-- ConstrainRefLattice.cpp - ConstrainRef lattice & utils --*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/LLZK/Analysis/ConstrainRefLattice.h"
#include "llzk/Dialect/LLZK/Analysis/ConstraintDependencyGraph.h"
#include "llzk/Dialect/LLZK/Analysis/DenseAnalysis.h"
#include "llzk/Dialect/LLZK/Util/Hash.h"
#include "llzk/Dialect/LLZK/Util/SymbolHelper.h"

#include <mlir/Analysis/DataFlow/DeadCodeAnalysis.h>
#include <mlir/IR/Value.h>

#include <llvm/Support/Debug.h>

#include <numeric>
#include <unordered_set>

#define DEBUG_TYPE "llzk-constrain-ref-lattice"

namespace llzk {

/* ConstrainRefLatticeValue */

mlir::ChangeResult ConstrainRefLatticeValue::insert(const ConstrainRef &rhs) {
  auto rhsVal = ConstrainRefLatticeValue(rhs);
  if (isScalar()) {
    return updateScalar(rhsVal.getScalarValue());
  } else {
    return foldAndUpdate(rhsVal);
  }
}

std::pair<ConstrainRefLatticeValue, mlir::ChangeResult>
ConstrainRefLatticeValue::translate(const TranslationMap &translation) const {
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
ConstrainRefLatticeValue::referenceField(SymbolLookupResult<FieldDefOp> fieldRef) const {
  ConstrainRefIndex idx(fieldRef);
  auto transform = [&idx](const ConstrainRef &r) -> ConstrainRef { return r.createChild(idx); };
  return elementwiseTransform(transform);
}

std::pair<ConstrainRefLatticeValue, mlir::ChangeResult>
ConstrainRefLatticeValue::extract(const std::vector<ConstrainRefIndex> &indices) const {
  if (isArray()) {
    ensure(indices.size() <= getNumArrayDims(), "invalid extract array operands");

    // First, compute what chunk(s) to index
    std::vector<size_t> currIdxs {0};
    for (unsigned i = 0; i < indices.size(); i++) {
      auto &idx = indices[i];
      auto currDim = getArrayDim(i);

      std::vector<size_t> newIdxs;
      ensure(idx.isIndex() || idx.isIndexRange(), "wrong type of index for array");
      if (idx.isIndex()) {
        auto idxVal = fromAPInt(idx.getIndex());
        std::transform(
            currIdxs.begin(), currIdxs.end(), std::back_inserter(newIdxs),
            [&currDim, &idxVal](size_t j) { return j * currDim + idxVal; }
        );
      } else {
        auto [low, high] = idx.getIndexRange();
        for (auto idxVal = fromAPInt(low); idxVal < fromAPInt(high); idxVal++) {
          std::transform(
              currIdxs.begin(), currIdxs.end(), std::back_inserter(newIdxs),
              [&currDim, &idxVal](size_t j) { return j * currDim + idxVal; }
          );
        }
      }

      currIdxs = newIdxs;
    }
    std::vector<int64_t> newArrayDims;
    size_t chunkSz = 1;
    for (unsigned i = indices.size(); i < getNumArrayDims(); i++) {
      auto dim = getArrayDim(i);
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
      auto transform = [&idx](const ConstrainRef &r) -> ConstrainRef { return r.createChild(idx); };
      auto [newVal, transformRes] = currVal.elementwiseTransform(transform);
      currVal = std::move(newVal);
      res |= transformRes;
    }
    return {currVal, res};
  }
}

mlir::ChangeResult ConstrainRefLatticeValue::translateScalar(const TranslationMap &translation) {
  auto res = mlir::ChangeResult::NoChange;
  // copy the current value
  auto currVal = getScalarValue();
  // reset this value
  getValue() = ScalarTy();
  for (auto &[ref, val] : translation) {
    auto it = currVal.find(ref);
    if (it != currVal.end()) {
      res |= update(val);
    }
  }
  return res;
}

std::pair<ConstrainRefLatticeValue, mlir::ChangeResult>
ConstrainRefLatticeValue::elementwiseTransform(
    llvm::function_ref<ConstrainRef(const ConstrainRef &)> transform
) const {
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

mlir::raw_ostream &operator<<(mlir::raw_ostream &os, const ConstrainRefLatticeValue &v) {
  v.print(os);
  return os;
}

/* ConstrainRefLattice */

mlir::FailureOr<ConstrainRef> ConstrainRefLattice::getSourceRef(mlir::Value val) {
  if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(val)) {
    return ConstrainRef(blockArg);
  } else if (auto defOp = val.getDefiningOp()) {
    if (auto constFelt = mlir::dyn_cast<FeltConstantOp>(defOp)) {
      return ConstrainRef(constFelt);
    } else if (auto constIdx = mlir::dyn_cast<mlir::arith::ConstantIndexOp>(defOp)) {
      return ConstrainRef(constIdx);
    } else if (auto readConst = mlir::dyn_cast<ConstReadOp>(defOp)) {
      return ConstrainRef(readConst);
    }
  }
  return mlir::failure();
}

void ConstrainRefLattice::print(mlir::raw_ostream &os) const {
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

mlir::ChangeResult ConstrainRefLattice::setValues(const ValueMap &rhs) {
  auto res = mlir::ChangeResult::NoChange;

  for (auto &[v, s] : rhs) {
    res |= setValue(v, s);
  }
  return res;
}

ConstrainRefLatticeValue ConstrainRefLattice::getOrDefault(mlir::Value v) const {
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

ConstrainRefLatticeValue ConstrainRefLattice::getReturnValue(unsigned i) const {
  auto op = this->getPoint().get<mlir::Operation *>();
  if (auto retOp = mlir::dyn_cast<ReturnOp>(op)) {
    if (i >= retOp.getNumOperands()) {
      llvm::report_fatal_error("return value requested is out of range");
    }
    return this->getOrDefault(retOp.getOperand(i));
  }
  return ConstrainRefLatticeValue();
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const ConstrainRefLattice &lattice) {
  lattice.print(os);
  return os;
}

} // namespace llzk
