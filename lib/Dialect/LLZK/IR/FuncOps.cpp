// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "llzk/Dialect/LLZK/IR/Ops.h"
#include "llzk/Dialect/LLZK/Util/SymbolHelper.h"

#include <mlir/IR/IRMapping.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/Interfaces/FunctionImplementation.h>

#include <llvm/ADT/MapVector.h>

namespace llzk {

using namespace mlir;

//===----------------------------------------------------------------------===//
// FuncOp
//===----------------------------------------------------------------------===//

FuncOp FuncOp::create(
    Location location, StringRef name, FunctionType type, ArrayRef<NamedAttribute> attrs
) {
  return delegate_to_build<FuncOp>(location, name, type, attrs);
}

FuncOp FuncOp::create(
    Location location, StringRef name, FunctionType type, Operation::dialect_attr_range attrs
) {
  SmallVector<NamedAttribute, 8> attrRef(attrs);
  return create(location, name, type, llvm::ArrayRef(attrRef));
}

FuncOp FuncOp::create(
    Location location, StringRef name, FunctionType type, ArrayRef<NamedAttribute> attrs,
    ArrayRef<DictionaryAttr> argAttrs
) {
  FuncOp func = create(location, name, type, attrs);
  func.setAllArgAttrs(argAttrs);
  return func;
}

void FuncOp::build(
    OpBuilder &builder, OperationState &state, StringRef name, FunctionType type,
    ArrayRef<NamedAttribute> attrs, ArrayRef<DictionaryAttr> argAttrs
) {
  state.addAttribute(SymbolTable::getSymbolAttrName(), builder.getStringAttr(name));
  state.addAttribute(getFunctionTypeAttrName(state.name), TypeAttr::get(type));
  state.attributes.append(attrs.begin(), attrs.end());
  state.addRegion();

  if (argAttrs.empty()) {
    return;
  }
  assert(type.getNumInputs() == argAttrs.size());
  function_interface_impl::addArgAndResultAttrs(
      builder, state, argAttrs, /*resultAttrs=*/std::nullopt, getArgAttrsAttrName(state.name),
      getResAttrsAttrName(state.name)
  );
}

ParseResult FuncOp::parse(OpAsmParser &parser, OperationState &result) {
  auto buildFuncType = [](Builder &builder, ArrayRef<Type> argTypes, ArrayRef<Type> results,
                          function_interface_impl::VariadicFlag,
                          std::string &) { return builder.getFunctionType(argTypes, results); };

  return function_interface_impl::parseFunctionOp(
      parser, result, /*allowVariadic=*/false, getFunctionTypeAttrName(result.name), buildFuncType,
      getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name)
  );
}

void FuncOp::print(OpAsmPrinter &p) {
  function_interface_impl::printFunctionOp(
      p, *this, /*isVariadic=*/false, getFunctionTypeAttrName(), getArgAttrsAttrName(),
      getResAttrsAttrName()
  );
}

/// Clone the internal blocks from this function into dest and all attributes
/// from this function to dest.
void FuncOp::cloneInto(FuncOp dest, IRMapping &mapper) {
  // Add the attributes of this function to dest.
  llvm::MapVector<StringAttr, Attribute> newAttrMap;
  for (const auto &attr : dest->getAttrs()) {
    newAttrMap.insert({attr.getName(), attr.getValue()});
  }
  for (const auto &attr : (*this)->getAttrs()) {
    newAttrMap.insert({attr.getName(), attr.getValue()});
  }

  auto newAttrs =
      llvm::to_vector(llvm::map_range(newAttrMap, [](std::pair<StringAttr, Attribute> attrPair) {
    return NamedAttribute(attrPair.first, attrPair.second);
  }));
  dest->setAttrs(DictionaryAttr::get(getContext(), newAttrs));

  // Clone the body.
  getBody().cloneInto(&dest.getBody(), mapper);
}

/// Create a deep copy of this function and all of its blocks, remapping
/// any operands that use values outside of the function using the map that is
/// provided (leaving them alone if no entry is present). Replaces references
/// to cloned sub-values with the corresponding value that is copied, and adds
/// those mappings to the mapper.
FuncOp FuncOp::clone(IRMapping &mapper) {
  // Create the new function.
  FuncOp newFunc = cast<FuncOp>(getOperation()->cloneWithoutRegions());

  // If the function has a body, then the user might be deleting arguments to
  // the function by specifying them in the mapper. If so, we don't add the
  // argument to the input type vector.
  if (!isExternal()) {
    FunctionType oldType = getFunctionType();

    unsigned oldNumArgs = oldType.getNumInputs();
    SmallVector<Type, 4> newInputs;
    newInputs.reserve(oldNumArgs);
    for (unsigned i = 0; i != oldNumArgs; ++i) {
      if (!mapper.contains(getArgument(i))) {
        newInputs.push_back(oldType.getInput(i));
      }
    }

    /// If any of the arguments were dropped, update the type and drop any
    /// necessary argument attributes.
    if (newInputs.size() != oldNumArgs) {
      newFunc.setType(FunctionType::get(oldType.getContext(), newInputs, oldType.getResults()));

      if (ArrayAttr argAttrs = getAllArgAttrs()) {
        SmallVector<Attribute> newArgAttrs;
        newArgAttrs.reserve(newInputs.size());
        for (unsigned i = 0; i != oldNumArgs; ++i) {
          if (!mapper.contains(getArgument(i))) {
            newArgAttrs.push_back(argAttrs[i]);
          }
        }
        newFunc.setAllArgAttrs(newArgAttrs);
      }
    }
  }

  /// Clone the current function into the new one and return it.
  cloneInto(newFunc, mapper);
  return newFunc;
}

FuncOp FuncOp::clone() {
  IRMapping mapper;
  return clone(mapper);
}

bool FuncOp::hasArgPublicAttr(unsigned index) {
  if (index < this->getNumArguments()) {
    DictionaryAttr res = function_interface_impl::getArgAttrDict(*this, index);
    return res ? res.contains(PublicAttr::name) : false;
  } else {
    // TODO: print error? requested attribute for non-existant argument index
    return false;
  }
}

LogicalResult FuncOp::verify() {
  auto emitErrorFunc = [op = this->getOperation()]() -> InFlightDiagnostic {
    return op->emitOpError();
  };
  // Ensure that only valid LLZK types are used for arguments and return
  FunctionType type = getFunctionType();
  llvm::ArrayRef<Type> inTypes = type.getInputs();
  for (auto ptr = inTypes.begin(); ptr < inTypes.end(); ptr++) {
    if (llzk::checkValidType(emitErrorFunc, *ptr).failed()) {
      return failure();
    }
  }
  llvm::ArrayRef<Type> resTypes = type.getResults();
  for (auto ptr = resTypes.begin(); ptr < resTypes.end(); ptr++) {
    if (llzk::checkValidType(emitErrorFunc, *ptr).failed()) {
      return failure();
    }
  }
  return success();
}

namespace {

LogicalResult
verifyFuncTypeCompute(FuncOp &origin, SymbolTableCollection &tables, StructDefOp &parent) {
  FunctionType funcType = origin.getFunctionType();
  llvm::ArrayRef<Type> resTypes = funcType.getResults();
  // Must return type of parent struct
  if (resTypes.size() != 1) {
    return origin.emitOpError().append(
        "\"@", llzk::FUNC_NAME_COMPUTE, "\" must have exactly one return type"
    );
  }
  if (failed(checkSelfType(tables, parent, resTypes.front(), origin, "return"))) {
    return failure();
  }

  // After the more specific checks (to ensure more specific error messages would be produced if
  // necessary), do the general check that all symbol references in the types are valid. The return
  // types were already checked so just check the input types.
  return verifyTypeResolution(tables, funcType.getInputs(), origin);
}

LogicalResult
verifyFuncTypeConstrain(FuncOp &origin, SymbolTableCollection &tables, StructDefOp &parent) {
  FunctionType funcType = origin.getFunctionType();
  // Must return '()' type, i.e. have no return types
  if (funcType.getResults().size() != 0) {
    return origin.emitOpError() << "\"@" << llzk::FUNC_NAME_CONSTRAIN
                                << "\" must have no return type";
  }

  // Type of the first parameter must match the parent StructDefOp of the current operation.
  llvm::ArrayRef<Type> inputTypes = funcType.getInputs();
  if (inputTypes.size() < 1) {
    return origin.emitOpError() << "\"@" << llzk::FUNC_NAME_CONSTRAIN
                                << "\" must have at least one input type";
  }
  if (failed(checkSelfType(tables, parent, inputTypes.front(), origin, "first input"))) {
    return failure();
  }

  // After the more specific checks (to ensure more specific error messages would be produced if
  // necessary), do the general check that all symbol references in the types are valid. There are
  // no return types, just check the remaining input types (the first was already checked via
  // the checkSelfType() call above).
  return verifyTypeResolution(tables, inputTypes.begin() + 1, inputTypes.end(), origin);
}

} // namespace

LogicalResult FuncOp::verifySymbolUses(SymbolTableCollection &tables) {
  // Additional checks for the compute/constrain functions w/in a struct
  FailureOr<StructDefOp> parentStructOpt = getParentOfType<StructDefOp>(*this);
  if (succeeded(parentStructOpt)) {
    // Verify return type restrictions for functions within a StructDefOp
    llvm::StringRef funcName = getSymName();
    if (llzk::FUNC_NAME_COMPUTE == funcName) {
      return verifyFuncTypeCompute(*this, tables, parentStructOpt.value());
    } else if (llzk::FUNC_NAME_CONSTRAIN == funcName) {
      return verifyFuncTypeConstrain(*this, tables, parentStructOpt.value());
    }
  }
  // In the general case, verify all input and output types are valid. Check both
  //  before returning to present all applicable type errors in one compilation.
  FunctionType funcType = getFunctionType();
  LogicalResult a = verifyTypeResolution(tables, funcType.getResults(), *this);
  LogicalResult b = verifyTypeResolution(tables, funcType.getInputs(), *this);
  return LogicalResult::success(succeeded(a) && succeeded(b));
}

SymbolRefAttr FuncOp::getFullyQualifiedName() const {
  auto res = getPathFromRoot(*const_cast<FuncOp *>(this));
  assert(succeeded(res));
  return res.value();
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

LogicalResult ReturnOp::verify() {
  auto function = cast<FuncOp>((*this)->getParentOp());

  // The operand number and types must match the function signature.
  const auto results = function.getFunctionType().getResults();
  if (getNumOperands() != results.size()) {
    return emitOpError("has ") << getNumOperands() << " operands, but enclosing function (@"
                               << function.getName() << ") returns " << results.size();
  }

  for (unsigned i = 0, e = results.size(); i != e; ++i) {
    if (!typesUnify(getOperand(i).getType(), results[i])) {
      return emitError() << "type of return operand " << i << " (" << getOperand(i).getType()
                         << ") doesn't match function result type (" << results[i] << ")"
                         << " in function @" << function.getName();
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

namespace {

struct CallOpVerifier {
  CallOpVerifier(CallOp *c) : callOp(c) {}
  virtual ~CallOpVerifier() {};

  LogicalResult verify() {
    // Rather than immediately returning on failure, we check all verifier steps and aggregate to
    // provide as many errors are possible in a single verifier run.
    LogicalResult aggregateResult = success();
    if (failed(verifyInputs())) {
      aggregateResult = failure();
    }
    if (failed(verifyOutputs())) {
      aggregateResult = failure();
    }
    if (failed(verifyStructTarget())) {
      aggregateResult = failure();
    }
    return aggregateResult;
  }

protected:
  CallOp *callOp;

  virtual LogicalResult verifyInputs() = 0;
  virtual LogicalResult verifyOutputs() = 0;
  virtual LogicalResult verifyStructTarget() = 0;

  LogicalResult verifyStructTarget(StringRef tgtName) {
    if (tgtName.compare(FUNC_NAME_COMPUTE) == 0) {
      return verifyInStructFunctionNamed<FUNC_NAME_COMPUTE, 32>(*callOp, [] {
        return llvm::SmallString<32>({"targeting \"@", FUNC_NAME_COMPUTE, "\" "});
      });
    } else if (tgtName.compare(FUNC_NAME_CONSTRAIN) == 0) {
      return verifyInStructFunctionNamed<FUNC_NAME_CONSTRAIN, 32>(*callOp, [] {
        return llvm::SmallString<32>({"targeting \"@", FUNC_NAME_CONSTRAIN, "\" "});
      });
    }
    return success();
  }
};

struct KnownTargetVerifier : public CallOpVerifier {
  KnownTargetVerifier(CallOp *c, SymbolLookupResult<FuncOp> &&tgtRes)
      : CallOpVerifier(c), tgt(*tgtRes), tgtType(tgt.getFunctionType()),
        includeSymNames(tgtRes.getIncludeSymNames()) {}

  LogicalResult verifyInputs() override {
    if (tgtType.getNumInputs() != callOp->getNumOperands()) {
      return callOp->emitOpError()
          .append("incorrect number of operands for callee, expected ", tgtType.getNumInputs())
          .attachNote(tgt.getLoc())
          .append("callee defined here");
    }
    for (unsigned i = 0, e = tgtType.getNumInputs(); i != e; ++i) {
      if (!typesUnify(callOp->getOperand(i).getType(), tgtType.getInput(i), includeSymNames)) {
        return callOp->emitOpError("operand type mismatch: expected type ")
               << tgtType.getInput(i) << ", but found " << callOp->getOperand(i).getType()
               << " for operand number " << i;
      }
    }
    return success();
  }

  LogicalResult verifyOutputs() override {
    if (tgtType.getNumResults() != callOp->getNumResults()) {
      return callOp->emitOpError()
          .append("incorrect number of results for callee, expected ", tgtType.getNumResults())
          .attachNote(tgt.getLoc())
          .append("callee defined here");
    }
    for (unsigned i = 0, e = tgtType.getNumResults(); i != e; ++i) {
      if (!typesUnify(callOp->getResult(i).getType(), tgtType.getResult(i), includeSymNames)) {
        return callOp->emitOpError("result type mismatch: expected type ")
               << tgtType.getResult(i) << ", but found " << callOp->getResult(i).getType()
               << " for result number " << i;
      }
    }
    return success();
  }

  LogicalResult verifyStructTarget() override {
    if (isInStruct(tgt.getOperation())) {
      return CallOpVerifier::verifyStructTarget(tgt.getSymName());
    }
    return success();
  }

private:
  FuncOp tgt;
  FunctionType tgtType;
  std::vector<llvm::StringRef> includeSymNames;
};

struct UnknownTargetVerifier : public CallOpVerifier {
  UnknownTargetVerifier(CallOp *c, SymbolRefAttr calleeAttr_)
      : CallOpVerifier(c), calleeAttr(calleeAttr_) {}

  LogicalResult verifyInputs() override { return success(); }
  LogicalResult verifyOutputs() override { return success(); }
  LogicalResult verifyStructTarget() override {
    return CallOpVerifier::verifyStructTarget(calleeAttr.getLeafReference().getValue());
  }

private:
  SymbolRefAttr calleeAttr;
};

} // namespace

LogicalResult CallOp::verifySymbolUses(SymbolTableCollection &tables) {
  // Check that the callee attribute was specified.
  SymbolRefAttr calleeAttr = (*this)->getAttrOfType<SymbolRefAttr>("callee");
  if (!calleeAttr) {
    return emitOpError("requires a 'callee' symbol reference attribute");
  }

  // If the callee references a parameter of the struct where this call appears, perform the subset
  // of checks that can be done even though the target is unknown.
  if (calleeAttr.getNestedReferences().size() == 1) {
    FailureOr<StructDefOp> parent = getParentOfType<StructDefOp>(*this);
    if (succeeded(parent) && parent->hasParamNamed(calleeAttr.getRootReference())) {
      return UnknownTargetVerifier(this, calleeAttr).verify();
    }
  }

  // Otherwise, callee must be specified via full path from the root module. Perform the full set of
  // checks against the known target function.
  auto tgtOpt = lookupTopLevelSymbol<FuncOp>(tables, calleeAttr, *this);
  if (failed(tgtOpt)) {
    return this->emitError() << "expected '" << FuncOp::getOperationName() << "' named \""
                             << calleeAttr << "\"";
  }
  return KnownTargetVerifier(this, std::move(*tgtOpt)).verify();
}

FunctionType CallOp::getCalleeType() {
  return FunctionType::get(getContext(), getOperandTypes(), getResultTypes());
}

/// Get the argument operands to the called function.
CallOp::operand_range CallOp::getArgOperands() { return {arg_operand_begin(), arg_operand_end()}; }

mlir::MutableOperandRange CallOp::getArgOperandsMutable() { return getOperandsMutable(); }

/// Return the callee of this operation.
mlir::CallInterfaceCallable CallOp::getCallableForCallee() {
  return (*this)->getAttrOfType<mlir::SymbolRefAttr>("callee");
}

/// Set the callee for this operation.
void CallOp::setCalleeFromCallable(mlir::CallInterfaceCallable callee) {
  (*this)->setAttr("callee", callee.get<mlir::SymbolRefAttr>());
}

} // namespace llzk
