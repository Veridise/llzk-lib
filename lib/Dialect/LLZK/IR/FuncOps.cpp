//===-- FuncOps.cpp - Func and call op implementations ----------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Adapted from the LLVM Project's lib/Dialect/Func/IR/FuncOps.cpp
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/LLZK/IR/Ops.h"
#include "llzk/Dialect/LLZK/Util/AttributeHelper.h"
#include "llzk/Dialect/LLZK/Util/SymbolHelper.h"

#include <mlir/IR/IRMapping.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/Interfaces/FunctionImplementation.h>

#include <llvm/ADT/MapVector.h>

namespace llzk {

using namespace mlir;

namespace {
/// Ensure that all symbols used within the FunctionType can be resolved.
inline LogicalResult
verifyTypeResolution(SymbolTableCollection &tables, Operation *origin, FunctionType funcType) {
  return llzk::verifyTypeResolution(
      tables, origin, ArrayRef<ArrayRef<Type>> {funcType.getInputs(), funcType.getResults()}
  );
}
} // namespace

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
  OwningEmitErrorFn emitErrorFunc = getEmitOpErrFn(this);
  // Ensure that only valid LLZK types are used for arguments and return.
  // @compute and @constrain functions also may not have AffineMapAttrs in their
  // parameters.
  FunctionType type = getFunctionType();
  llvm::ArrayRef<Type> inTypes = type.getInputs();
  for (auto ptr = inTypes.begin(); ptr < inTypes.end(); ptr++) {
    if (llzk::checkValidType(emitErrorFunc, *ptr).failed()) {
      return failure();
    }
    if (isInStruct() && (nameIsCompute() || nameIsConstrain()) && hasAffineMapAttr(*ptr)) {
      emitErrorFunc().append(
          "\"@", getName(), "\" parameters cannot contain affine map attributes but found ", *ptr
      );
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
        "\"@", FUNC_NAME_COMPUTE, "\" must have exactly one return type"
    );
  }
  if (failed(checkSelfType(tables, parent, resTypes.front(), origin, "return"))) {
    return failure();
  }

  // After the more specific checks (to ensure more specific error messages would be produced if
  // necessary), do the general check that all symbol references in the types are valid. The return
  // types were already checked so just check the input types.
  return verifyTypeResolution(tables, origin, funcType.getInputs());
}

LogicalResult
verifyFuncTypeConstrain(FuncOp &origin, SymbolTableCollection &tables, StructDefOp &parent) {
  FunctionType funcType = origin.getFunctionType();
  // Must return '()' type, i.e. have no return types
  if (funcType.getResults().size() != 0) {
    return origin.emitOpError() << "\"@" << FUNC_NAME_CONSTRAIN << "\" must have no return type";
  }

  // Type of the first parameter must match the parent StructDefOp of the current operation.
  llvm::ArrayRef<Type> inputTypes = funcType.getInputs();
  if (inputTypes.size() < 1) {
    return origin.emitOpError() << "\"@" << FUNC_NAME_CONSTRAIN
                                << "\" must have at least one input type";
  }
  if (failed(checkSelfType(tables, parent, inputTypes.front(), origin, "first input"))) {
    return failure();
  }

  // After the more specific checks (to ensure more specific error messages would be produced if
  // necessary), do the general check that all symbol references in the types are valid. There are
  // no return types, just check the remaining input types (the first was already checked via
  // the checkSelfType() call above).
  return verifyTypeResolution(tables, origin, inputTypes.drop_front());
}

} // namespace

LogicalResult FuncOp::verifySymbolUses(SymbolTableCollection &tables) {
  // Additional checks for the compute/constrain functions w/in a struct
  FailureOr<StructDefOp> parentStructOpt = getParentOfType<StructDefOp>(*this);
  if (succeeded(parentStructOpt)) {
    // Verify return type restrictions for functions within a StructDefOp
    if (nameIsCompute()) {
      return verifyFuncTypeCompute(*this, tables, parentStructOpt.value());
    } else if (nameIsConstrain()) {
      return verifyFuncTypeConstrain(*this, tables, parentStructOpt.value());
    }
  }
  // In the general case, verify symbol resolution in all input and output types.
  return verifyTypeResolution(tables, *this, getFunctionType());
}

SymbolRefAttr FuncOp::getFullyQualifiedName() {
  auto res = getPathFromRoot(*this);
  assert(succeeded(res));
  return res.value();
}

StructType FuncOp::getComputeSingleResultType() {
  assert(isStructCompute() && "violated implementation pre-condition");
  return getIfSingleton<StructType>(getResultTypes());
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

void CallOp::build(
    OpBuilder &odsBuilder, OperationState &odsState, TypeRange resultTypes, SymbolRefAttr callee,
    ValueRange argOperands
) {
  odsState.addTypes(resultTypes);
  odsState.addOperands(argOperands);
  Properties &props = affineMapHelpers::buildInstantiationAttrsEmpty<CallOp>(
      odsBuilder, odsState, static_cast<int32_t>(argOperands.size())
  );
  props.setCallee(callee);
}

void CallOp::build(
    OpBuilder &odsBuilder, OperationState &odsState, TypeRange resultTypes, SymbolRefAttr callee,
    ArrayRef<ValueRange> mapOperands, DenseI32ArrayAttr numDimsPerMap, ValueRange argOperands
) {
  odsState.addTypes(resultTypes);
  odsState.addOperands(argOperands);
  Properties &props = affineMapHelpers::buildInstantiationAttrs<CallOp>(
      odsBuilder, odsState, mapOperands, numDimsPerMap, argOperands.size()
  );
  props.setCallee(callee);
}

namespace {
enum class CalleeKind { Compute, Constrain, Other };

CalleeKind calleeNameToKind(StringRef tgtName) {
  if (FUNC_NAME_COMPUTE == tgtName) {
    return CalleeKind::Compute;
  } else if (FUNC_NAME_CONSTRAIN == tgtName) {
    return CalleeKind::Constrain;
  } else {
    return CalleeKind::Other;
  }
}

struct CallOpVerifier {
  CallOpVerifier(CallOp *c, StringRef tgtName) : callOp(c), tgtKind(calleeNameToKind(tgtName)) {}
  virtual ~CallOpVerifier() = default;

  LogicalResult verify() {
    // Rather than immediately returning on failure, we check all verifier steps and aggregate to
    // provide as many errors are possible in a single verifier run.
    LogicalResult aggregateResult = success();
    if (failed(verifyStructTarget())) {
      aggregateResult = failure();
    }
    if (failed(verifyInputs())) {
      aggregateResult = failure();
    }
    if (failed(verifyOutputs())) {
      aggregateResult = failure();
    }
    if (failed(verifyAffineMapParams())) {
      aggregateResult = failure();
    }
    return aggregateResult;
  }

protected:
  CallOp *callOp;
  CalleeKind tgtKind;

  virtual LogicalResult verifyStructTarget() = 0;
  virtual LogicalResult verifyInputs() = 0;
  virtual LogicalResult verifyOutputs() = 0;
  virtual LogicalResult verifyAffineMapParams() = 0;

  /// Ensure compute/constrain functions are only called by a like-named struct function.
  LogicalResult verifyStructTargetMatch() {
    switch (tgtKind) {
    case CalleeKind::Compute:
      return verifyInStructFunctionNamed<FUNC_NAME_COMPUTE, 32>(*callOp, [] {
        return llvm::SmallString<32>({"targeting \"@", FUNC_NAME_COMPUTE, "\" "});
      });
    case CalleeKind::Constrain:
      return verifyInStructFunctionNamed<FUNC_NAME_CONSTRAIN, 32>(*callOp, [] {
        return llvm::SmallString<32>({"targeting \"@", FUNC_NAME_CONSTRAIN, "\" "});
      });
    default:
      // Precondition: the target function is within a struct so only above names are valid
      // Note: This error can occur in the unknown case but in the known case, the symbol lookup
      // would actually fail before this step is reached.
      return callOp->emitOpError().append(
          "targeting a struct must call \"@", FUNC_NAME_COMPUTE, "\" or \"@", FUNC_NAME_CONSTRAIN,
          "\" only"
      );
    }
  }

  LogicalResult verifyNoAffineMapInstantiations() {
    if (!isNullOrEmpty(callOp->getMapOpGroupSizesAttr())) {
      // Tested in call_with_affinemap_fail.llzk
      return callOp->emitOpError().append(
          "can only have affine map instantiations when targeting a \"@", FUNC_NAME_COMPUTE,
          "\" function"
      );
    }
    // ASSERT: the check above is sufficient due to VerifySizesForMultiAffineOps trait.
    assert(isNullOrEmpty(callOp->getNumDimsPerMapAttr()));
    assert(callOp->getMapOperands().empty());
    return success();
  }
};

struct KnownTargetVerifier : public CallOpVerifier {
  KnownTargetVerifier(CallOp *c, SymbolLookupResult<FuncOp> &&tgtRes)
      : CallOpVerifier(c, tgtRes.get().getSymName()), tgt(*tgtRes), tgtType(tgt.getFunctionType()),
        includeSymNames(tgtRes.getIncludeSymNames()) {}

  LogicalResult verifyStructTarget() override {
    if (isInStruct(tgt.getOperation())) {
      // When the target is within a struct, check restrictions on the name.
      return CallOpVerifier::verifyStructTargetMatch();
    } else {
      // No target name restrictions when the target is a global function.
      return success();
    }
  }

  LogicalResult verifyInputs() override {
    return verifyTypesMatch(callOp->getArgOperands().getTypes(), tgtType.getInputs(), "operand");
  }

  LogicalResult verifyOutputs() override {
    return verifyTypesMatch(callOp->getResultTypes(), tgtType.getResults(), "result");
  }

  LogicalResult verifyAffineMapParams() override {
    if (CalleeKind::Compute == tgtKind && isInStruct(tgt.getOperation())) {
      // Return type should be a single StructType. If that is not the case here, just bail without
      // producing an error. The combination of this KnownTargetVerifier resolving the callee to a
      // specific FuncOp and verifyFuncTypeCompute() ensuring all FUNC_NAME_COMPUTE FuncOps have a
      // single StructType return value will produce a more relevant error message in that case.
      if (StructType retTy = callOp->getComputeSingleResultType()) {
        if (ArrayAttr params = retTy.getParams()) {
          // Collect the struct parameters that are defined via AffineMapAttr
          SmallVector<AffineMapAttr> mapAttrs;
          for (Attribute a : params) {
            if (AffineMapAttr m = dyn_cast<AffineMapAttr>(a)) {
              mapAttrs.push_back(m);
            }
          }
          return affineMapHelpers::verifyAffineMapInstantiations(
              callOp->getMapOperands(), callOp->getNumDimsPerMap(), mapAttrs, *callOp
          );
        }
      }
      return success();
    } else {
      // Global functions and constrain functions cannot have affine map instantiations.
      return verifyNoAffineMapInstantiations();
    }
  }

private:
  template <typename T>
  LogicalResult
  verifyTypesMatch(ValueTypeRange<T> callOpTypes, ArrayRef<Type> tgtTypes, const char *aspect) {
    if (tgtTypes.size() != callOpTypes.size()) {
      return callOp->emitOpError()
          .append("incorrect number of ", aspect, "s for callee, expected ", tgtTypes.size())
          .attachNote(tgt.getLoc())
          .append("callee defined here");
    }
    for (unsigned i = 0, e = tgtTypes.size(); i != e; ++i) {
      if (!typesUnify(callOpTypes[i], tgtTypes[i], includeSymNames)) {
        return callOp->emitOpError().append(
            aspect, " type mismatch: expected type ", tgtTypes[i], ", but found ", callOpTypes[i],
            " for ", aspect, " number ", i
        );
      }
    }
    return success();
  }

  FuncOp tgt;
  FunctionType tgtType;
  std::vector<llvm::StringRef> includeSymNames;
};

/// Version of checkSelfType() that performs the subset of verification checks that can be done when
/// the exact target of the `CallOp` is unknown.
LogicalResult checkSelfTypeUnknownTarget(
    StringAttr expectedParamName, Type actualType, CallOp *origin, const char *aspect
) {
  if (!llvm::isa<TypeVarType>(actualType) ||
      llvm::cast<TypeVarType>(actualType).getRefName() != expectedParamName) {
    // Tested in function_restrictions_fail.llzk:
    //    Non-tvar for constrain input via "call_target_constrain_without_self_non_struct"
    //    Non-tvar for compute output via "call_target_compute_wrong_type_ret"
    //    Wrong tvar for constrain input via "call_target_constrain_without_self_wrong_tvar_param"
    //    Wrong tvar for compute output via "call_target_compute_wrong_tvar_param_ret"
    return origin->emitOpError().append(
        "target \"@", origin->getCallee().getLeafReference().getValue(), "\" expected ", aspect,
        " type '!", TypeVarType::name, "<@", expectedParamName.getValue(), ">' but found ",
        actualType
    );
  }
  return success();
}

/// Precondition: the CallOp callee references a parameter of the CallOp's parent struct. This
/// creates a restriction that the referenced parameter must be instantiated with a StructType.
/// Hence, the call must target a function within a struct, not a global function, so the callee
/// name must be `compute` or `constrain`, nothing else.
/// Normally, full verification of the `compute` and `constrain` callees is done via
/// KnownTargetVerifier, which checks that input and output types of the caller match the callee,
/// plus verifyFuncTypeCompute() when the callee is `compute` or verifyFuncTypeConstrain() when
/// the callee is `constrain`. Those checks can take place after all parameterized structs are
/// instantiated (and thus the call target is known). For now, only minimal checks can be done.
struct UnknownTargetVerifier : public CallOpVerifier {
  UnknownTargetVerifier(CallOp *c, SymbolRefAttr callee)
      : CallOpVerifier(c, callee.getLeafReference().getValue()), calleeAttr(callee) {}

  LogicalResult verifyStructTarget() override {
    // Since the target is known to be within a struct, just check restrictions on the name. It can
    // be `compute` or `constrain`, nothing else
    return CallOpVerifier::verifyStructTargetMatch();
  }

  LogicalResult verifyInputs() override {
    if (CalleeKind::Compute == tgtKind) {
      // Without known target, no additional checks can be done.
    } else if (CalleeKind::Constrain == tgtKind) {
      // Without known target, this can only check that the first input is VarType using the same
      // struct parameter as the base of the callee (later replaced with the target struct's type).
      Operation::operand_type_range inputTypes = callOp->getArgOperands().getTypes();
      if (inputTypes.size() < 1) {
        // Tested in function_restrictions_fail.llzk
        return callOp->emitOpError()
               << "target \"@" << FUNC_NAME_CONSTRAIN << "\" must have at least one input type";
      }
      return checkSelfTypeUnknownTarget(
          calleeAttr.getRootReference(), inputTypes.front(), callOp, "first input"
      );
    }
    return success();
  }

  LogicalResult verifyOutputs() override {
    if (CalleeKind::Compute == tgtKind) {
      // Without known target, this can only check that the function returns VarType using the same
      // struct parameter as the base of the callee (later replaced with the target struct's type).
      Operation::result_type_range resTypes = callOp->getResultTypes();
      if (resTypes.size() != 1) {
        // Tested in function_restrictions_fail.llzk
        return callOp->emitOpError().append(
            "target \"@", FUNC_NAME_COMPUTE, "\" must have exactly one return type"
        );
      }
      return checkSelfTypeUnknownTarget(
          calleeAttr.getRootReference(), resTypes.front(), callOp, "return"
      );
    } else if (CalleeKind::Constrain == tgtKind) {
      // Without known target, this can only check that the function has no return
      if (callOp->getNumResults() != 0) {
        // Tested in function_restrictions_fail.llzk
        return callOp->emitOpError()
               << "target \"@" << FUNC_NAME_CONSTRAIN << "\" must have no return type";
      }
    }
    return success();
  }

  LogicalResult verifyAffineMapParams() override {
    if (CalleeKind::Compute == tgtKind) {
      // Without known target, no additional checks can be done.
    } else if (CalleeKind::Constrain == tgtKind) {
      // Without known target, this can only check that there are no affine map instantiations.
      return verifyNoAffineMapInstantiations();
    }
    return success();
  }

private:
  SymbolRefAttr calleeAttr;
};

} // namespace

LogicalResult CallOp::verifySymbolUses(SymbolTableCollection &tables) {
  // First, verify symbol resolution in all input and output types.
  if (failed(verifyTypeResolution(tables, *this, getCalleeType()))) {
    return failure(); // verifyTypeResolution() already emits a sufficient error message
  }

  // Check that the callee attribute was specified.
  SymbolRefAttr calleeAttr = getCalleeAttr();
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
  return FunctionType::get(getContext(), getArgOperands().getTypes(), getResultTypes());
}

namespace {

bool calleeIsStructFunctionImpl(
    const char *funcName, SymbolRefAttr callee, llvm::function_ref<StructType()> getType
) {
  if (callee.getLeafReference() == funcName) {
    if (StructType t = getType()) {
      // If the name ref within the StructType matches the `callee` prefix (i.e. sans the function
      // name itself), then the `callee` target must be within a StructDefOp because validation
      // checks elsewhere ensure that every StructType references a StructDefOp (i.e. the `callee`
      // function is not simply a global function nested within a ModuleOp)
      return t.getNameRef() == getPrefixAsSymbolRefAttr(callee);
    }
  }
  return false;
}

} // namespace

bool CallOp::calleeIsStructCompute() {
  return calleeIsStructFunctionImpl(FUNC_NAME_COMPUTE, getCallee(), [this]() {
    return this->getComputeSingleResultType();
  });
}

bool CallOp::calleeIsStructConstrain() {
  return calleeIsStructFunctionImpl(FUNC_NAME_CONSTRAIN, getCallee(), [this]() {
    return getAtIndex<StructType>(this->getArgOperands().getTypes(), 0);
  });
}

StructType CallOp::getComputeSingleResultType() {
  assert(calleeIsCompute() && "violated implementation pre-condition");
  return getIfSingleton<StructType>(getResultTypes());
}

/// Return the callee of this operation.
CallInterfaceCallable CallOp::getCallableForCallee() { return getCalleeAttr(); }

/// Set the callee for this operation.
void CallOp::setCalleeFromCallable(CallInterfaceCallable callee) {
  setCalleeAttr(callee.get<SymbolRefAttr>());
}

} // namespace llzk
