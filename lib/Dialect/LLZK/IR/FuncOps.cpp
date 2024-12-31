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
    DictionaryAttr res = mlir::function_interface_impl::getArgAttrDict(*this, index);
    return res ? res.contains(PublicAttr::name) : false;
  } else {
    // TODO: print error? requested attribute for non-existant argument index
    return false;
  }
}

mlir::LogicalResult FuncOp::verify() {
  auto emitErrorFunc = [op = this->getOperation()]() -> mlir::InFlightDiagnostic {
    return op->emitOpError();
  };
  // Ensure that only valid LLZK types are used for arguments and return
  FunctionType type = getFunctionType();
  llvm::ArrayRef<mlir::Type> inTypes = type.getInputs();
  for (auto ptr = inTypes.begin(); ptr < inTypes.end(); ptr++) {
    if (llzk::checkValidType(emitErrorFunc, *ptr).failed()) {
      return mlir::failure();
    }
  }
  llvm::ArrayRef<mlir::Type> resTypes = type.getResults();
  for (auto ptr = resTypes.begin(); ptr < resTypes.end(); ptr++) {
    if (llzk::checkValidType(emitErrorFunc, *ptr).failed()) {
      return mlir::failure();
    }
  }
  return mlir::success();
}

namespace {

mlir::LogicalResult
verifyFuncTypeCompute(FuncOp &origin, SymbolTableCollection &symbolTable, StructDefOp &parent) {
  mlir::FunctionType funcType = origin.getFunctionType();
  llvm::ArrayRef<mlir::Type> resTypes = funcType.getResults();
  // Must return type of parent struct
  if (resTypes.size() != 1) {
    return origin.emitOpError().append(
        "\"@", llzk::FUNC_NAME_COMPUTE, "\" must have exactly one return type"
    );
  }
  if (mlir::failed(checkSelfType(symbolTable, parent, resTypes.front(), origin, "return"))) {
    return mlir::failure();
  }

  // After the more specific checks (to ensure more specific error messages would be produced if
  // necessary), do the general check that all symbol references in the types are valid. The return
  // types were already checked so just check the input types.
  return verifyTypeResolution(symbolTable, funcType.getInputs(), origin);
}

mlir::LogicalResult
verifyFuncTypeConstrain(FuncOp &origin, SymbolTableCollection &symbolTable, StructDefOp &parent) {
  mlir::FunctionType funcType = origin.getFunctionType();
  // Must return '()' type, i.e. have no return types
  if (funcType.getResults().size() != 0) {
    return origin.emitOpError() << "\"@" << llzk::FUNC_NAME_CONSTRAIN
                                << "\" must have no return type";
  }

  // Type of the first parameter must match the parent StructDefOp of the current operation.
  llvm::ArrayRef<mlir::Type> inputTypes = funcType.getInputs();
  if (inputTypes.size() < 1) {
    return origin.emitOpError() << "\"@" << llzk::FUNC_NAME_CONSTRAIN
                                << "\" must have at least one input type";
  }
  if (mlir::failed(checkSelfType(symbolTable, parent, inputTypes.front(), origin, "first input"))) {
    return mlir::failure();
  }

  // After the more specific checks (to ensure more specific error messages would be produced if
  // necessary), do the general check that all symbol references in the types are valid. There are
  // no return types, just check the remaining input types (the first was already checked via
  // the checkSelfType() call above).
  return verifyTypeResolution(symbolTable, inputTypes.begin() + 1, inputTypes.end(), origin);
}

} // namespace

mlir::LogicalResult FuncOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // Additional checks for the compute/constrain functions w/in a struct
  mlir::FailureOr<StructDefOp> parentStructOpt = getParentOfType<StructDefOp>(*this);
  if (mlir::succeeded(parentStructOpt)) {
    // Verify return type restrictions for functions within a StructDefOp
    llvm::StringRef funcName = getSymName();
    if (llzk::FUNC_NAME_COMPUTE == funcName) {
      return verifyFuncTypeCompute(*this, symbolTable, parentStructOpt.value());
    } else if (llzk::FUNC_NAME_CONSTRAIN == funcName) {
      return verifyFuncTypeConstrain(*this, symbolTable, parentStructOpt.value());
    }
  }
  // In the general case, verify all input and output types are valid. Check both
  //  before returning to present all applicable type errors in one compilation.
  mlir::FunctionType funcType = getFunctionType();
  mlir::LogicalResult a = verifyTypeResolution(symbolTable, funcType.getResults(), *this);
  mlir::LogicalResult b = verifyTypeResolution(symbolTable, funcType.getInputs(), *this);
  return mlir::LogicalResult::success(mlir::succeeded(a) && mlir::succeeded(b));
}

mlir::SymbolRefAttr FuncOp::getFullyQualifiedName() const {
  auto res = getPathFromRoot(*const_cast<FuncOp *>(this));
  assert(mlir::succeeded(res));
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

LogicalResult CallOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // Check that the callee attribute was specified.
  SymbolRefAttr fnAttr = (*this)->getAttrOfType<SymbolRefAttr>("callee");
  if (!fnAttr) {
    return emitOpError("requires a 'callee' symbol reference attribute");
  }
  // Call target must be specified via full path from the root module.
  auto tgtOpt = lookupTopLevelSymbol<FuncOp>(symbolTable, fnAttr, *this);
  if (mlir::failed(tgtOpt)) {
    return this->emitError() << "no '" << FuncOp::getOperationName() << "' named \"" << fnAttr
                             << "\"";
  }
  FuncOp tgt = tgtOpt.value().get();

  // Verify that the operand and result types match the callee.
  FunctionType fnType = tgt.getFunctionType();
  if (fnType.getNumInputs() != getNumOperands()) {
    return emitOpError("incorrect number of operands for callee");
  }

  for (unsigned i = 0, e = fnType.getNumInputs(); i != e; ++i) {
    if (!typesUnify(getOperand(i).getType(), fnType.getInput(i), tgtOpt->getIncludeSymNames())) {
      return emitOpError("operand type mismatch: expected type ")
             << fnType.getInput(i) << ", but found " << getOperand(i).getType()
             << " for operand number " << i;
    }
  }

  if (fnType.getNumResults() != getNumResults()) {
    return emitOpError("incorrect number of results for callee");
  }

  for (unsigned i = 0, e = fnType.getNumResults(); i != e; ++i) {
    if (!typesUnify(getResult(i).getType(), fnType.getResult(i), tgtOpt->getIncludeSymNames())) {
      return emitOpError("result type mismatch: expected type ")
             << fnType.getResult(i) << ", but found " << getResult(i).getType()
             << " for result number " << i;
    }
  }

  // Enforce restrictions on callers of compute/constrain functions within structs.
  if (isInStruct(tgt.getOperation())) {
    if (tgt.getSymName().compare(FUNC_NAME_COMPUTE) == 0) {
      return verifyInStructFunctionNamed<FUNC_NAME_COMPUTE, 32>(*this, [] {
        return llvm::SmallString<32>({"targeting \"@", FUNC_NAME_COMPUTE, "\" "});
      });
    } else if (tgt.getSymName().compare(FUNC_NAME_CONSTRAIN) == 0) {
      return verifyInStructFunctionNamed<FUNC_NAME_CONSTRAIN, 32>(*this, [] {
        return llvm::SmallString<32>({"targeting \"@", FUNC_NAME_CONSTRAIN, "\" "});
      });
    }
  }

  return success();
}

FunctionType CallOp::getCalleeType() {
  return FunctionType::get(getContext(), getOperandTypes(), getResultTypes());
}

} // namespace llzk
