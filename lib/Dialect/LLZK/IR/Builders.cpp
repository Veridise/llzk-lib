#include "llzk/Dialect/LLZK/IR/Builders.h"

#include "llzk/Dialect/LLZK/Util/SymbolHelper.h"

#include <cassert>

using namespace mlir;

namespace llzk {

/* ModuleBuilder */

ModuleBuilder::ModuleBuilder(mlir::MLIRContext *context) : context(context) {
  auto dialect = context->getOrLoadDialect<llzk::LLZKDialect>();
  auto langAttr = StringAttr::get(context, dialect->getNamespace());
  rootModule = ModuleOp::create(UnknownLoc::get(context));
  rootModule->setAttr(llzk::LANG_ATTR_NAME, langAttr);
}

llzk::StructDefOp ModuleBuilder::insertEmptyStruct(std::string_view structName) {
  assert(structMap.find(structName) == structMap.end());

  OpBuilder opBuilder(rootModule.getBody(), rootModule.getBody()->begin());
  auto structNameAtrr = StringAttr::get(context, structName);
  auto structDef =
      opBuilder.create<llzk::StructDefOp>(UnknownLoc::get(context), structNameAtrr, nullptr);
  // populate the initial region
  auto &region = structDef.getRegion();
  if (region.empty()) {
    region.push_back(new mlir::Block());
  }
  structMap[structName] = structDef;

  return structDef;
}

llzk::FuncOp ModuleBuilder::insertComputeFn(llzk::StructDefOp *op) {
  OpBuilder opBuilder(op->getBody());
  assert(computeFnMap.find(op->getName()) == computeFnMap.end());

  auto structType = llzk::StructType::get(context, SymbolRefAttr::get(*op));

  auto fnOp = opBuilder.create<llzk::FuncOp>(
      UnknownLoc::get(context), StringAttr::get(context, llzk::FUNC_NAME_COMPUTE),
      FunctionType::get(context, {}, {structType})
  );
  fnOp.addEntryBlock();
  computeFnMap[op->getName()] = fnOp;
  return fnOp;
}

llzk::FuncOp ModuleBuilder::insertConstrainFn(llzk::StructDefOp *op) {
  assert(constrainFnMap.find(op->getName()) == constrainFnMap.end());

  OpBuilder opBuilder(op->getBody());

  auto structType = llzk::StructType::get(context, SymbolRefAttr::get(*op));

  auto fnOp = opBuilder.create<llzk::FuncOp>(
      UnknownLoc::get(context), StringAttr::get(context, llzk::FUNC_NAME_CONSTRAIN),
      FunctionType::get(context, {structType}, {})
  );
  fnOp.addEntryBlock();

  constrainFnMap[op->getName()] = fnOp;
  return fnOp;
}

ModuleBuilder &
ModuleBuilder::insertComputeCall(llzk::StructDefOp *caller, llzk::StructDefOp *callee) {
  auto callerFn = computeFnMap.at(caller->getName());
  auto calleeFn = computeFnMap.at(callee->getName());

  OpBuilder builder(callerFn.getBody());
  builder.create<llzk::CallOp>(
      UnknownLoc::get(context),
      calleeFn.getFullyQualifiedName(),
      mlir::ValueRange{}
  );
  updateComputeReachability(caller, callee);
  return *this;
}

ModuleBuilder &
ModuleBuilder::insertConstrainCall(llzk::StructDefOp *caller, llzk::StructDefOp *callee) {
  auto callerFn = constrainFnMap.at(caller->getName());
  auto calleeFn = constrainFnMap.at(callee->getName());
  auto calleeTy = llzk::StructType::get(context, SymbolRefAttr::get(*callee));

  size_t numOps = 0;
  for (auto it = caller->getBody().begin(); it != caller->getBody().end(); it++, numOps++)
    ;
  auto fieldName = StringAttr::get(context, callee->getName().str() + std::to_string(numOps));

  // Insert the field declaration op
  {
    OpBuilder builder(caller->getBody());
    builder.create<llzk::FieldDefOp>(UnknownLoc::get(context), fieldName, calleeTy);
  }

  // Insert the constrain function ops
  {
    OpBuilder builder(callerFn.getBody());

    auto field = builder.create<llzk::FieldReadOp>(
        UnknownLoc::get(context), calleeTy,
        callerFn.getBody().getArgument(0), // first arg is self
        fieldName
    );
    builder.create<llzk::CallOp>(
        UnknownLoc::get(context),
        calleeFn.getFullyQualifiedName(),
        mlir::ValueRange{field}
    );
  }
  updateConstrainReachability(caller, callee);
  return *this;
}

} // namespace llzk