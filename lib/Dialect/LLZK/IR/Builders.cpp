#include "llzk/Dialect/LLZK/IR/Builders.h"
#include "llzk/Dialect/LLZK/Util/SymbolHelper.h"

#include <llvm/Support/ErrorHandling.h>

namespace llzk {

using namespace mlir;

OwningOpRef<ModuleOp> createLLZKModule(MLIRContext *context, Location loc) {
  auto dialect = context->getOrLoadDialect<LLZKDialect>();
  if (!dialect) {
    llvm::report_fatal_error("Could not load LLZK dialect!");
  }
  auto langAttr = StringAttr::get(context, dialect->getNamespace());
  auto mod = ModuleOp::create(loc);
  mod->setAttr(LANG_ATTR_NAME, langAttr);
  return mod;
}

OwningOpRef<ModuleOp> createLLZKModule(MLIRContext *context) {
  return createLLZKModule(context, UnknownLoc::get(context));
}

/* ModuleBuilder */

ModuleBuilder::ModuleBuilder(ModuleOp m) : context(m.getContext()), rootModule(m) {}

void ModuleBuilder::ensureNoSuchStruct(std::string_view structName) {
  if (structMap.find(structName) != structMap.end()) {
    auto error_message = "struct " + Twine(structName) + " already exists!";
    llvm::report_fatal_error(error_message);
  }
}

void ModuleBuilder::ensureNoSuchComputeFn(std::string_view structName) {
  if (computeFnMap.find(structName) != computeFnMap.end()) {
    auto error_message = "struct " + Twine(structName) + " already has a compute function!";
    llvm::report_fatal_error(error_message);
  }
}

void ModuleBuilder::ensureComputeFnExists(std::string_view structName) {
  if (computeFnMap.find(structName) == computeFnMap.end()) {
    auto error_message = "struct " + Twine(structName) + " has no compute function!";
    llvm::report_fatal_error(error_message);
  }
}

void ModuleBuilder::ensureNoSuchConstrainFn(std::string_view structName) {
  if (constrainFnMap.find(structName) != constrainFnMap.end()) {
    auto error_message = "struct " + Twine(structName) + " already has a constrain function!";
    llvm::report_fatal_error(error_message);
  }
}

void ModuleBuilder::ensureConstrainFnExists(std::string_view structName) {
  if (constrainFnMap.find(structName) == constrainFnMap.end()) {
    auto error_message = "struct " + Twine(structName) + " has no constrain function!";
    llvm::report_fatal_error(error_message);
  }
}

ModuleBuilder &ModuleBuilder::insertEmptyStruct(std::string_view structName, Location loc) {
  ensureNoSuchStruct(structName);

  OpBuilder opBuilder(rootModule.getBody(), rootModule.getBody()->begin());
  auto structNameAtrr = StringAttr::get(context, structName);
  auto structDef = opBuilder.create<StructDefOp>(loc, structNameAtrr, nullptr);
  // populate the initial region
  auto &region = structDef.getRegion();
  (void)region.emplaceBlock();
  structMap[structName] = structDef;

  return *this;
}

ModuleBuilder &ModuleBuilder::insertComputeFn(StructDefOp op, Location loc) {
  ensureNoSuchComputeFn(op.getName());

  OpBuilder opBuilder(op.getBody());

  auto fnOp = opBuilder.create<FuncOp>(
      loc, StringAttr::get(context, FUNC_NAME_COMPUTE),
      FunctionType::get(context, {}, {op.getType()})
  );
  fnOp.addEntryBlock();
  computeFnMap[op.getName()] = fnOp;
  return *this;
}

ModuleBuilder &ModuleBuilder::insertConstrainFn(StructDefOp op, Location loc) {
  ensureNoSuchConstrainFn(op.getName());

  OpBuilder opBuilder(op.getBody());

  auto fnOp = opBuilder.create<FuncOp>(
      loc, StringAttr::get(context, FUNC_NAME_CONSTRAIN),
      FunctionType::get(context, {op.getType()}, {})
  );
  fnOp.addEntryBlock();
  constrainFnMap[op.getName()] = fnOp;
  return *this;
}

ModuleBuilder &
ModuleBuilder::insertComputeCall(StructDefOp caller, StructDefOp callee, Location callLoc) {
  ensureComputeFnExists(caller.getName());
  ensureComputeFnExists(callee.getName());

  auto callerFn = computeFnMap.at(caller.getName());
  auto calleeFn = computeFnMap.at(callee.getName());

  OpBuilder builder(callerFn.getBody());
  builder.create<CallOp>(callLoc, calleeFn.getResultTypes(), calleeFn.getFullyQualifiedName());
  updateComputeReachability(caller, callee);
  return *this;
}

ModuleBuilder &
ModuleBuilder::insertConstrainCall(StructDefOp caller, StructDefOp callee, Location callLoc) {
  ensureConstrainFnExists(caller.getName());
  ensureConstrainFnExists(callee.getName());

  auto callerFn = constrainFnMap.at(caller.getName());
  auto calleeFn = constrainFnMap.at(callee.getName());
  auto calleeTy = callee.getType();

  size_t numOps = 0;
  for (auto it = caller.getBody().begin(); it != caller.getBody().end(); it++, numOps++)
    ;
  auto fieldName = StringAttr::get(context, callee.getName().str() + std::to_string(numOps));

  // Insert the field declaration op
  {
    OpBuilder builder(caller.getBody());
    /// TODO: add a specific location for this declaration?
    builder.create<FieldDefOp>(UnknownLoc::get(context), fieldName, calleeTy);
  }

  // Insert the constrain function ops
  {
    OpBuilder builder(callerFn.getBody());

    auto field = builder.create<FieldReadOp>(
        callLoc, calleeTy,
        callerFn.getBody().getArgument(0), // first arg is self
        fieldName
    );
    builder.create<CallOp>(
        callLoc, TypeRange {}, calleeFn.getFullyQualifiedName(), ValueRange {field}
    );
  }
  updateConstrainReachability(caller, callee);
  return *this;
}

} // namespace llzk
