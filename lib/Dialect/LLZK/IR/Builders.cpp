#include "llzk/Dialect/LLZK/IR/Builders.h"
#include "llzk/Dialect/LLZK/Util/SymbolHelper.h"

#include <llvm/Support/ErrorHandling.h>

using namespace mlir;

namespace llzk {

mlir::OwningOpRef<mlir::ModuleOp> createLLZKModule(mlir::MLIRContext *context, mlir::Location loc) {
  auto dialect = context->getOrLoadDialect<llzk::LLZKDialect>();
  if (!dialect) {
    llvm::report_fatal_error("Could not load LLZK dialect!");
  }
  auto langAttr = StringAttr::get(context, dialect->getNamespace());
  auto mod = ModuleOp::create(loc);
  mod->setAttr(llzk::LANG_ATTR_NAME, langAttr);
  return mod;
}

mlir::OwningOpRef<mlir::ModuleOp> createLLZKModule(mlir::MLIRContext *context) {
  return createLLZKModule(context, mlir::UnknownLoc::get(context));
}

/* ModuleBuilder */

ModuleBuilder::ModuleBuilder(mlir::ModuleOp m) : context(m.getContext()), rootModule(m) {}

void ModuleBuilder::ensureNoSuchStruct(std::string_view structName) {
  if (structMap.find(structName) != structMap.end()) {
    auto error_message = "struct " + mlir::Twine(structName) + " already exists!";
    llvm::report_fatal_error(error_message);
  }
}

void ModuleBuilder::ensureNoSuchComputeFn(std::string_view structName) {
  if (computeFnMap.find(structName) != computeFnMap.end()) {
    auto error_message = "struct " + mlir::Twine(structName) + " already has a compute function!";
    llvm::report_fatal_error(error_message);
  }
}

void ModuleBuilder::ensureComputeFnExists(std::string_view structName) {
  if (computeFnMap.find(structName) == computeFnMap.end()) {
    auto error_message = "struct " + mlir::Twine(structName) + " has no compute function!";
    llvm::report_fatal_error(error_message);
  }
}

void ModuleBuilder::ensureNoSuchConstrainFn(std::string_view structName) {
  if (constrainFnMap.find(structName) != constrainFnMap.end()) {
    auto error_message = "struct " + mlir::Twine(structName) + " already has a constrain function!";
    llvm::report_fatal_error(error_message);
  }
}

void ModuleBuilder::ensureConstrainFnExists(std::string_view structName) {
  if (constrainFnMap.find(structName) == constrainFnMap.end()) {
    auto error_message = "struct " + mlir::Twine(structName) + " has no constrain function!";
    llvm::report_fatal_error(error_message);
  }
}

ModuleBuilder &ModuleBuilder::insertEmptyStruct(std::string_view structName, mlir::Location loc) {
  ensureNoSuchStruct(structName);

  OpBuilder opBuilder(rootModule.getBody(), rootModule.getBody()->begin());
  auto structNameAtrr = StringAttr::get(context, structName);
  auto structDef = opBuilder.create<llzk::StructDefOp>(loc, structNameAtrr, nullptr);
  // populate the initial region
  auto &region = structDef.getRegion();
  (void)region.emplaceBlock();
  structMap[structName] = structDef;

  return *this;
}

ModuleBuilder &ModuleBuilder::insertComputeFn(llzk::StructDefOp op, mlir::Location loc) {
  ensureNoSuchComputeFn(op.getName());

  OpBuilder opBuilder(op.getBody());

  /// TODO: Replace with llzk::StructDefOp::getType() when available.
  auto structType = llzk::StructType::get(context, SymbolRefAttr::get(op));

  auto fnOp = opBuilder.create<llzk::FuncOp>(
      loc, StringAttr::get(context, llzk::FUNC_NAME_COMPUTE),
      FunctionType::get(context, {}, {structType})
  );
  fnOp.addEntryBlock();
  computeFnMap[op.getName()] = fnOp;
  return *this;
}

ModuleBuilder &ModuleBuilder::insertConstrainFn(llzk::StructDefOp op, mlir::Location loc) {
  ensureNoSuchConstrainFn(op.getName());

  OpBuilder opBuilder(op.getBody());

  auto structType = llzk::StructType::get(context, SymbolRefAttr::get(op));

  auto fnOp = opBuilder.create<llzk::FuncOp>(
      loc, StringAttr::get(context, llzk::FUNC_NAME_CONSTRAIN),
      FunctionType::get(context, {structType}, {})
  );
  fnOp.addEntryBlock();
  constrainFnMap[op.getName()] = fnOp;
  return *this;
}

ModuleBuilder &ModuleBuilder::insertComputeCall(
    llzk::StructDefOp caller, llzk::StructDefOp callee, mlir::Location callLoc
) {
  ensureComputeFnExists(caller.getName());
  ensureComputeFnExists(callee.getName());

  auto callerFn = computeFnMap.at(caller.getName());
  auto calleeFn = computeFnMap.at(callee.getName());

  OpBuilder builder(callerFn.getBody());
  builder.create<llzk::CallOp>(callLoc, calleeFn.getFullyQualifiedName(), mlir::ValueRange{});
  updateComputeReachability(caller, callee);
  return *this;
}

ModuleBuilder &ModuleBuilder::insertConstrainCall(
    llzk::StructDefOp caller, llzk::StructDefOp callee, mlir::Location callLoc
) {
  ensureConstrainFnExists(caller.getName());
  ensureConstrainFnExists(callee.getName());

  auto callerFn = constrainFnMap.at(caller.getName());
  auto calleeFn = constrainFnMap.at(callee.getName());
  auto calleeTy = llzk::StructType::get(context, SymbolRefAttr::get(callee));

  size_t numOps = 0;
  for (auto it = caller.getBody().begin(); it != caller.getBody().end(); it++, numOps++)
    ;
  auto fieldName = StringAttr::get(context, callee.getName().str() + std::to_string(numOps));

  // Insert the field declaration op
  {
    OpBuilder builder(caller.getBody());
    /// TODO: add a specific location for this declaration?
    builder.create<llzk::FieldDefOp>(UnknownLoc::get(context), fieldName, calleeTy);
  }

  // Insert the constrain function ops
  {
    OpBuilder builder(callerFn.getBody());

    auto field = builder.create<llzk::FieldReadOp>(
        callLoc, calleeTy,
        callerFn.getBody().getArgument(0), // first arg is self
        fieldName
    );
    builder.create<llzk::CallOp>(
        callLoc, calleeFn.getFullyQualifiedName(), mlir::ValueRange{field}
    );
  }
  updateConstrainReachability(caller, callee);
  return *this;
}

} // namespace llzk