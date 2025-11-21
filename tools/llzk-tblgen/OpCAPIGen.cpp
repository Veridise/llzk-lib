//===- OpCAPIGen.cpp - C API generator for operations ---------------------===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// OpCAPIGen uses the description of operations to generate C API for the ops.
//
//===----------------------------------------------------------------------===//

#include <mlir/TableGen/GenInfo.h>
#include <mlir/TableGen/Operator.h>

#include <llvm/Support/CommandLine.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/TableGen/Error.h>
#include <llvm/TableGen/Record.h>
#include <llvm/TableGen/TableGenBackend.h>

#include <string>
#include <vector>

#include "CommonCAPIGen.h"

using namespace mlir;
using namespace mlir::tblgen;

/// @brief Common between header and implementation generators for operations
struct OpGeneratorData {
  void setOperandName(mlir::StringRef name) { this->operandNameCapitalized = toPascalCase(name); }
  void setAttributeName(mlir::StringRef name) { this->attrNameCapitalized = toPascalCase(name); }
  void setResultName(mlir::StringRef name, int resultIndex) {
    this->resultNameCapitalized =
        name.empty() ? llvm::formatv("Result{0}", resultIndex).str() : toPascalCase(name.str());
  }
  void setRegionName(mlir::StringRef name, unsigned regionIndex) {
    this->regionNameCapitalized =
        name.empty() ? llvm::formatv("Region{0}", regionIndex).str() : toPascalCase(name.str());
  }

protected:
  std::string operandNameCapitalized;
  std::string attrNameCapitalized;
  std::string resultNameCapitalized;
  std::string regionNameCapitalized;
};

/// @brief Generator for operation C header files
struct OpHeaderGenerator : public HeaderGenerator, OpGeneratorData {
  using HeaderGenerator::HeaderGenerator;
  virtual ~OpHeaderGenerator() = default;

  void genOpCreateDecl(std::string const &params) const {
    static constexpr char fmt[] = R"(
/* Create a {2}::{3} Operation. */
MLIR_CAPI_EXPORTED MlirOperation {0}{1}{3}Create(MlirContext ctx, MlirLocation location{4});
)";
    assert(!className.empty() && "className must be set");
    os << llvm::formatv(
        fmt, FunctionPrefix, dialectNameCapitalized, dialect->getCppNamespace(), className, params
    );
  }

  void genOperandGetterDecl() const {
    static constexpr char fmt[] = R"(
/* Get {4} operand from {2}::{3} Operation. */
MLIR_CAPI_EXPORTED MlirValue {0}{1}{3}Get{4}(MlirOperation op);
)";
    assert(!className.empty() && "className must be set");
    assert(!operandNameCapitalized.empty() && "operandName must be set");
    os << llvm::formatv(
        fmt, FunctionPrefix, dialectNameCapitalized, dialect->getCppNamespace(), className,
        operandNameCapitalized
    );
  }

  void genOperandSetterDecl() const {
    static constexpr char fmt[] = R"(
/* Set {4} operand of {2}::{3} Operation. */
MLIR_CAPI_EXPORTED void {0}{1}{3}Set{4}(MlirOperation op, MlirValue value);
)";
    assert(!className.empty() && "className must be set");
    assert(!operandNameCapitalized.empty() && "operandName must be set");
    os << llvm::formatv(
        fmt, FunctionPrefix, dialectNameCapitalized, dialect->getCppNamespace(), className,
        operandNameCapitalized
    );
  }

  void genVariadicOperandCountGetterDecl() const {
    static constexpr char fmt[] = R"(
/* Get number of {4} operands in {2}::{3} Operation. */
MLIR_CAPI_EXPORTED intptr_t {0}{1}{3}Get{4}Count(MlirOperation op);
)";
    assert(!className.empty() && "className must be set");
    assert(!operandNameCapitalized.empty() && "operandName must be set");
    os << llvm::formatv(
        fmt, FunctionPrefix, dialectNameCapitalized, dialect->getCppNamespace(), className,
        operandNameCapitalized
    );
  }

  void genVariadicOperandIndexedGetterDecl() const {
    static constexpr char fmt[] = R"(
/* Get {4} operand at index from {2}::{3} Operation. */
MLIR_CAPI_EXPORTED MlirValue {0}{1}{3}Get{4}(MlirOperation op, intptr_t index);
)";
    assert(!className.empty() && "className must be set");
    assert(!operandNameCapitalized.empty() && "operandName must be set");
    os << llvm::formatv(
        fmt, FunctionPrefix, dialectNameCapitalized, dialect->getCppNamespace(), className,
        operandNameCapitalized
    );
  }

  void genVariadicOperandSetterDecl() const {
    static constexpr char fmt[] = R"(
/* Set {4} operands of {2}::{3} Operation. */
MLIR_CAPI_EXPORTED void {0}{1}{3}Set{4}(MlirOperation op, intptr_t count, MlirValue const *values);
)";
    assert(!className.empty() && "className must be set");
    assert(!operandNameCapitalized.empty() && "operandName must be set");
    os << llvm::formatv(
        fmt, FunctionPrefix, dialectNameCapitalized, dialect->getCppNamespace(), className,
        operandNameCapitalized
    );
  }

  void genAttributeGetterDecl() const {
    static constexpr char fmt[] = R"(
/* Get {4} attribute from {2}::{3} Operation. */
MLIR_CAPI_EXPORTED MlirAttribute {0}{1}{3}Get{4}(MlirOperation op);
)";
    assert(!className.empty() && "className must be set");
    assert(!attrNameCapitalized.empty() && "attrName must be set");
    os << llvm::formatv(
        fmt, FunctionPrefix, dialectNameCapitalized, dialect->getCppNamespace(), className,
        attrNameCapitalized
    );
  }

  void genAttributeSetterDecl() const {
    static constexpr char fmt[] = R"(
/* Set {4} attribute of {2}::{3} Operation. */
MLIR_CAPI_EXPORTED void {0}{1}{3}Set{4}(MlirOperation op, MlirAttribute attr);
)";
    assert(!className.empty() && "className must be set");
    assert(!attrNameCapitalized.empty() && "attrName must be set");
    os << llvm::formatv(
        fmt, FunctionPrefix, dialectNameCapitalized, dialect->getCppNamespace(), className,
        attrNameCapitalized
    );
  }

  void genResultGetterDecl() const {
    static constexpr char fmt[] = R"(
/* Get {4} result from {2}::{3} Operation. */
MLIR_CAPI_EXPORTED MlirValue {0}{1}{3}Get{4}(MlirOperation op);
)";
    assert(!className.empty() && "className must be set");
    assert(!resultNameCapitalized.empty() && "resultName must be set");
    os << llvm::formatv(
        fmt, FunctionPrefix, dialectNameCapitalized, dialect->getCppNamespace(), className,
        resultNameCapitalized
    );
  }

  void genVariadicResultCountGetterDecl() const {
    static constexpr char fmt[] = R"(
/* Get number of {4} results in {2}::{3} Operation. */
MLIR_CAPI_EXPORTED intptr_t {0}{1}{3}Get{4}Count(MlirOperation op);
)";
    assert(!className.empty() && "className must be set");
    assert(!resultNameCapitalized.empty() && "resultName must be set");
    os << llvm::formatv(
        fmt, FunctionPrefix, dialectNameCapitalized, dialect->getCppNamespace(), className,
        resultNameCapitalized
    );
  }

  void genVariadicResultIndexedGetterDecl() const {
    static constexpr char fmt[] = R"(
/* Get {4} result at index from {2}::{3} Operation. */
MLIR_CAPI_EXPORTED MlirValue {0}{1}{3}Get{4}(MlirOperation op, intptr_t index);
)";
    assert(!className.empty() && "className must be set");
    assert(!resultNameCapitalized.empty() && "resultName must be set");
    os << llvm::formatv(
        fmt, FunctionPrefix, dialectNameCapitalized, dialect->getCppNamespace(), className,
        resultNameCapitalized
    );
  }

  void genRegionGetterDecl() const {
    static constexpr char fmt[] = R"(
/* Get {4} region from {2}::{3} Operation. */
MLIR_CAPI_EXPORTED MlirRegion {0}{1}{3}Get{4}(MlirOperation op);
)";
    assert(!className.empty() && "className must be set");
    assert(!regionNameCapitalized.empty() && "regionName must be set");
    os << llvm::formatv(
        fmt, FunctionPrefix, dialectNameCapitalized, dialect->getCppNamespace(), className,
        regionNameCapitalized
    );
  }

  void genVariadicRegionCountGetterDecl() const {
    static constexpr char fmt[] = R"(
/* Get number of {4} regions in {2}::{3} Operation. */
MLIR_CAPI_EXPORTED intptr_t {0}{1}{3}Get{4}Count(MlirOperation op);
)";
    assert(!className.empty() && "className must be set");
    assert(!regionNameCapitalized.empty() && "regionName must be set");
    os << llvm::formatv(
        fmt, FunctionPrefix, dialectNameCapitalized, dialect->getCppNamespace(), className,
        regionNameCapitalized
    );
  }

  void genVariadicRegionIndexedGetterDecl() const {
    static constexpr char fmt[] = R"(
/* Get {4} region at index from {2}::{3} Operation. */
MLIR_CAPI_EXPORTED MlirRegion {0}{1}{3}Get{4}(MlirOperation op, intptr_t index);
)";
    assert(!className.empty() && "className must be set");
    assert(!regionNameCapitalized.empty() && "regionName must be set");
    os << llvm::formatv(
        fmt, FunctionPrefix, dialectNameCapitalized, dialect->getCppNamespace(), className,
        regionNameCapitalized
    );
  }

  void genOperationNameGetterDecl() const {
    static constexpr char fmt[] = R"(
/* Get operation name for {2}::{3} Operation. */
MLIR_CAPI_EXPORTED MlirStringRef {0}{1}{3}GetOperationName(MlirOperation op);
)";
    assert(!className.empty() && "className must be set");
    os << llvm::formatv(
        fmt, FunctionPrefix, dialectNameCapitalized, dialect->getCppNamespace(), className
    );
  }
};

/// Generate C API parameter list from operation arguments
///
/// This function builds a comma-separated parameter list for the operation create function.
/// It includes operands, attributes, result types (if not inferred), and regions.
/// Variadic parameters are represented as (count, array) pairs.
static std::string generateCAPIParams(const Operator &op) {
  std::string params;
  llvm::raw_string_ostream oss(params);

  // Add operands
  for (const auto &operand : op.getOperands()) {
    if (operand.isVariadic()) {
      oss << llvm::formatv(", intptr_t {0}Size, MlirValue const *{0}", operand.name).str();
    } else {
      oss << llvm::formatv(", MlirValue {0}", operand.name).str();
    }
  }

  // Add attributes
  for (const auto &namedAttr : op.getAttributes()) {
    std::optional<std::string> attrType = tryCppTypeToCapiType(namedAttr.attr.getStorageType());
    oss << llvm::formatv(", {0} {1}", attrType.value_or("MlirAttribute"), namedAttr.name).str();
  }

  // Add result types if not inferred
  if (!op.allResultTypesKnown()) {
    for (int i = 0, e = op.getNumResults(); i < e; ++i) {
      const auto &result = op.getResult(i);
      std::string resultName =
          result.name.empty() ? llvm::formatv("result{0}", i).str() : result.name.str();
      if (result.isVariadic()) {
        oss << llvm::formatv(", intptr_t {0}Size, MlirType const *{0}Types", resultName).str();
      } else {
        oss << llvm::formatv(", MlirType {0}Type", resultName).str();
      }
    }
  }

  // Add regions
  for (unsigned i = 0, e = op.getNumRegions(); i < e; ++i) {
    const auto &region = op.getRegion(i);
    std::string regionName =
        region.name.empty() ? llvm::formatv("region{0}", i).str() : region.name.str();
    if (region.isVariadic()) {
      oss << llvm::formatv(", intptr_t {0}Size, MlirRegion const *{0}", regionName).str();
    } else {
      oss << llvm::formatv(", MlirRegion {0}", regionName).str();
    }
  }

  return params;
}

/// Emit C API header
static bool emitOpCAPIHeader(const llvm::RecordKeeper &records, raw_ostream &os) {
  emitSourceFileHeader("Op C API Declarations", os, records);

  OpHeaderGenerator generator("Operation", os);
  generator.genPrologue();

  for (const auto *def : records.getAllDerivedDefinitions("Op")) {
    const Operator op(def);
    const Dialect &dialect = op.getDialect();

    // Generate for the selected dialect only (specified via -dialect command-line option)
    if (dialect.getName() != DialectName) {
      continue;
    }

    generator.setDialectAndClassName(&dialect, op.getCppClassName());

    // Generate create function
    if (GenOpCreate && !op.skipDefaultBuilders()) {
      generator.genOpCreateDecl(generateCAPIParams(op));
    }

    // Generate IsA check
    if (GenIsA) {
      generator.genIsADecl();
    }

    // Generate operand getters and setters
    for (int i = 0, e = op.getNumOperands(); i < e; ++i) {
      const auto &operand = op.getOperand(i);
      generator.setOperandName(operand.name);
      if (operand.isVariadic()) {
        if (GenOpOperandGetters) {
          generator.genVariadicOperandCountGetterDecl();
          generator.genVariadicOperandIndexedGetterDecl();
        }
        if (GenOpOperandSetters) {
          generator.genVariadicOperandSetterDecl();
        }
      } else {
        if (GenOpOperandGetters) {
          generator.genOperandGetterDecl();
        }
        if (GenOpOperandSetters) {
          generator.genOperandSetterDecl();
        }
      }
    }

    // Generate attribute getters and setters
    for (const auto &namedAttr : op.getAttributes()) {
      generator.setAttributeName(namedAttr.name);
      if (GenOpAttributeGetters) {
        generator.genAttributeGetterDecl();
      }
      if (GenOpAttributeSetters) {
        generator.genAttributeSetterDecl();
      }
    }

    // Generate result getters
    if (GenOpResultGetters) {
      for (int i = 0, e = op.getNumResults(); i < e; ++i) {
        const auto &result = op.getResult(i);
        generator.setResultName(result.name, i);
        if (result.isVariadic()) {
          generator.genVariadicResultCountGetterDecl();
          generator.genVariadicResultIndexedGetterDecl();
        } else {
          generator.genResultGetterDecl();
        }
      }
    }

    // Generate region getters
    if (GenOpRegionGetters) {
      for (unsigned i = 0, e = op.getNumRegions(); i < e; ++i) {
        const auto &region = op.getRegion(i);
        generator.setRegionName(region.name, i);
        if (region.isVariadic()) {
          generator.genVariadicRegionCountGetterDecl();
          generator.genVariadicRegionIndexedGetterDecl();
        } else {
          generator.genRegionGetterDecl();
        }
      }
    }

    // Generate operation name getter
    if (GenOpNameGetter) {
      generator.genOperationNameGetterDecl();
    }

    // Generate extra class method wrappers
    if (GenExtraClassMethods) {
      generator.genExtraMethods(op.getExtraClassDeclaration());
    }
  }

  generator.genEpilogue();
  return false;
}

/// @brief Generator for operation C implementation files
struct OpImplementationGenerator : public ImplementationGenerator, OpGeneratorData {
  using ImplementationGenerator::ImplementationGenerator;
  virtual ~OpImplementationGenerator() = default;

  void genOpCreateImpl(
      std::string const &params, std::string const &operationName, std::string const &assignments
  ) const {
    static constexpr char fmt[] = R"(
MlirOperation {0}{1}{2}Create(MlirContext ctx, MlirLocation location{3}) {{
  MlirOperationState state = mlirOperationStateGet(mlirStringRefCreateFromCString("{4}"), location);
{5}
  return mlirOperationCreate(&state);
}
)";
    assert(!className.empty() && "className must be set");
    os << llvm::formatv(
        fmt, FunctionPrefix, dialectNameCapitalized, className, params, operationName, assignments
    );
  }

  void genOperandGetterImpl(int index) const {
    static constexpr char fmt[] = R"(
MlirValue {0}{1}{2}Get{3}(MlirOperation op) {{
  return mlirOperationGetOperand(op, {4});
}
)";
    assert(!className.empty() && "className must be set");
    assert(!operandNameCapitalized.empty() && "operandName must be set");
    os << llvm::formatv(
        fmt, FunctionPrefix, dialectNameCapitalized, className, operandNameCapitalized, index
    );
  }

  void genOperandSetterImpl(int index) const {
    static constexpr char fmt[] = R"(
void {0}{1}{2}Set{3}(MlirOperation op, MlirValue value) {{
  mlirOperationSetOperand(op, {4}, value);
}
)";
    assert(!className.empty() && "className must be set");
    assert(!operandNameCapitalized.empty() && "operandName must be set");
    os << llvm::formatv(
        fmt, FunctionPrefix, dialectNameCapitalized, className, operandNameCapitalized, index
    );
  }

  void genVariadicOperandCountGetterImpl(int startIdx) const {
    static constexpr char fmt[] = R"(
intptr_t {0}{1}{2}Get{3}Count(MlirOperation op) {{
  intptr_t count = mlirOperationGetNumOperands(op);
  assert(count >= {4} && "operand count less than start index");
  return count - {4};
}
)";
    assert(!className.empty() && "className must be set");
    assert(!operandNameCapitalized.empty() && "operandName must be set");
    os << llvm::formatv(
        fmt, FunctionPrefix, dialectNameCapitalized, className, operandNameCapitalized, startIdx
    );
  }

  void genVariadicOperandIndexedGetterImpl(int startIdx) const {
    static constexpr char fmt[] = R"(
MlirValue {0}{1}{2}Get{3}(MlirOperation op, intptr_t index) {{
  return mlirOperationGetOperand(op, {4} + index);
}
)";
    assert(!className.empty() && "className must be set");
    assert(!operandNameCapitalized.empty() && "operandName must be set");
    os << llvm::formatv(
        fmt, FunctionPrefix, dialectNameCapitalized, className, operandNameCapitalized, startIdx
    );
  }

  void genVariadicOperandSetterImpl(int startIdx) const {
    static constexpr char fmt[] = R"(
void {0}{1}{2}Set{3}(MlirOperation op, intptr_t count, MlirValue const *values) {{
  intptr_t numOperands = mlirOperationGetNumOperands(op);
  intptr_t startIdx = {4};
  
  // Validate bounds
  if (startIdx < 0 || startIdx > numOperands) {{
    return;
  }
  if (count < 0) {{
    return;
  }
  
  intptr_t oldCount = numOperands - startIdx;
  intptr_t newNumOperands = numOperands - oldCount + count;

  std::vector<MlirValue> newOperands(newNumOperands);

  // Copy operands before this variadic group
  for (intptr_t i = 0; i < startIdx; ++i) {{
    newOperands[i] = mlirOperationGetOperand(op, i);
  }

  // Copy new variadic operands
  for (intptr_t i = 0; i < count; ++i) {{
    newOperands[startIdx + i] = values[i];
  }

  // Copy operands after this variadic group
  for (intptr_t i = startIdx + oldCount; i < numOperands; ++i) {{
    newOperands[i - oldCount + count] = mlirOperationGetOperand(op, i);
  }

  mlirOperationSetOperands(op, newNumOperands, newOperands.data());
}
)";
    assert(!className.empty() && "className must be set");
    assert(!operandNameCapitalized.empty() && "operandName must be set");
    os << llvm::formatv(
        fmt, FunctionPrefix, dialectNameCapitalized, className, operandNameCapitalized, startIdx
    );
  }

  void genAttributeGetterImpl(mlir::StringRef attrName) const {
    static constexpr char fmt[] = R"(
MlirAttribute {0}{1}{2}Get{3}(MlirOperation op) {{
  return mlirOperationGetAttributeByName(op, mlirStringRefCreateFromCString("{4}"));
}
)";
    assert(!className.empty() && "className must be set");
    assert(!attrNameCapitalized.empty() && "attrName must be set");
    os << llvm::formatv(
        fmt, FunctionPrefix, dialectNameCapitalized, className, attrNameCapitalized, attrName
    );
  }

  void genAttributeSetterImpl(mlir::StringRef attrName) const {
    static constexpr char fmt[] = R"(
void {0}{1}{2}Set{3}(MlirOperation op, MlirAttribute attr) {{
  mlirOperationSetAttributeByName(op, mlirStringRefCreateFromCString("{4}"), attr);
}
)";
    assert(!className.empty() && "className must be set");
    assert(!attrNameCapitalized.empty() && "attrName must be set");
    os << llvm::formatv(
        fmt, FunctionPrefix, dialectNameCapitalized, className, attrNameCapitalized, attrName
    );
  }

  void genResultGetterImpl(int index) const {
    static constexpr char fmt[] = R"(
MlirValue {0}{1}{2}Get{3}(MlirOperation op) {{
  return mlirOperationGetResult(op, {4});
}
)";
    assert(!className.empty() && "className must be set");
    assert(!resultNameCapitalized.empty() && "resultName must be set");
    os << llvm::formatv(
        fmt, FunctionPrefix, dialectNameCapitalized, className, resultNameCapitalized, index
    );
  }

  void genVariadicResultCountGetterImpl(int startIdx) const {
    static constexpr char fmt[] = R"(
intptr_t {0}{1}{2}Get{3}Count(MlirOperation op) {{
  intptr_t count = mlirOperationGetNumResults(op);
  assert(count >= {4} && "result count less than start index");
  return count - {4};
}
)";
    assert(!className.empty() && "className must be set");
    assert(!resultNameCapitalized.empty() && "resultName must be set");
    os << llvm::formatv(
        fmt, FunctionPrefix, dialectNameCapitalized, className, resultNameCapitalized, startIdx
    );
  }

  void genVariadicResultIndexedGetterImpl(int startIdx) const {
    static constexpr char fmt[] = R"(
MlirValue {0}{1}{2}Get{3}(MlirOperation op, intptr_t index) {{
  return mlirOperationGetResult(op, {4} + index);
}
)";
    assert(!className.empty() && "className must be set");
    assert(!resultNameCapitalized.empty() && "resultName must be set");
    os << llvm::formatv(
        fmt, FunctionPrefix, dialectNameCapitalized, className, resultNameCapitalized, startIdx
    );
  }

  void genRegionGetterImpl(unsigned index) const {
    static constexpr char fmt[] = R"(
MlirRegion {0}{1}{2}Get{3}(MlirOperation op) {{
  return mlirOperationGetRegion(op, {4});
}
)";
    assert(!className.empty() && "className must be set");
    assert(!regionNameCapitalized.empty() && "regionName must be set");
    os << llvm::formatv(
        fmt, FunctionPrefix, dialectNameCapitalized, className, regionNameCapitalized, index
    );
  }

  void genVariadicRegionCountGetterImpl(unsigned startIdx) const {
    static constexpr char fmt[] = R"(
intptr_t {0}{1}{2}Get{3}Count(MlirOperation op) {{
  intptr_t count = mlirOperationGetNumRegions(op);
  assert(count >= {4} && "region count less than start index");
  return count - {4};
}
)";
    assert(!className.empty() && "className must be set");
    assert(!regionNameCapitalized.empty() && "regionName must be set");
    os << llvm::formatv(
        fmt, FunctionPrefix, dialectNameCapitalized, className, regionNameCapitalized, startIdx
    );
  }

  void genVariadicRegionIndexedGetterImpl(unsigned startIdx) const {
    static constexpr char fmt[] = R"(
MlirRegion {0}{1}{2}Get{3}(MlirOperation op, intptr_t index) {{
  return mlirOperationGetRegion(op, {4} + index);
}
)";
    assert(!className.empty() && "className must be set");
    assert(!regionNameCapitalized.empty() && "regionName must be set");
    os << llvm::formatv(
        fmt, FunctionPrefix, dialectNameCapitalized, className, regionNameCapitalized, startIdx
    );
  }

  void GenOperationNameGetterImpl() const {
    static constexpr char fmt[] = R"(
MlirStringRef {0}{1}{2}GetOperationName(MlirOperation op) {{
  return wrap(mlir::unwrap_cast<{2}>(op).getOperationName());
}
)";
    assert(!className.empty() && "className must be set");
    os << llvm::formatv(fmt, FunctionPrefix, dialectNameCapitalized, className);
  }
};

/// Generate C API parameter assignments for operation creation
///
/// This function generates the C code that populates an MlirOperationState with
/// operands, attributes, result types, and regions. It handles both regular and
/// variadic parameters appropriately.
static std::string generateCAPIAssignments(const Operator &op) {
  std::string assignments;
  llvm::raw_string_ostream oss(assignments);

  // Add operands
  for (const auto &operand : op.getOperands()) {
    std::string name = operand.name.str();
    if (operand.isVariadic()) {
      oss << llvm::formatv("  mlirOperationStateAddOperands(&state, {0}Size, {0});\n", name).str();
    } else {
      oss << llvm::formatv("  mlirOperationStateAddOperands(&state, 1, &{0});\n", name).str();
    }
  }

  // Add attributes
  if (!op.getAttributes().empty()) {
    oss << "  MlirNamedAttribute attributes[] = {\n";
    for (const auto &namedAttr : op.getAttributes()) {
      std::string name = namedAttr.name.str();
      oss << "    mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString(\""
          << name << "\")), ";
      // The second parameter to `mlirNamedAttributeGet()` must be an "MlirAttribute". However, if
      // it ends up as "MlirIdentifier", a reinterpret cast is needed. These C structs have the same
      // layout and the C++ mlir::StringAttr is a subclass of mlir::Attribute so the cast is safe.
      std::optional<std::string> attrType = tryCppTypeToCapiType(namedAttr.attr.getStorageType());
      if (attrType.has_value() && attrType.value() == "MlirIdentifier") {
        oss << "reinterpret_cast<MlirAttribute&>(" << name << ")";
      } else {
        oss << name;
      }
      oss << " ),\n";
    }
    oss << "  };\n";
    oss << llvm::formatv(
               "  mlirOperationStateAddAttributes(&state, {0}, attributes);\n",
               op.getNumAttributes()
    )
               .str();
  }

  // Add result types if not inferred
  if (!op.allResultTypesKnown()) {
    for (int i = 0, e = op.getNumResults(); i < e; ++i) {
      const auto &result = op.getResult(i);
      std::string name =
          result.name.empty() ? llvm::formatv("result{0}", i).str() : result.name.str();
      if (result.isVariadic()) {
        oss << llvm::formatv("  mlirOperationStateAddResults(&state, {0}Size, {0}Types);\n", name)
                   .str();
      } else {
        oss << llvm::formatv("  mlirOperationStateAddResults(&state, 1, &{0}Type);\n", name).str();
      }
    }
  } else {
    oss << "  mlirOperationStateEnableResultTypeInference(&state);\n";
  }

  // Add regions
  for (unsigned i = 0, e = op.getNumRegions(); i < e; ++i) {
    const auto &region = op.getRegion(i);
    std::string name =
        region.name.empty() ? llvm::formatv("region{0}", i).str() : region.name.str();
    if (region.isVariadic()) {
      oss << llvm::formatv("  mlirOperationStateAddOwnedRegions(&state, {0}Size, {0});\n", name)
                 .str();
    } else {
      oss << llvm::formatv("  mlirOperationStateAddOwnedRegions(&state, 1, &{0});\n", name).str();
    }
  }

  return assignments;
}

/// Emit C API implementation
static bool emitOpCAPIImpl(const llvm::RecordKeeper &records, raw_ostream &os) {
  emitSourceFileHeader("Op C API Definitions", os, records);

  OpImplementationGenerator generator("Operation", os);

  // Capitalize dialect name for function names
  std::string dialectNameCapitalized = toPascalCase(DialectName);

  for (const auto *def : records.getAllDerivedDefinitions("Op")) {
    const Operator op(def);
    const Dialect &dialect = op.getDialect();
    generator.setDialectAndClassName(&dialect, op.getCppClassName());

    // Generate create function
    if (GenOpCreate && !op.skipDefaultBuilders()) {
      std::string params = generateCAPIParams(op);
      std::string assignments = generateCAPIAssignments(op);
      generator.genOpCreateImpl(params, op.getOperationName(), assignments);
    }

    // Generate IsA check implementation
    if (GenIsA) {
      generator.genIsAImpl();
    }

    // Generate operand getters and setters
    for (int i = 0, e = op.getNumOperands(); i < e; ++i) {
      const auto &operand = op.getOperand(i);
      generator.setOperandName(operand.name);
      if (operand.isVariadic()) {
        if (GenOpOperandGetters) {
          generator.genVariadicOperandCountGetterImpl(i);
          generator.genVariadicOperandIndexedGetterImpl(i);
        }
        if (GenOpOperandSetters) {
          generator.genVariadicOperandSetterImpl(i);
        }
      } else {
        if (GenOpOperandGetters) {
          generator.genOperandGetterImpl(i);
        }
        if (GenOpOperandSetters) {
          generator.genOperandSetterImpl(i);
        }
      }
    }

    // Generate attribute getters and setters
    for (const auto &namedAttr : op.getAttributes()) {
      generator.setAttributeName(namedAttr.name);
      if (GenOpAttributeGetters) {
        generator.genAttributeGetterImpl(namedAttr.name);
      }
      if (GenOpAttributeSetters) {
        generator.genAttributeSetterImpl(namedAttr.name);
      }
    }

    // Generate result getters
    if (GenOpResultGetters) {
      for (int i = 0, e = op.getNumResults(); i < e; ++i) {
        const auto &result = op.getResult(i);
        generator.setResultName(result.name, i);
        if (result.isVariadic()) {
          generator.genVariadicResultCountGetterImpl(i);
          generator.genVariadicResultIndexedGetterImpl(i);
        } else {
          generator.genResultGetterImpl(i);
        }
      }
    }

    // Generate region getters
    if (GenOpRegionGetters) {
      for (unsigned i = 0, e = op.getNumRegions(); i < e; ++i) {
        const auto &region = op.getRegion(i);
        generator.setRegionName(region.name, i);
        if (region.isVariadic()) {
          generator.genVariadicRegionCountGetterImpl(i);
          generator.genVariadicRegionIndexedGetterImpl(i);
        } else {
          generator.genRegionGetterImpl(i);
        }
      }
    }

    // Generate operation name getter implementation
    if (GenOpNameGetter) {
      generator.GenOperationNameGetterImpl();
    }

    // Generate extra class method implementations
    if (GenExtraClassMethods) {
      generator.genExtraMethods(op.getExtraClassDeclaration());
    }
  }

  return false;
}

static mlir::GenRegistration
    genOpCAPIHeader("gen-op-capi-header", "Generate operation C API header", &emitOpCAPIHeader);

static mlir::GenRegistration
    genOpCAPIImpl("gen-op-capi-impl", "Generate operation C API implementation", &emitOpCAPIImpl);
