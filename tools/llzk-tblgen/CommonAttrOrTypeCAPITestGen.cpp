//===- CommonAttrOrTypeCAPITestGen.cpp - Common test generation utilities -===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// Implementation of CAPI test generation utilities for Attribute and Type.
//
//===----------------------------------------------------------------------===//

#include "CommonAttrOrTypeCAPITestGen.h"

#include <llvm/Support/FormatVariadic.h>

using namespace mlir;
using namespace mlir::tblgen;

/// Generate dummy parameters for Get builder
std::string generateDummyParamsForAttrOrTypeGet(const AttrOrTypeDef &def, bool isType) {
  // Use raw_string_ostream for efficient string building
  std::string paramsBuffer;
  llvm::raw_string_ostream paramsStream(paramsBuffer);

  for (const auto &param : def.getParameters()) {
    StringRef cppType = param.getCppType();
    std::string pName = param.getName().str();

    if (isArrayRefType(cppType)) {
      paramsStream << llvm::formatv("    intptr_t {0}Count = 0;\n", pName);
      mlir::StringRef cppElemType = extractArrayRefElementType(cppType);
      std::string elemType = mapCppTypeToCapiType(cppElemType);
      if (isPrimitiveType(cppElemType)) {
        paramsStream << llvm::formatv("    {0} {1}Array = 0;\n", elemType, pName);
        paramsStream << llvm::formatv("    {0} *{1} = &{1}Array;\n", elemType, pName);
      } else if (isType && elemType == "MlirType") {
        paramsStream << llvm::formatv("    auto {0}Elem = createTestType();\n", pName);
        paramsStream << llvm::formatv("    {0} *{1} = &{0}Elem;\n", elemType, pName);
      } else if (!isType && elemType == "MlirAttribute") {
        paramsStream << llvm::formatv("    auto {0}Elem = createTestAttr();\n", pName);
        paramsStream << llvm::formatv("    {0} *{1} = &{0}Elem;\n", elemType, pName);
      } else {
        paramsStream << llvm::formatv("    {0} {1}Elem = {{}};\n", elemType, pName);
        paramsStream << llvm::formatv("    {0} *{1} = &{1}Elem;\n", elemType, pName);
      }
    } else {
      std::string capiType = mapCppTypeToCapiType(cppType);
      if (isPrimitiveType(cppType)) {
        paramsStream << llvm::formatv("    {0} {1} = 0;\n", capiType, pName);
      } else if (isType && capiType == "MlirType") {
        paramsStream << llvm::formatv("    auto {0} = createTestType();\n", pName);
      } else if (!isType && capiType == "MlirAttribute") {
        paramsStream << llvm::formatv("    auto {0} = createTestAttr();\n", pName);
      } else {
        // For enum or other types, use static_cast to initialize with 0
        paramsStream << llvm::formatv("    {0} {1} = static_cast<{0}>(0);\n", capiType, pName);
      }
    }
  }

  return paramsBuffer;
}

/// Generate parameter list for Get builder call
std::string generateParamListForAttrOrTypeGet(const AttrOrTypeDef &def) {
  // Use raw_string_ostream for efficient string building
  std::string paramsBuffer;
  llvm::raw_string_ostream paramsStream(paramsBuffer);

  for (const auto &param : def.getParameters()) {
    std::string pName = param.getName().str();
    if (isArrayRefType(param.getCppType())) {
      paramsStream << llvm::formatv(", {0}Count, {0}", pName);
    } else {
      paramsStream << llvm::formatv(", {0}", pName);
    }
  }

  return paramsBuffer;
}
