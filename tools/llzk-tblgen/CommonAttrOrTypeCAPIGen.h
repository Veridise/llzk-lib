//===- CommonAttrOrTypeCAPIGen.h ------------------------------------------===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// Common utilities shared between Attr and Type CAPI generators
//
//===----------------------------------------------------------------------===//

#pragma once

#include <mlir/TableGen/AttrOrTypeDef.h>

#include "CommonCAPIGen.h"

/// @brief Generator for attribute/type C header files
///
/// This class extends HeaderGenerator to provide attribute and type-specific
/// header generation capabilities, including parameter getters and builders.
struct AttrOrTypeHeaderGenerator : public HeaderGenerator {
  using HeaderGenerator::HeaderGenerator;
  virtual ~AttrOrTypeHeaderGenerator() = default;

  /// @brief Set the parameter name for code generation
  /// @param name The parameter name from the TableGen definition
  void setParamName(mlir::StringRef name) {
    this->paramName = name;
    this->paramNameCapitalized = toPascalCase(name);
  }

  /// @brief Generate regular getter for non-ArrayRef type parameter
  virtual void genParameterGetterDecl(std::string capiType) const {
    static constexpr char fmt[] = R"(
/* Get '{6}' parameter from a {2}::{3} {1}. */
MLIR_CAPI_EXPORTED {7} {0}{4}{3}Get{5}(Mlir{1});
)";
    assert(dialect && "Dialect must be set");
    assert(!paramName.empty() && "paramName must be set");
    os << llvm::formatv(
        fmt, FunctionPrefix, kind, dialect->getCppNamespace(), className, dialectNameCapitalized,
        paramNameCapitalized, paramName, capiType
    );
  }

  /// @brief Generate count function for ArrayRef parameter
  virtual void genArrayRefParameterCountDecl() const {
    static constexpr char fmt[] = R"(
/* Get count of '{6}' parameter from a {2}::{3} {1}. */
MLIR_CAPI_EXPORTED intptr_t {0}{4}{3}Get{5}Count(Mlir{1});
)";
    assert(dialect && "Dialect must be set");
    assert(!paramName.empty() && "paramName must be set");
    os << llvm::formatv(
        fmt, FunctionPrefix, kind, dialect->getCppNamespace(), className, dialectNameCapitalized,
        paramNameCapitalized, paramName
    );
  }

  /// @brief Generate accessor function for ArrayRef parameter elements
  virtual void genArrayRefParameterAtDecl(std::string elemType) const {
    static constexpr char fmt[] = R"(
/* Get element at index from '{6}' parameter from a {2}::{3} {1}. */
MLIR_CAPI_EXPORTED {7} {0}{4}{3}Get{5}At(Mlir{1}, intptr_t pos);
)";
    assert(dialect && "Dialect must be set");
    assert(!paramName.empty() && "paramName must be set");
    os << llvm::formatv(
        fmt, FunctionPrefix, kind, dialect->getCppNamespace(), className, dialectNameCapitalized,
        paramNameCapitalized, paramName, elemType
    );
  }

  /// @brief Generate default Get builder declaration
  virtual void genDefaultGetBuilderDecl(const mlir::tblgen::AttrOrTypeDef &def) const {
    static constexpr char fmt[] = R"(
/* Create a {2}::{3} {1} with the given parameters. */
MLIR_CAPI_EXPORTED Mlir{1} {0}{4}{3}Get(MlirContext ctx{5});
)";
    assert(dialect && "Dialect must be set");

    // Use raw_string_ostream for efficient string building
    std::string paramListBuffer;
    llvm::raw_string_ostream paramListStream(paramListBuffer);

    for (const auto &param : def.getParameters()) {
      mlir::StringRef cppType = param.getCppType();
      if (isArrayRefType(cppType)) {
        // For ArrayRef parameters, use intptr_t for count and pointer to element type
        paramListStream << ", intptr_t " << param.getName() << "Count, "
                        << extractArrayRefElementType(cppType) << " *" << param.getName();
      } else {
        paramListStream << ", " << mapCppTypeToCapiType(cppType) << " " << param.getName();
      }
    }

    os << llvm::formatv(
        fmt, FunctionPrefix, kind, dialect->getCppNamespace(), className, dialectNameCapitalized,
        paramListStream.str()
    );
  }

  void genCompleteRecord(const mlir::tblgen::AttrOrTypeDef def) {
    const mlir::tblgen::Dialect &dialect = def.getDialect();

    // Generate for the selected dialect only
    if (dialect.getName() != DialectName) {
      return;
    }

    this->setDialectAndClassName(&dialect, def.getCppClassName());

    // Generate IsA check
    if (GenIsA) {
      this->genIsADecl();
    }

    // Generate default Get builder if not skipped
    if (!def.skipDefaultBuilders()) {
      this->genDefaultGetBuilderDecl(def);
    }

    // Generate parameter getters
    if (GenTypeOrAttrParamGetters) {
      for (const auto &param : def.getParameters()) {
        this->setParamName(param.getName());
        mlir::StringRef cppType = param.getCppType();

        if (isArrayRefType(cppType)) {
          this->genArrayRefParameterCountDecl();
          this->genArrayRefParameterAtDecl(extractArrayRefElementType(cppType));
        } else {
          this->genParameterGetterDecl(mapCppTypeToCapiType(cppType));
        }
      }
    }

    // Generate extra class method declarations
    if (GenExtraClassMethods) {
      std::optional<mlir::StringRef> extraDecls = def.getExtraDecls();
      if (extraDecls.has_value()) {
        this->genExtraMethods(extraDecls.value());
      }
    }
  }

protected:
  mlir::StringRef paramName;
  std::string paramNameCapitalized;
};

/// @brief Generator for attribute/type C implementation files
///
/// This class extends ImplementationGenerator to provide attribute and type-specific
/// implementation generation capabilities, including parameter getters and builders.
struct AttrOrTypeImplementationGenerator : public ImplementationGenerator {
  using ImplementationGenerator::ImplementationGenerator;
  virtual ~AttrOrTypeImplementationGenerator() = default;

  /// @brief Set the parameter name for code generation
  /// @param name The parameter name from the TableGen definition
  void setParamName(mlir::StringRef name) {
    this->paramName = name;
    this->paramNameCapitalized = toPascalCase(name);
  }

  virtual void genPrologue() const {
    os << R"(
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llzk/Dialect/LLZK/IR/AttributeHelper.h"

using namespace mlir;
using namespace llvm;

)";
  }

  virtual void genArrayRefParameterCountImpl() const {
    static constexpr char fmt[] = R"(
intptr_t {0}{2}{3}Get{4}Count(Mlir{1} inp) {{
  return static_cast<intptr_t>(llvm::cast<{3}>(unwrap(inp)).get{4}().size());
}
 )";
    assert(!className.empty() && "className must be set");
    assert(!paramName.empty() && "paramName must be set");
    os << llvm::formatv(
        fmt, FunctionPrefix, kind, dialectNameCapitalized, className, paramNameCapitalized
    );
  }

  virtual void genArrayRefParameterAtImplAPInt() const {
    static constexpr char fmt[] = R"(
int64_t {0}{2}{3}Get{4}At(Mlir{1} inp, intptr_t pos) {{
  return ::llzk::fromAPInt(llvm::cast<{3}>(unwrap(inp)).get{4}()[pos]);
}
 )";
    assert(!className.empty() && "className must be set");
    assert(!paramName.empty() && "paramName must be set");
    os << llvm::formatv(
        fmt, FunctionPrefix, kind, dialectNameCapitalized, className, paramNameCapitalized
    );
  }

  virtual void genArrayRefParameterAtImplRaw(std::string elemType) const {
    static constexpr char fmt[] = R"(
{5} {0}{2}{3}Get{4}At(Mlir{1} inp, intptr_t pos) {{
  return llvm::cast<{3}>(unwrap(inp)).get{4}()[pos];
}
 )";
    assert(!className.empty() && "className must be set");
    assert(!paramName.empty() && "paramName must be set");
    os << llvm::formatv(
        fmt, FunctionPrefix, kind, dialectNameCapitalized, className, paramNameCapitalized, elemType
    );
  }

  virtual void genArrayRefParameterAtImplWrapped(std::string elemType) const {
    static constexpr char fmt[] = R"(
{5} {0}{2}{3}Get{4}At(Mlir{1} inp, intptr_t pos) {{
  return wrap(llvm::cast<{3}>(unwrap(inp)).get{4}()[pos]);
}
 )";
    assert(!className.empty() && "className must be set");
    assert(!paramName.empty() && "paramName must be set");
    os << llvm::formatv(
        fmt, FunctionPrefix, kind, dialectNameCapitalized, className, paramNameCapitalized, elemType
    );
  }

  virtual void genParameterGetterImplAPInt() const {
    static constexpr char fmt[] = R"(
int64_t {0}{2}{3}Get{4}(Mlir{1} inp) {{
  return ::llzk::fromAPInt(llvm::cast<{3}>(unwrap(inp)).get{4}());
}
 )";
    assert(!className.empty() && "className must be set");
    assert(!paramName.empty() && "paramName must be set");
    os << llvm::formatv(
        fmt, FunctionPrefix, kind, dialectNameCapitalized, className, paramNameCapitalized
    );
  }

  virtual void genParameterGetterImplRaw(std::string capiType) const {
    static constexpr char fmt[] = R"(
{5} {0}{2}{3}Get{4}(Mlir{1} inp) {{
  return llvm::cast<{3}>(unwrap(inp)).get{4}();
}
 )";
    assert(!className.empty() && "className must be set");
    assert(!paramName.empty() && "paramName must be set");
    os << llvm::formatv(
        fmt, FunctionPrefix, kind, dialectNameCapitalized, className, paramNameCapitalized, capiType
    );
  }

  virtual void genParameterGetterImplWrapped(std::string capiType) const {
    static constexpr char fmt[] = R"(
{5} {0}{2}{3}Get{4}(Mlir{1} inp) {{
  return wrap(llvm::cast<{3}>(unwrap(inp)).get{4}());
}
 )";
    assert(!className.empty() && "className must be set");
    assert(!paramName.empty() && "paramName must be set");
    os << llvm::formatv(
        fmt, FunctionPrefix, kind, dialectNameCapitalized, className, paramNameCapitalized, capiType
    );
  }

  /// @brief Generate default Get builder implementation
  virtual void genDefaultGetBuilderImpl(const mlir::tblgen::AttrOrTypeDef &def) const {
    static constexpr char fmt[] = R"(
Mlir{1} {0}{2}{3}Get(MlirContext ctx{4}) {{
  return wrap({3}::get(unwrap(ctx){5}));
}
 )";
    assert(!className.empty() && "className must be set");

    // Use raw_string_ostream for efficient string building
    std::string paramListBuffer;
    std::string argListBuffer;
    llvm::raw_string_ostream paramListStream(paramListBuffer);
    llvm::raw_string_ostream argListStream(argListBuffer);

    for (const auto &param : def.getParameters()) {
      mlir::StringRef cppType = param.getCppType();
      std::string pName = param.getName().str();

      if (isArrayRefType(cppType)) {
        // For ArrayRef parameters, convert from pointer + count to ArrayRef
        std::string elemType = extractArrayRefElementType(cppType);
        paramListStream << ", intptr_t " << pName << "Count, " << elemType << " *" << pName;

        // In the call, we need to convert back to ArrayRef
        // Check if elements need unwrapping
        if (isPrimitiveType(elemType)) {
          argListStream << ", ::llvm::ArrayRef<" << elemType << ">(" << pName << ", " << pName
                        << "Count)";
        } else {
          argListStream << ", ::llvm::ArrayRef<" << elemType << ">(unwrapList(" << pName << ", "
                        << pName << "Count))";
        }
      } else {
        std::string capiType = mapCppTypeToCapiType(cppType);
        paramListStream << ", " << capiType << " " << pName;

        // Add unwrapping if needed
        argListStream << ", ";
        if (isPrimitiveType(cppType)) {
          argListStream << pName;
        } else if (capiType == "MlirAttribute" || capiType == "MlirType") {
          // Needs additional cast to the specific attribute/type class
          argListStream << "::llvm::cast<" << cppType << ">(unwrap(" << pName << "))";
        } else {
          // Any other cases, just use an "unwrap" function
          argListStream << "unwrap(" << pName << ")";
        }
      }
    }

    os << llvm::formatv(
        fmt, FunctionPrefix, kind, dialectNameCapitalized, className, paramListStream.str(),
        argListStream.str()
    );
  }

  void genCompleteRecord(const mlir::tblgen::AttrOrTypeDef def) {
    const mlir::tblgen::Dialect &dialect = def.getDialect();

    // Generate for the selected dialect only
    if (dialect.getName() != DialectName) {
      return;
    }

    this->setDialectAndClassName(&dialect, def.getCppClassName());

    // Generate IsA check implementation
    if (GenIsA) {
      this->genIsAImpl();
    }

    // Generate default Get builder implementation if not skipped
    if (GenTypeOrAttrGet && !def.skipDefaultBuilders()) {
      this->genDefaultGetBuilderImpl(def);
    }

    // Generate parameter getter implementations
    if (GenTypeOrAttrParamGetters) {
      for (const auto &param : def.getParameters()) {
        this->setParamName(param.getName());
        mlir::StringRef cppType = param.getCppType();

        if (isArrayRefType(cppType)) {
          // Generate getter functions for ArrayRef parameters
          this->genArrayRefParameterCountImpl();
          std::string elemType = extractArrayRefElementType(cppType);
          if (isAPIntType(elemType)) {
            this->genArrayRefParameterAtImplAPInt();
          } else if (isPrimitiveType(elemType)) {
            this->genArrayRefParameterAtImplRaw(elemType);
          } else {
            this->genArrayRefParameterAtImplWrapped(elemType);
          }
        } else {
          // Generate regular getter implementation
          if (isAPIntType(cppType)) {
            this->genParameterGetterImplAPInt();
          } else {
            std::string capiType = mapCppTypeToCapiType(cppType);
            if (isPrimitiveType(capiType)) {
              this->genParameterGetterImplRaw(capiType);
            } else {
              this->genParameterGetterImplWrapped(capiType);
            }
          }
        }
      }
    }

    // Generate extra class method implementations
    if (GenExtraClassMethods) {
      std::optional<mlir::StringRef> extraDecls = def.getExtraDecls();
      if (extraDecls.has_value()) {
        this->genExtraMethods(extraDecls.value());
      }
    }
  }

protected:
  mlir::StringRef paramName;
  std::string paramNameCapitalized;
};
