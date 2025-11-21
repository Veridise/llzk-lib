//===- CommonAttrOrTypeCAPITestGen.h - Common test generation utilities ---===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// Common utilities shared between Attribute and Type CAPI test generators.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <mlir/TableGen/AttrOrTypeDef.h>
#include <mlir/TableGen/Dialect.h>

#include <llvm/ADT/StringExtras.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/raw_ostream.h>

#include "CommonCAPIGen.h"

/// @brief Generate dummy parameters for Get builder (used by both Attr and Type)
/// @param def The attribute or type definition
/// @param isType true if generating for a type, false for an attribute
/// @return String containing dummy parameter declarations
///
/// This function generates C code that declares dummy variables for all parameters
/// of an attribute or type Get builder. For ArrayRef parameters, it generates both
/// a count variable and an array variable. For MlirType/MlirAttribute parameters,
/// it calls helper functions to create test instances.
std::string
generateDummyParamsForAttrOrTypeGet(const mlir::tblgen::AttrOrTypeDef &def, bool isType);

/// @brief Generate parameter list for Get builder call (used by both Attr and Type)
/// @param def The attribute or type definition
/// @return String containing the parameter list
///
/// This function generates a comma-separated list of parameter names to pass to
/// a Get builder function. For ArrayRef parameters, it includes both the count
/// and array pointer. For regular parameters, it includes just the parameter name.
std::string generateParamListForAttrOrTypeGet(const mlir::tblgen::AttrOrTypeDef &def);

/// @brief Base class for attribute and type test generators
///
/// This class provides common functionality for generating unit tests
/// for attributes and types. It extends the base Generator class.
struct AttrOrTypeTestGenerator : public Generator {
  /// @brief Construct a test generator
  /// @param recordKind The kind of record ("Attribute" or "Type")
  /// @param outputStream The output stream for generated code
  /// @param testObjCreateExpression C expression to create a test object
  /// @param testObjDescriptionComment Description for the test object
  AttrOrTypeTestGenerator(
      std::string_view recordKind, llvm::raw_ostream &outputStream,
      mlir::StringRef testObjCreateExpression, mlir::StringRef testObjDescriptionComment
  )
      : Generator(recordKind, outputStream), testObjCreateExpr(testObjCreateExpression),
        testObjDescription(testObjDescriptionComment) {}

  virtual ~AttrOrTypeTestGenerator() = default;

  /// @brief Set the parameter name for code generation
  /// @param name The parameter name from the TableGen definition
  void setParamName(mlir::StringRef name) {
    this->paramName = name;
    this->paramNameCapitalized = toPascalCase(name);
  }

  /// @brief Generate test for an extra method from extraClassDeclaration
  virtual void genExtraMethod(const ExtraMethod &method) const override {
    // Convert return type to C API type, skip if it can't be converted
    std::optional<std::string> capiReturnTypeOpt = tryCppTypeToCapiType(method.returnType);
    if (!capiReturnTypeOpt.has_value()) {
      return;
    }

    // Build parameter list for dummy values
    std::string dummyParams;
    llvm::raw_string_ostream dummyParamsStream(dummyParams);
    std::string paramList;
    llvm::raw_string_ostream paramListStream(paramList);

    for (const auto &param : method.parameters) {
      // Convert C++ type to C API type for parameter, skip if it can't be converted
      std::optional<std::string> capiParamTypeOpt = tryCppTypeToCapiType(param.type);
      if (!capiParamTypeOpt.has_value()) {
        return;
      }
      std::string capiParamType = capiParamTypeOpt.value();

      // Generate dummy value creation for each parameter
      if (capiParamType == "bool") {
        dummyParamsStream << "    bool " << param.name << " = false;\n";
      } else if (capiParamType == "MlirType") {
        dummyParamsStream << "    auto " << param.name << " = mlirIndexTypeGet(context);\n";
      } else if (capiParamType == "MlirAttribute") {
        dummyParamsStream << "    auto " << param.name
                          << " = mlirIntegerAttrGet(mlirIndexTypeGet(context), 0);\n";
      } else if (capiParamType == "MlirStringRef") {
        dummyParamsStream << "    auto " << param.name
                          << " = mlirStringRefCreateFromCString(\"\");\n";
      } else if (capiParamType == "intptr_t" || capiParamType == "int" ||
                 capiParamType == "int64_t") {
        dummyParamsStream << "    " << capiParamType << " " << param.name << " = 0;\n";
      } else {
        // For unknown types, create a default-initialized variable
        dummyParamsStream << "    " << capiParamType << " " << param.name << " = {};\n";
      }

      paramListStream << ", " << param.name;
    }

    std::string capitalizedMethodName = toPascalCase(method.methodName);

    static constexpr char fmt[] = R"(
TEST_F({0}{1}LinkTests, {2}_{3}) {{
  // This test ensures {4}{0}{2}{3} links properly.
  auto test{1} = createTest{1}();
  
  if ({4}{1}IsA{0}{2}(test{1})) {{
{5}
    (void){4}{0}{2}{3}(test{1}{6});
  }
}
)";
    assert(!className.empty() && "className must be set");
    os << llvm::formatv(
        fmt, dialectNameCapitalized, kind, className, capitalizedMethodName, FunctionPrefix,
        dummyParamsStream.str(), paramListStream.str()
    );
  }

  /// @brief Generate the test class prologue
  virtual void genTestClassPrologue() const {
    static constexpr char fmt[] = R"(#include "llzk-c/Dialect/{0}.h"

#include <mlir-c/Builtin{1}s.h>
#include <mlir-c/IR.h>

class {0}{1}LinkTests : public CAPITest {{
protected:
  // Helper to create a simple test {1}
  Mlir{1} createTest{1}() {{
    return {2};
  }
};
)";
    os << llvm::formatv(fmt, dialectNameCapitalized, kind, testObjCreateExpr);
  }

  /// @brief Generate IsA test for a class
  virtual void genIsATest() const {
    static constexpr char fmt[] = R"(
TEST_F({0}{1}LinkTests, IsA_{0}{2}) {{
  // This test ensures {3}{1}IsA{0}{2} links properly.
  auto test{1} = createTest{1}();
  
  // This should always return false since test{1} is {4}
  EXPECT_FALSE({3}{1}IsA{0}{2}(test{1}));
}
)";
    assert(!className.empty() && "className must be set");
    os << llvm::formatv(
        fmt, dialectNameCapitalized, kind, className, FunctionPrefix, testObjDescription
    );
  }

  /// @brief Generate Get builder test for a definition
  /// @param dummyParams Dummy parameter declarations
  /// @param paramList Parameter list for the call
  virtual void
  genGetBuilderTest(const std::string &dummyParams, const std::string &paramList) const {
    static constexpr char fmt[] = R"(
TEST_F({0}{1}LinkTests, Get_{2}) {{
  // This test ensures {3}{0}{2}Get links properly.
  auto test{1} = createTest{1}();
  
  // We only verify the function compiles and links, wrapped in an unreachable condition
  if ({3}{1}IsA{0}{2}(test{1})) {{
{4}
    (void){3}{0}{2}Get(context{5});
  }
}
)";
    assert(!className.empty() && "className must be set");
    os << llvm::formatv(
        fmt, dialectNameCapitalized, kind, className, FunctionPrefix, dummyParams, paramList
    );
  }

  /// @brief Generate parameter getter test
  virtual void genParamGetterTest() const {

    static constexpr char fmt[] = R"(
TEST_F({0}{1}LinkTests, Get_{2}_{3}) {{
  // This test ensures {4}{0}{2}Get{5} links properly.
  auto test{1} = createTest{1}();
  
  if ({4}{1}IsA{0}{2}(test{1})) {{
    (void){4}{0}{2}Get{5}(test{1});
  }
}
)";
    assert(!className.empty() && "className must be set");
    assert(!paramName.empty() && "paramName must be set");
    os << llvm::formatv(
        fmt, dialectNameCapitalized, kind, className, paramName, FunctionPrefix,
        paramNameCapitalized
    );
  }

  /// @brief Generate ArrayRef parameter count getter test
  virtual void genArrayRefParamCountTest() const {
    static constexpr char fmt[] = R"(
TEST_F({0}{1}LinkTests, Get_{2}_{3}Count) {{
  // This test ensures {4}{0}{2}Get{5}Count links properly.
  auto test{1} = createTest{1}();
  
  if ({4}{1}IsA{0}{2}(test{1})) {{
    (void){4}{0}{2}Get{5}Count(test{1});
  }
}
)";
    assert(!className.empty() && "className must be set");
    assert(!paramName.empty() && "paramName must be set");
    os << llvm::formatv(
        fmt, dialectNameCapitalized, kind, className, paramName, FunctionPrefix,
        paramNameCapitalized
    );
  }

  /// @brief Generate ArrayRef parameter element getter test
  virtual void genArrayRefParamAtTest() const {
    static constexpr char fmt[] = R"(
TEST_F({0}{1}LinkTests, Get_{2}_{3}At) {{
  // This test ensures {4}{0}{2}Get{5}At links properly.
  auto test{1} = createTest{1}();
  
  if ({4}{1}IsA{0}{2}(test{1})) {{
    (void){4}{0}{2}Get{5}At(test{1}, 0);
  }
}
)";
    assert(!className.empty() && "className must be set");
    assert(!paramName.empty() && "paramName must be set");
    os << llvm::formatv(
        fmt, dialectNameCapitalized, kind, className, paramName, FunctionPrefix,
        paramNameCapitalized
    );
  }

  void genCompleteRecord(const mlir::tblgen::AttrOrTypeDef def, bool isType) {
    const mlir::tblgen::Dialect &defDialect = def.getDialect();

    // Generate for the selected dialect only
    if (defDialect.getName() != DialectName) {
      return;
    }

    this->setDialectAndClassName(&defDialect, def.getCppClassName());

    // Generate IsA test
    if (GenIsA) {
      this->genIsATest();
    }

    // Generate Get builder test
    if (GenTypeOrAttrGet && !def.skipDefaultBuilders()) {
      std::string dummyParams = generateDummyParamsForAttrOrTypeGet(def, isType);
      std::string paramList = generateParamListForAttrOrTypeGet(def);
      this->genGetBuilderTest(dummyParams, paramList);
    }

    // Generate parameter getter tests
    if (GenTypeOrAttrParamGetters) {
      for (const auto &param : def.getParameters()) {
        this->setParamName(param.getName());
        mlir::StringRef cppType = param.getCppType();
        if (isArrayRefType(cppType)) {
          this->genArrayRefParamCountTest();
          this->genArrayRefParamAtTest();
        } else {
          this->genParamGetterTest();
        }
      }
    }

    // Generate extra class method tests
    if (GenExtraClassMethods) {
      std::optional<mlir::StringRef> extraDecls = def.getExtraDecls();
      if (extraDecls.has_value()) {
        this->genExtraMethods(extraDecls.value());
      }
    }
  }

protected:
  mlir::StringRef testObjCreateExpr;
  mlir::StringRef testObjDescription;
  mlir::StringRef paramName;
  std::string paramNameCapitalized;
};
