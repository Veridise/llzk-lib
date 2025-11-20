//===- CommonCAPIGen.h - Common utilities for C API generation -----------===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// Common utilities shared between all CAPI generators (ops, attrs, types)
//
//===----------------------------------------------------------------------===//

#pragma once

#include <mlir/TableGen/Dialect.h>

#include <llvm/ADT/StringExtras.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/StringSwitch.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/FormatVariadic.h>

#include <memory>
#include <string>

// Forward declarations for Clang classes
namespace clang {
class Lexer;
class SourceManager;
class FileManager;
class DiagnosticsEngine;
} // namespace clang

// Shared command-line options used by all CAPI generators
extern llvm::cl::OptionCategory OpGenCat;
extern llvm::cl::opt<std::string> DialectName;
extern llvm::cl::opt<std::string> FunctionPrefix;

// Shared flags for controlling code generation
extern llvm::cl::opt<bool> GenIsAChecks;
extern llvm::cl::opt<bool> GenIsATests;
extern llvm::cl::opt<bool> GenTypeOrAttrParamGetters;
extern llvm::cl::opt<bool> GenExtraClassMethods;

/// @brief Convert names separated by underscore or colon to PascalCase.
/// @param str The input string to convert (may contain underscores or colons)
/// @return The converted PascalCase string
///
/// Examples:
///   "no_inline" -> "NoInline"
///   "::llzk::boolean::Type" -> "LlzkBooleanType"
inline std::string toPascalCase(mlir::StringRef str) {
  if (str.empty()) {
    return "";
  }

  std::string result;
  result.reserve(str.size());
  bool capitalizeNext = true;

  for (char c : str) {
    if (c == '_' || c == ':') {
      capitalizeNext = true;
    } else {
      result += capitalizeNext ? llvm::toUpper(c) : c;
      capitalizeNext = false;
    }
  }

  return result;
}

/// @brief Check if a C++ type is a known primitive type
/// @param type The type string to check
/// @return true if the type is a primitive (bool, void, int, etc.)
inline bool isPrimitiveType(mlir::StringRef type) {
  type.consume_front("::");
  return llvm::StringSwitch<bool>(type)
      .Case("bool", true)
      .Case("void", true)
      .Case("int", true)
      .Case("unsigned", true)
      .Case("size_t", true)
      .Case("intptr_t", true)
      .Case("int32_t", true)
      .Case("int64_t", true)
      .Case("uint32_t", true)
      .Case("uint64_t", true)
      .Default(false);
}

/// @brief Check if a token text represents a C++ modifier/specifier keyword
/// @param tokenText The token to check
/// @return true if the token is a C++ modifier (inline, static, virtual, etc.)
inline bool isCppModifierKeyword(mlir::StringRef tokenText) {
  return llvm::StringSwitch<bool>(tokenText)
      .Case("inline", true)
      .Case("static", true)
      .Case("virtual", true)
      .Case("explicit", true)
      .Case("constexpr", true)
      .Case("consteval", true)
      .Case("extern", true)
      .Case("mutable", true)
      .Case("friend", true)
      .Default(false);
}

/// @brief Check if a method name represents a C++ control flow keyword or language construct
/// @param methodName The method name to check
/// @return true if the name is a C++ language construct (if, for, while, etc.)
inline bool isCppLanguageConstruct(mlir::StringRef methodName) {
  return llvm::StringSwitch<bool>(methodName)
      .Case("if", true)
      .Case("for", true)
      .Case("while", true)
      .Case("switch", true)
      .Case("return", true)
      .Case("sizeof", true)
      .Case("decltype", true)
      .Case("alignof", true)
      .Case("typeid", true)
      .Case("static_assert", true)
      .Case("noexcept", true)
      .Default(false);
}

/// @brief Check if a C++ type is a known integer type
/// @param type The type string to check
/// @return true if the type is an integer type (size_t, unsigned, int*, uint*, etc.)
inline bool isIntegerType(mlir::StringRef type) {
  type.consume_front("::");
  return type == "size_t" || type == "unsigned" || type.starts_with("int") ||
         type.starts_with("uint");
}

/// @brief Check if a C++ type is an ArrayRef type
/// @param cppType The C++ type string to check
/// @return true if the type is ArrayRef, llvm::ArrayRef, or ::llvm::ArrayRef
inline bool isArrayRefType(mlir::StringRef cppType) {
  cppType.consume_front("::");
  cppType.consume_front("llvm::");
  return cppType.starts_with("ArrayRef<");
}

/// @brief Check if a C++ type is APInt
/// @param cppType The C++ type string to check
/// @return true if the type is APInt, llvm::APInt, or ::llvm::APInt
inline bool isAPIntType(mlir::StringRef cppType) {
  cppType.consume_front("::");
  cppType.consume_front("llvm::");
  return cppType == "APInt";
}

/// @brief RAII wrapper for Clang lexer infrastructure
///
/// This class simplifies setting up Clang's lexer for parsing C++ code snippets.
/// It manages the lifetime of all required Clang objects (FileManager, SourceManager,
/// DiagnosticsEngine, etc.) and provides easy access to the lexer.
///
/// The Lexer is used instead of the Parser so that comments preceding method declarations
/// can be captured for documentation generation.
class ClangLexerContext {
public:
  /// @brief Construct a lexer context for the given source code
  /// @param source The C++ source code to lex
  /// @param bufferName Optional name for the memory buffer (for diagnostics)
  explicit ClangLexerContext(mlir::StringRef source, mlir::StringRef bufferName = "input");

  /// @brief Get the lexer instance
  /// @return Reference to the Clang lexer
  clang::Lexer &getLexer() const;

  /// @brief Get the source manager instance
  /// @return Reference to the Clang source manager
  clang::SourceManager &getSourceManager() const;

  /// @brief Check if the lexer was successfully created
  /// @return true if the lexer is valid and ready to use
  bool isValid() const { return lexer != nullptr; }

private:
  struct Impl;
  std::unique_ptr<Impl> impl;
  clang::Lexer *lexer = nullptr;
};

struct MethodParameter {
  /// The C++ type of the parameter
  std::string type;
  /// The name of the parameter
  std::string name;
};

/// @brief Structure to represent a parsed method signature from extraClassDeclaration
///
/// This structure holds information extracted from parsing C++ method declarations.
/// It is used to generate corresponding C API wrapper functions.
struct ExtraMethod {
  /// The C++ return type of the method
  std::string returnType;
  /// The name of the method
  std::string methodName;
  /// Documentation comment (if any)
  std::string documentation;
  /// Whether the method is const-qualified
  bool isConst = false;
  /// Whether the method has parameters (unsupported for now)
  bool hasParameters = false;
  /// The parameters of the method
  std::vector<MethodParameter> parameters;
};

/// @brief Parse method declarations from extraClassDeclaration using Clang's Lexer
/// @param extraDecl The C++ code from extraClassDeclaration
/// @return Vector of parsed method signatures
///
/// This function parses C++ method declarations to extract signatures that can be
/// wrapped in C API functions. It identifies methods by looking for the pattern:
/// [modifiers] <return_type> <identifier> '(' [params] ')' [const] ';'
///
/// Example input:
/// @code
///   /// Get the width of this type
///   unsigned getWidth() const;
///   bool isSignless() const;
/// @endcode
///
/// Example output:
/// - ExtraMethod { returnType="unsigned", methodName="getWidth", isConst=true, hasParameters=false
/// }
/// - ExtraMethod { returnType="bool", methodName="isSignless", isConst=true, hasParameters=false }
///
/// Note: Methods with parameters are detected but currently skipped during code generation.
llvm::SmallVector<ExtraMethod> parseExtraMethods(mlir::StringRef extraDecl);

/// @brief Check if a C++ type matches an MLIR type pattern
/// @param cppType The C++ type to check
/// @param typeName The MLIR type name to match against
/// @return true if the C++ type matches the MLIR type
bool matchesMLIRType(mlir::StringRef cppType, mlir::StringRef typeName);

/// @brief Convert C++ return type to MLIR C API type
/// @param cppType The C++ type to convert
/// @return The corresponding MLIR C API type
std::string cppTypeToCapiType(mlir::StringRef cppType);

/// @brief Determine the wrapping code needed for a return value
/// @param capiType The MLIR C API type
/// @return The wrapper function name (e.g., "wrap") or empty string if no wrapping needed
mlir::StringRef getReturnWrapCode(mlir::StringRef capiType);

/// @brief Check if a return type conversion is valid for C API generation
/// @param capiReturnType The target MLIR C API return type
/// @param cppReturnType The source C++ return type
/// @return true if the conversion is supported
bool isValidTypeConversion(const std::string &capiReturnType, const std::string &cppReturnType);

/// @brief Map C++ type to corresponding MLIR C API return type
/// @param cppType The C++ type to map
/// @return The corresponding MLIR C API type string
///
/// @note This function should not be called for ArrayRef types.
/// Use extractArrayRefElementType() for those instead.
inline std::string mapCppTypeToCapiType(mlir::StringRef cppType) {
  assert(!isArrayRefType(cppType) && "use extractArrayRefElementType instead");

  // Primitive types
  if (isPrimitiveType(cppType)) {
    return cppType.str();
  }

  // Direct type mappings
  if (matchesMLIRType(cppType, "Type")) {
    return "MlirType";
  }
  if (matchesMLIRType(cppType, "Attribute")) {
    return "MlirAttribute";
  }

  // APInt types - convert to int64_t using fromAPInt helper
  if (isAPIntType(cppType)) {
    return "int64_t";
  }

  // Specific MLIR attribute types
  if ((cppType.starts_with("mlir::") || cppType.starts_with("::mlir::")) &&
      cppType.ends_with("Attr")) {
    return "MlirAttribute";
  }

  // Otherwise assume it's a type where the C name is a direct translation from the C++ name.
  return toPascalCase(cppType);
}

/// Extract element type from ArrayRef<...>
inline std::string extractArrayRefElementType(mlir::StringRef cppType) {
  // Remove "::llvm::ArrayRef<" or "ArrayRef<" prefix and ">" suffix
  cppType.consume_front("::");
  cppType.consume_front("llvm::");
  if (cppType.consume_front("ArrayRef<") && cppType.consume_back(">")) {
    return mapCppTypeToCapiType(cppType);
  }
  return "MlirAttribute"; // fallback
}

/// @brief Base class for C API generators
struct Generator {
  Generator(std::string_view recordKind, llvm::raw_ostream &outputStream)
      : kind(recordKind), os(outputStream), dialectNameCapitalized(toPascalCase(DialectName)) {}
  virtual ~Generator() = default;

  /// @brief Set the dialect and class name for code generation
  /// @param d Pointer to the dialect definition
  /// @param cppClassName The C++ class name of the entity being generated
  virtual void
  setDialectAndClassName(const mlir::tblgen::Dialect *d, mlir::StringRef cppClassName) {
    this->dialect = d;
    this->className = cppClassName;
  }

  /// @brief Generate code for extra methods from extraClassDeclaration
  /// @param extraDecl The extra class declaration string
  virtual void genExtraMethods(mlir::StringRef extraDecl) const {
    if (extraDecl.empty()) {
      return;
    }
    for (const ExtraMethod &method : parseExtraMethods(extraDecl)) {
      genExtraMethod(method);
    }
  }

  /// @brief Generate code for an extra method
  /// @param method The extra method to generate code for
  virtual void genExtraMethod(const ExtraMethod &method) const = 0;

protected:
  std::string kind;
  llvm::raw_ostream &os;
  std::string dialectNameCapitalized;
  const mlir::tblgen::Dialect *dialect;
  mlir::StringRef className;
};

/// @brief Generator for common C header file elements
struct HeaderGenerator : public Generator {
  using Generator::Generator;
  virtual ~HeaderGenerator() = default;

  virtual void genPrologue() const {
    os << R"(#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif
)";
  }

  virtual void genEpilogue() const {
    os << R"(
#ifdef __cplusplus
}
#endif
)";
  }

  virtual void genIsADecl() const {
    static constexpr char fmt[] = R"(
/* Returns true if the {1} is a {3}::{4}. */
MLIR_CAPI_EXPORTED bool {0}{1}IsA{2}{4}(Mlir{1});
)";
    assert(dialect && "Dialect must be set");
    os << llvm::formatv(
        fmt, FunctionPrefix, kind, dialectNameCapitalized, dialect->getCppNamespace(), className
    );
  }

  /// @brief Generate declaration for an extra method from extraClassDeclaration
  virtual void genExtraMethod(const ExtraMethod &method) const override {
    // Convert return type to C API type
    std::string capiReturnType = cppTypeToCapiType(method.returnType);

    // Skip if the return type couldn't be converted
    if (!isValidTypeConversion(capiReturnType, method.returnType)) {
      return;
    }

    // Build parameter list
    std::string paramList = llvm::formatv("Mlir{0} inp", kind).str();
    for (const auto &param : method.parameters) {
      // Convert C++ type to C API type for parameter declaration
      std::string capiParamType = cppTypeToCapiType(param.type);
      // Skip if the parameter type couldn't be converted
      if (!isValidTypeConversion(capiParamType, param.type)) {
        return;
      }
      paramList += ", " + capiParamType + " " + param.name;
    }

    // Generate declaration
    std::string docComment =
        method.documentation.empty() ? method.methodName : method.documentation;

    os << llvm::formatv("\n/* {0} */\n", docComment);
    os << llvm::formatv(
        "MLIR_CAPI_EXPORTED {0} {1}{2}{3}{4}({5});\n",
        capiReturnType,                  // {0}
        FunctionPrefix,                  // {1}
        dialectNameCapitalized,          // {2}
        className,                       // {3}
        toPascalCase(method.methodName), // {4}
        paramList                        // {5}
    );
  }
};

/// @brief Generator for common C implementation file elements
struct ImplementationGenerator : public Generator {
  using Generator::Generator;
  virtual ~ImplementationGenerator() = default;

  virtual void genIsAImpl() const {
    static constexpr char fmt[] = R"(
bool {0}{1}IsA{2}{3}(Mlir{1} inp) {{
  return llvm::isa<{3}>(unwrap(inp));
}
)";
    assert(!className.empty() && "className must be set");
    os << llvm::formatv(fmt, FunctionPrefix, kind, dialectNameCapitalized, className);
  }

  /// @brief Generate implementation for an extra method from extraClassDeclaration
  virtual void genExtraMethod(const ExtraMethod &method) const override {
    // Convert return type to C API type
    std::string capiReturnType = cppTypeToCapiType(method.returnType);

    // Skip if the return type couldn't be converted
    if (!isValidTypeConversion(capiReturnType, method.returnType)) {
      return;
    }

    // Build parameter list for C API function signature
    std::string paramList = llvm::formatv("Mlir{0} inp", kind).str();
    for (const auto &param : method.parameters) {
      // Convert C++ type to C API type for parameter declaration
      std::string capiParamType = cppTypeToCapiType(param.type);
      // Skip if the parameter type couldn't be converted
      if (!isValidTypeConversion(capiParamType, param.type)) {
        return;
      }
      paramList += ", " + capiParamType + " " + param.name;
    }

    // Build argument list for C++ method call
    std::string argList;
    for (size_t i = 0; i < method.parameters.size(); ++i) {
      if (i > 0) {
        argList += ", ";
      }
      const auto &param = method.parameters[i];
      std::string capiParamType = cppTypeToCapiType(param.type);

      // Check if parameter needs unwrapping
      if (isPrimitiveType(capiParamType)) {
        // Primitive types don't need unwrapping
        argList += param.name;
      } else if (capiParamType.starts_with("Mlir")) {
        // MLIR C API types need unwrapping
        argList += "unwrap(" + param.name + ")";
      } else {
        // Unknown types - pass through as-is
        argList += param.name;
      }
    }

    std::string capitalizedMethodName = toPascalCase(method.methodName);
    mlir::StringRef wrapCode = getReturnWrapCode(capiReturnType);

    // Build the return statement prefix and suffix
    std::string returnPrefix;
    std::string returnSuffix;

    if (capiReturnType == "void") {
      returnPrefix = "";
      returnSuffix = "";
    } else if (!wrapCode.empty()) {
      returnPrefix = std::string("return ") + wrapCode.str() + "(";
      returnSuffix = ")";
    } else {
      returnPrefix = "return ";
      returnSuffix = "";
    }

    // Generate implementation
    os << "\n";
    os << llvm::formatv(
        "{0} {1}{2}{3}{4}({5}) {{\n",
        capiReturnType,         // {0}
        FunctionPrefix,         // {1}
        dialectNameCapitalized, // {2}
        className,              // {3}
        capitalizedMethodName,  // {4}
        paramList               // {5}
    );
    os << llvm::formatv(
        "  {0}llvm::cast<{1}>(unwrap(inp)).{2}({3}){4};\n",
        returnPrefix,      // {0}
        className,         // {1}
        method.methodName, // {2}
        argList,           // {3}
        returnSuffix       // {4}
    );
    os << "}\n";
  }
};
