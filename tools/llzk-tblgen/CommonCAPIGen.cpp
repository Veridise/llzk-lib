//===- CommonCAPIGen.cpp - Common utilities for C API generation ---------===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// Shared command-line options for all CAPI generators (ops, attrs, types)
//
//===----------------------------------------------------------------------===//

#include "CommonCAPIGen.h"

#include <clang/Basic/FileManager.h>
#include <clang/Basic/LangOptions.h>
#include <clang/Basic/SourceManager.h>
#include <clang/Lex/Lexer.h>

using namespace mlir;
using namespace clang;

llvm::cl::OptionCategory
    OpGenCat("Options for -gen-op-capi-header, -gen-op-capi-impl, and -gen-op-capi-tests");

llvm::cl::opt<std::string> DialectName(
    "dialect",
    llvm::cl::desc(
        "The dialect name to use for this group of ops. "
        "Must match across header, implementation, and test generation."
    ),
    llvm::cl::cat(OpGenCat)
);

llvm::cl::opt<std::string> FunctionPrefix(
    "prefix",
    llvm::cl::desc(
        "The prefix to use for generated C API function names. "
        "Default is 'mlir'. Must match across header, implementation, and test generation."
    ),
    llvm::cl::init("mlir"), llvm::cl::cat(OpGenCat)
);

llvm::cl::opt<bool> GenIsAChecks(
    "gen-isa-checks", llvm::cl::desc("Generate IsA type checks"), llvm::cl::init(true),
    llvm::cl::cat(OpGenCat)
);

llvm::cl::opt<bool> GenIsATests(
    "gen-isa-tests", llvm::cl::desc("Generate tests for IsA type checks"), llvm::cl::init(true),
    llvm::cl::cat(OpGenCat)
);

llvm::cl::opt<bool> GenTypeOrAttrParamGetters(
    "gen-parameter-getters", llvm::cl::desc("Generate parameter getters for types and attributes"),
    llvm::cl::init(true), llvm::cl::cat(OpGenCat)
);

llvm::cl::opt<bool> GenExtraClassMethods(
    "gen-extra-class-methods",
    llvm::cl::desc("Generate C API wrappers for methods in `extraClassDeclaration`"),
    llvm::cl::init(true), llvm::cl::cat(OpGenCat)
);

//===----------------------------------------------------------------------===//
// ClangLexerContext Implementation
//===----------------------------------------------------------------------===//
struct ClangLexerContext::Impl {
  /// C++ language options for lexer configuration
  LangOptions langOpts;
  /// File manager for handling virtual files
  IntrusiveRefCntPtr<FileManager> fileMgr;
  /// Diagnostic IDs for error reporting
  IntrusiveRefCntPtr<DiagnosticIDs> diagIDs;
  /// Diagnostic options for configuring diagnostics
  IntrusiveRefCntPtr<DiagnosticOptions> diagOpts;
  /// Diagnostics engine for handling errors and warnings
  std::unique_ptr<DiagnosticsEngine> diags;
  /// Source manager for tracking file locations
  std::unique_ptr<SourceManager> sourceMgr;
  /// The actual lexer instance
  std::unique_ptr<clang::Lexer> lexer;

  Impl() : diagIDs(new DiagnosticIDs()), diagOpts(new DiagnosticOptions()) {
    // Enable C++ language features for lexing
    langOpts.CPlusPlus = true;
    langOpts.CPlusPlus11 = true;

    FileSystemOptions fileSystemOpts;
    fileMgr = new FileManager(fileSystemOpts);
    diags = std::make_unique<DiagnosticsEngine>(diagIDs, diagOpts);
    sourceMgr = std::make_unique<SourceManager>(*diags, *fileMgr);
  }
};

ClangLexerContext::ClangLexerContext(StringRef source, StringRef bufferName)
    : impl(std::make_unique<Impl>()) {
  if (source.empty()) {
    llvm::errs() << "Warning: ClangLexerContext created with empty source\n";
    return;
  }

  // Create a memory buffer for the input
  std::unique_ptr<llvm::MemoryBuffer> buffer = llvm::MemoryBuffer::getMemBuffer(source, bufferName);
  if (!buffer) {
    llvm::errs() << "Error: Failed to create memory buffer for ClangLexerContext\n";
    return;
  }

  FileID fileID = impl->sourceMgr->createFileID(std::move(buffer), SrcMgr::C_User);
  llvm::MemoryBufferRef bufferRef = impl->sourceMgr->getBufferOrFake(fileID);

  if (bufferRef.getBufferSize() == 0 && !source.empty()) {
    llvm::errs() << "Error: Failed to get buffer from source manager in ClangLexerContext\n";
    return;
  }

  // Create the lexer
  impl->lexer = std::make_unique<clang::Lexer>(fileID, bufferRef, *impl->sourceMgr, impl->langOpts);
  lexer = impl->lexer.get();
}

clang::Lexer &ClangLexerContext::getLexer() const {
  assert(lexer && "Lexer not initialized");
  return *lexer;
}

SourceManager &ClangLexerContext::getSourceManager() const {
  assert(impl->sourceMgr && "SourceManager not initialized");
  return *impl->sourceMgr;
}

//===----------------------------------------------------------------------===//
// Method Parsing Implementation
//===----------------------------------------------------------------------===//

/// Parse method declarations from extraClassDeclaration using Clang's Lexer
///
/// This function parses C++ method declarations to extract method signatures.
/// It identifies methods by looking for the pattern: <return_type> <identifier> '(' [params] ')'
/// [const] ';'
///
/// Note: Currently, only methods without parameters are fully supported. Methods with
/// parameters are detected but skipped during code generation.
std::vector<ExtraMethod> parseExtraMethods(StringRef extraDecl) {
  std::vector<ExtraMethod> methods;

  if (extraDecl.empty()) {
    return methods;
  }

  // Use ClangLexerContext for simplified setup
  ClangLexerContext lexerCtx(extraDecl, "extraClassDecl");
  if (!lexerCtx.isValid()) {
    llvm::errs() << "Error: Failed to create lexer context for parseExtraMethods\n";
    return methods;
  }

  clang::Lexer &lexer = lexerCtx.getLexer();
  SourceManager &sourceMgr = lexerCtx.getSourceManager();

  // Token stream parsing state
  Token tok;
  std::vector<Token> tokens;

  // Collect all tokens first for easier lookahead parsing
  // For very large extraClassDeclaration blocks, this could use streaming,
  // but in practice these blocks are small (typically < 100 tokens)
  tokens.reserve(128); // Reasonable default
  while (!lexer.LexFromRawLexer(tok)) {
    if (tok.is(clang::tok::eof)) {
      break;
    }
    tokens.push_back(tok);
  }

  // Parse tokens to find method declarations
  for (size_t i = 0; i < tokens.size(); ++i) {
    // Look for pattern: [modifiers] <return_type> <identifier> '(' [params] ')' [const] ';'

    // Skip comments (they'll be extracted separately)
    if (tokens[i].is(clang::tok::comment)) {
      continue;
    }

    // Look for an identifier followed by '('
    if (i + 1 < tokens.size() && tokens[i].is(clang::tok::identifier) &&
        tokens[i + 1].is(clang::tok::l_paren)) {

      std::string methodName =
          std::string(sourceMgr.getCharacterData(tokens[i].getLocation()), tokens[i].getLength());

      // Skip control flow keywords and other language constructs that use parentheses
      if (isCppLanguageConstruct(methodName)) {
        continue;
      }

      // Extract return type (everything before method name)
      std::string returnType;
      size_t returnTypeStart = 0;
      {
        // Look backwards for return type start, skipping modifiers
        for (size_t j = i; j > 0; --j) {
          if (tokens[j - 1].is(tok::semi) || tokens[j - 1].is(tok::r_brace) ||
              tokens[j - 1].is(tok::l_brace)) {
            returnTypeStart = j;
            break;
          }
        }

        // Build return type from tokens, skipping modifiers
        // Use raw_string_ostream for efficient string building
        std::string returnTypeBuffer;
        returnTypeBuffer.reserve(128); // Reasonable default for most type names
        llvm::raw_string_ostream returnTypeStream(returnTypeBuffer);

        for (size_t j = returnTypeStart; j < i; ++j) {
          if (j >= tokens.size()) {
            llvm::errs() << "Error: Token index out of bounds while parsing return type\n";
            break;
          }

          std::string tokenText(
              sourceMgr.getCharacterData(tokens[j].getLocation()), tokens[j].getLength()
          );

          // Skip modifiers
          if (isCppModifierKeyword(tokenText)) {
            continue;
          }

          // Add spacing between tokens (but not around ::)
          if (!returnTypeBuffer.empty() && !StringRef(returnTypeBuffer).ends_with("::") &&
              tokenText != "::" && !tokenText.starts_with("::")) {
            returnTypeStream << ' ';
          }
          returnTypeStream << tokenText;
        }
        returnType = returnTypeStream.str();
      }

      // Find matching ')' and check for parameters
      size_t parenDepth = 0;
      size_t closeParenIdx = i + 1;
      bool hasParameters = false;
      size_t paramTokenCount = 0;

      for (size_t j = i + 1; j < tokens.size(); ++j) {
        if (tokens[j].is(tok::l_paren)) {
          parenDepth++;
        } else if (tokens[j].is(tok::r_paren)) {
          if (parenDepth == 0) {
            closeParenIdx = j;

            // Check if there are non-whitespace/comment tokens between '(' and ')'
            for (size_t k = i + 2; k < j; ++k) {
              if (k >= tokens.size()) {
                break;
              }
              if (!tokens[k].is(tok::comment)) {
                std::string paramToken(
                    sourceMgr.getCharacterData(tokens[k].getLocation()), tokens[k].getLength()
                );
                paramTokenCount++;
                // Consider it as having parameters if not just "void"
                if (paramToken != "void" || paramTokenCount > 1) {
                  hasParameters = true;
                  break;
                }
              }
            }
            break;
          }
          parenDepth--;
        }
      }

      if (closeParenIdx >= tokens.size()) {
        // Couldn't find closing paren, skip this
        continue;
      }

      // Check for 'const' and find end of declaration (';' or '{')
      bool isConst = false;
      size_t endIdx = closeParenIdx + 1;

      while (endIdx < tokens.size()) {
        if (tokens[endIdx].is(tok::kw_const)) {
          isConst = true;
          endIdx++;
        } else if (tokens[endIdx].is(tok::semi) || tokens[endIdx].is(tok::l_brace)) {
          break;
        } else if (tokens[endIdx].is(tok::comment)) {
          endIdx++;
        } else {
          // Some other token; might be noexcept, override, etc. - skip for now
          endIdx++;
        }
      }

      // Extract documentation from preceding comment tokens
      std::string documentation;
      for (size_t j = returnTypeStart; j > 0; --j) {
        if (tokens[j - 1].is(tok::comment)) {
          std::string comment(
              sourceMgr.getCharacterData(tokens[j - 1].getLocation()), tokens[j - 1].getLength()
          );

          // Strip comment markers
          if (comment.starts_with("///")) {
            comment = comment.substr(3);
          } else if (comment.starts_with("//")) {
            comment = comment.substr(2);
          } else if (comment.starts_with("/*") && comment.ends_with("*/")) {
            comment = comment.substr(2, comment.length() - 4);
          }

          // Trim whitespace
          StringRef trimmedComment = StringRef(comment).trim();
          if (!trimmedComment.empty()) {
            if (!documentation.empty()) {
              documentation = trimmedComment.str() + " " + documentation;
            } else {
              documentation = trimmedComment.str();
            }
          }
        } else if (!tokens[j - 1].is(tok::comment)) {
          // Stop looking backwards when we hit a non-comment token
          // (unless it's the start of our return type)
          if (j - 1 < returnTypeStart) {
            break;
          }
        }
      }

      // Trim return type
      returnType = StringRef(returnType).trim().str();

      // Create method struct
      if (!returnType.empty() && !methodName.empty()) {
        ExtraMethod method;
        method.returnType = returnType;
        method.methodName = methodName;
        method.documentation = documentation;
        method.isConst = isConst;
        method.hasParameters = hasParameters;

        methods.push_back(method);
      }

      // Skip to end of this declaration
      i = endIdx;
    }
  }

  return methods;
}

/// Check if a C++ type matches an MLIR type pattern
bool matchesMLIRType(StringRef cppType, StringRef typeName) {
  if (cppType == typeName) {
    return true;
  }

  // Check for "::mlir::" or "mlir::" prefix
  StringRef prefix = cppType;
  prefix.consume_front("::");
  if (prefix.consume_front("mlir::")) {
    return prefix == typeName;
  }

  return false;
}

/// Convert C++ return type to MLIR C API type
std::string cppTypeToCapiType(StringRef cppType) {
  cppType = cppType.trim();

  // Handle primitive types
  if (isPrimitiveType(cppType)) {
    return cppType.str();
  }

  // Handle pointer types
  if (cppType.ends_with(" *") || cppType.ends_with("*")) {
    size_t starPos = cppType.rfind('*');
    if (starPos != StringRef::npos) {
      StringRef baseType = cppType.substr(0, starPos).trim();

      if (matchesMLIRType(baseType, "Region") || baseType.ends_with("Region")) {
        return "MlirRegion";
      }
      if (matchesMLIRType(baseType, "Operation") || baseType.ends_with("Operation")) {
        return "MlirOperation";
      }
    } else {
      llvm::errs() << "Error: Failed to parse pointer type: " << cppType << "\n";
    }
  }

  // Handle unsupported template types (ArrayRef, SmallVector, etc.)
  if (cppType.contains("ArrayRef") || cppType.contains("SmallVector") ||
      cppType.contains("iterator_range")) {
    return cppType.str(); // Return as-is, will be skipped by caller
  }

  // Map MLIR types to their C API equivalents
  if (matchesMLIRType(cppType, "Value")) {
    return "MlirValue";
  }
  if (matchesMLIRType(cppType, "Type") || cppType.ends_with("Type")) {
    return "MlirType";
  }
  if (matchesMLIRType(cppType, "Attribute") || cppType.ends_with("Attr")) {
    return "MlirAttribute";
  }
  if (matchesMLIRType(cppType, "Region")) {
    return "MlirRegion";
  }
  if (matchesMLIRType(cppType, "Block")) {
    return "MlirBlock";
  }
  if (matchesMLIRType(cppType, "Operation")) {
    return "MlirOperation";
  }

  // Default: return as-is (may need manual wrapping)
  return cppType.str();
}

/// Determine the wrapping code needed for a return value
StringRef getReturnWrapCode(StringRef capiType) {
  // Primitive types need no wrapping
  if (isPrimitiveType(capiType)) {
    return "";
  }

  // All MLIR C API types use "wrap"
  if (capiType.starts_with("Mlir")) {
    return "wrap";
  }

  return ""; // Unknown, no wrapping
}

/// Check if a return type conversion is valid for C API generation
bool isValidReturnTypeConversion(
    const std::string &capiReturnType, const std::string &cppReturnType
) {
  // Skip if the return type couldn't be converted (e.g., ArrayRef, SmallVector)
  // Check if conversion failed by seeing if the type is unchanged and not a known primitive
  if (capiReturnType == cppReturnType && !isPrimitiveType(capiReturnType) &&
      !isIntegerType(capiReturnType)) {
    llvm::errs() << "Error: Unsupported return type conversion from C++ type '" << cppReturnType
                 << "' to C API type\n";
    return false;
  }
  return true;
}
