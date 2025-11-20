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

#include <llvm/ADT/StringMap.h>

#include <clang/Basic/FileManager.h>
#include <clang/Basic/LangOptions.h>
#include <clang/Basic/SourceManager.h>
#include <clang/Lex/Lexer.h>
#include <optional>

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

llvm::cl::opt<bool> GenIsA(
    "gen-isa", llvm::cl::desc("Generate IsA checks"), llvm::cl::init(true), llvm::cl::cat(OpGenCat)
);

llvm::cl::opt<bool> GenOpCreate(
    "gen-op-create", llvm::cl::desc("Generate operation create functions"), llvm::cl::init(true),
    llvm::cl::cat(OpGenCat)
);

llvm::cl::opt<bool> GenOpNameGetter(
    "gen-op-name-getter", llvm::cl::desc("Generate operation name getter"), llvm::cl::init(true),
    llvm::cl::cat(OpGenCat)
);

llvm::cl::opt<bool> GenOpOperandGetters(
    "gen-operand-getters", llvm::cl::desc("Generate operand getters for operations"),
    llvm::cl::init(true), llvm::cl::cat(OpGenCat)
);

llvm::cl::opt<bool> GenOpOperandSetters(
    "gen-operand-setters", llvm::cl::desc("Generate operand setters for operations"),
    llvm::cl::init(true), llvm::cl::cat(OpGenCat)
);

llvm::cl::opt<bool> GenOpAttributeGetters(
    "gen-attribute-getters", llvm::cl::desc("Generate attribute getters for operations"),
    llvm::cl::init(true), llvm::cl::cat(OpGenCat)
);

llvm::cl::opt<bool> GenOpAttributeSetters(
    "gen-attribute-setters", llvm::cl::desc("Generate attribute setters for operations"),
    llvm::cl::init(true), llvm::cl::cat(OpGenCat)
);

llvm::cl::opt<bool> GenOpRegionGetters(
    "gen-region-getters", llvm::cl::desc("Generate region getters for operations"),
    llvm::cl::init(true), llvm::cl::cat(OpGenCat)
);

llvm::cl::opt<bool> GenOpResultGetters(
    "gen-result-getters", llvm::cl::desc("Generate result getters for operations"),
    llvm::cl::init(true), llvm::cl::cat(OpGenCat)
);

llvm::cl::opt<bool> GenTypeOrAttrGet(
    "gen-type-attr-get", llvm::cl::desc("Generate get functions for types and attributes"),
    llvm::cl::init(true), llvm::cl::cat(OpGenCat)
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
  std::unique_ptr<Lexer> lexer;

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
  impl->lexer = std::make_unique<Lexer>(fileID, bufferRef, *impl->sourceMgr, impl->langOpts);
  // Enable comment parsing for extraClassDeclaration method extraction
  impl->lexer->SetCommentRetentionState(true);
  lexer = impl->lexer.get();
}

Lexer &ClangLexerContext::getLexer() const {
  assert(lexer && "Lexer not initialized");
  return *lexer;
}

SourceManager &ClangLexerContext::getSourceManager() const {
  assert(impl->sourceMgr && "SourceManager not initialized");
  return *impl->sourceMgr;
}

namespace {

static inline bool isAccessModifier(StringRef tokenText) {
  return tokenText == "private" || tokenText == "public" || tokenText == "protected";
}

/// Collect all tokens first for easier lookahead parsing
/// For very large extraClassDeclaration blocks, this could use streaming,
/// but in practice these blocks are small (typically < 100 tokens)
static inline std::vector<Token> tokenize(const ClangLexerContext &lexerCtx) {
  Lexer &lexer = lexerCtx.getLexer();
  std::vector<Token> tokens;
  tokens.reserve(128); // Reasonable default
  for (Token tok; !lexer.LexFromRawLexer(tok);) {
    if (tok.is(tok::eof)) {
      break;
    }
    tokens.push_back(tok);
  }
  return tokens;
}

/// Extract documentation from comment tokens preceding the function declaration.
static inline std::string getDocumentation(
    size_t returnTypeStart, const std::vector<Token> &tokens, const SourceManager &sourceMgr
) {
  std::string documentation;
  for (size_t j = returnTypeStart; j > 0; --j) {
    Token curr = tokens[j - 1];
    if (curr.is(tok::comment)) {
      std::string comment(sourceMgr.getCharacterData(curr.getLocation()), curr.getLength());
      StringRef cleanComment(comment);
      cleanComment.consume_front("///");
      cleanComment.consume_front("//");
      if (cleanComment.consume_front("/*")) {
        cleanComment.consume_back("*/");
      }
      cleanComment = cleanComment.trim();

      // Trim whitespace
      if (!cleanComment.empty()) {
        if (!documentation.empty()) {
          documentation = cleanComment.str() + " " + documentation;
        } else {
          documentation = cleanComment.str();
        }
      }
    } else if (!curr.is(tok::unknown)) {
      // Stop looking backwards when we hit a non-comment, non-whitespace token
      // that could be part of another declaration
      if (curr.is(tok::semi) || curr.is(tok::r_brace) || curr.is(tok::l_brace)) {
        break;
      }
      if (curr.is(tok::raw_identifier) && isAccessModifier(curr.getRawIdentifier())) {
        break;
      }
    }
  }
  return documentation;
}

} // namespace

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
SmallVector<ExtraMethod> parseExtraMethods(StringRef extraDecl) {
  if (extraDecl.empty()) {
    return {};
  }

  // Use ClangLexerContext for simplified setup
  const ClangLexerContext lexerCtx(extraDecl, "extraClassDecl");
  if (!lexerCtx.isValid()) {
    llvm::errs() << "Error: Failed to create lexer context for parseExtraMethods\n";
    return {};
  }

  // Store methods uniqued by name to detect and skip overloads (duplicate method names).
  llvm::StringMap<std::optional<ExtraMethod>> methods;

  // Parse tokens to find method declarations
  const std::vector<Token> tokens = tokenize(lexerCtx);
  const size_t tokenCount = tokens.size();
  const SourceManager &sourceMgr = lexerCtx.getSourceManager();

  // Track current access level to avoid generating C API wrappers for private functions. Code
  // generated by `mlir-tblgen` puts the extra declarations in the public section by default.
  enum class AccessLevel { Public, Private, Protected };
  AccessLevel currentAccess = AccessLevel::Public;

  for (size_t i = 0; i < tokenCount; ++i) {
    // Skip comments (they'll be extracted separately)
    if (tokens[i].is(tok::comment)) {
      continue;
    }

    // Check for access specifier changes (e.g., "private:", "public:", "protected:").
    // In raw token stream, these appear as a `raw_identifier` rather than `kw_*`.
    if (i + 1 < tokenCount && tokens[i + 1].is(tok::colon)) {
      if (tokens[i].is(tok::raw_identifier)) {
        StringRef name = tokens[i].getRawIdentifier();
        if (name == "private") {
          currentAccess = AccessLevel::Private;
          i++; // extra skip for the colon
          continue;
        } else if (name == "public") {
          currentAccess = AccessLevel::Public;
          i++; // extra skip for the colon
          continue;
        } else if (name == "protected") {
          currentAccess = AccessLevel::Protected;
          i++; // extra skip for the colon
          continue;
        }
      }
    }

    // Skip private and protected methods - no need to generate C API wrappers
    if (currentAccess != AccessLevel::Public) {
      continue;
    }

    // Look for pattern: [modifiers] <return_type> <identifier> '(' [params] ')' [const] ';'
    //
    // Look for an identifier followed by '('
    if (i + 1 < tokenCount && tokens[i + 1].is(tok::l_paren) && tokens[i].is(tok::raw_identifier)) {
      StringRef methodName = tokens[i].getRawIdentifier();

      // Skip control flow keywords and other language constructs that use parentheses
      if (isCppLanguageConstruct(methodName)) {
        continue;
      }

      // Extract return type (everything before method name)
      std::string returnType;
      size_t returnTypeStart = 0;
      {
        bool isStaticMethod = false;
        // Look backwards for return type start, stopping at declaration boundaries
        for (size_t j = i; j > 0; --j) {
          Token curr = tokens[j - 1];
          // Semicolon or right brace indicates lookback has reached the end of a prior declaration.
          if (curr.is(tok::semi) || curr.is(tok::r_brace)) {
            returnTypeStart = j;
            break;
          }
          // Check for "static" or access modifiers in the return type lookback (both appear as
          // `raw_identifier` in raw token stream).
          if (curr.is(tok::raw_identifier)) {
            StringRef text = curr.getRawIdentifier();
            if (text == "static") {
              isStaticMethod = true;
              break;
            }
            if (tokens[j].is(tok::colon) && isAccessModifier(text)) {
              // In this case, `returnTypeStart` must be after the colon.
              returnTypeStart = j + 1;
              // ASSERT: Safe because `j<=i` is a invariant in the wrapping loop and
              // there is an if-condition outside the loop for `i + 1 < tokenCount`.
              assert(returnTypeStart < tokenCount);
              break;
            }
          }
        }
        // Skip static methods (for now)
        if (isStaticMethod) {
          continue;
        }

        // Adjust `returnTypeStart` for potential comment tokens. The check here is needed instead
        // of having a comment case in the loop above since that could get stuck on inline comments
        // appearing within the sequence of tokens that make up the return type. Skip as many
        // sequential comments as needed.
        while (tokens[returnTypeStart].is(tok::comment)) {
          returnTypeStart++;
          // ASSERT: This loop cannot run past `i` because the outer if-condition
          // checks `tokens[i].is(tok::raw_identifier)`.
          assert(returnTypeStart <= i);
        }

        // Build return type from tokens, skipping modifiers and comments.
        // Use raw_string_ostream for efficient string building.
        returnType.reserve(32); // Reasonable default for most type names
        llvm::raw_string_ostream returnTypeStream(returnType);

        for (size_t j = returnTypeStart; j < i; ++j) {
          // Skip comments - they should be extracted as documentation, not part of the return type
          if (tokens[j].is(tok::comment)) {
            continue;
          }

          std::string tokenText(
              sourceMgr.getCharacterData(tokens[j].getLocation()), tokens[j].getLength()
          );

          // Skip access specifiers (e.g., "private", "public", "protected").
          // These can appear with or without colons in the token stream.
          // In raw token stream, these appear as a `raw_identifier` rather than `kw_*`.
          if (tokens[j].is(tok::raw_identifier) && isAccessModifier(tokenText)) {
            // If followed by a colon, skip that too
            if (j + 1 < i && tokens[j + 1].is(tok::colon)) {
              j++; // Skip the colon too
            }
            continue;
          }

          // Skip common implementation keywords that indicate we're in code, not a declaration.
          // In raw token stream, this appears as a `raw_identifier` rather than `kw_return`.
          if (tokens[j].is(tok::raw_identifier) && tokenText == "return") {
            // This indicates we've hit implementation code, stop parsing
            returnType.clear();
            break;
          }

          // Skip modifiers and language keywords that shouldn't be in the return type
          if (tokens[j].is(tok::raw_identifier) && isCppModifierKeyword(tokenText)) {
            continue;
          }

          // Skip standalone colons (from lookback to access specifiers)
          if (tokens[j].is(tok::colon)) {
            // Only skip if it's not part of ::
            if (j == 0 || j + 1 >= i || !tokens[j - 1].is(tok::colon)) {
              continue;
            }
          }

          // Add spacing between tokens (but not around ::)
          if (!returnType.empty() && !returnType.ends_with("::") && tokenText != "::" &&
              !tokenText.starts_with("::")) {
            returnTypeStream << ' ';
          }
          returnTypeStream << tokenText;
        }
        // Trim possible whitespace
        returnType = StringRef(returnType).trim().str();
      }

      // Find matching ')' and check for parameters
      size_t closeParenIdx = tokenCount;
      bool hasParameters = false;
      std::vector<MethodParameter> parameters;
      {
        // Initialize parenDepth to 1 to account for the opening '(' at tokens[i+1]
        // Start scanning from i+2 (the first token after the opening paren)
        size_t parenDepth = 1;
        for (size_t j = i + 2; j < tokenCount; ++j) {
          if (tokens[j].is(tok::l_paren)) {
            parenDepth++;
          } else if (tokens[j].is(tok::r_paren)) {
            parenDepth--;
            if (parenDepth == 0) {
              closeParenIdx = j;

              // Parse parameters between '(' and ')'
              // Parameters follow the pattern: type name [, type name ...]
              std::vector<Token> paramTokens;
              for (size_t k = i + 2; k < j; ++k) {
                if (k >= tokenCount) {
                  break;
                }
                if (!tokens[k].is(tok::comment)) {
                  paramTokens.push_back(tokens[k]);
                }
              }

              // Check if we have actual parameters
              if (!paramTokens.empty()) {
                // Check if it's just "void"
                if (paramTokens.size() == 1) {
                  std::string paramToken(
                      sourceMgr.getCharacterData(paramTokens[0].getLocation()),
                      paramTokens[0].getLength()
                  );
                  if (paramToken != "void") {
                    hasParameters = true;
                  }
                } else {
                  hasParameters = true;
                }
              }

              // Parse individual parameters
              if (hasParameters) {
                std::string currentParamType;
                std::string currentParamName;
                bool inDefaultValue = false;

                for (size_t k = 0; k < paramTokens.size(); ++k) {
                  std::string tokenText(
                      sourceMgr.getCharacterData(paramTokens[k].getLocation()),
                      paramTokens[k].getLength()
                  );

                  // Check for '=' which indicates start of default value
                  if (paramTokens[k].is(tok::equal)) {
                    inDefaultValue = true;
                    continue;
                  }

                  if (paramTokens[k].is(tok::comma)) {
                    // End of current parameter
                    if (!currentParamType.empty() && !currentParamName.empty()) {
                      MethodParameter param;
                      param.type = StringRef(currentParamType).trim().str();
                      param.name = StringRef(currentParamName).trim().str();
                      parameters.push_back(param);
                    }
                    currentParamType.clear();
                    currentParamName.clear();
                    inDefaultValue = false;
                  } else if (inDefaultValue) {
                    // Skip tokens that are part of the default value
                    continue;
                  } else if (paramTokens[k].is(tok::raw_identifier)) {
                    // Could be part of type or the parameter name
                    // Simple heuristic: last identifier before comma, '=' or end is the name
                    if (k + 1 < paramTokens.size() &&
                        (paramTokens[k + 1].is(tok::comma) || paramTokens[k + 1].is(tok::equal) ||
                         k + 1 == paramTokens.size())) {
                      // This is the parameter name
                      currentParamName = tokenText;
                    } else if (k + 1 == paramTokens.size()) {
                      // Last token, must be parameter name
                      currentParamName = tokenText;
                    } else {
                      // Part of the type
                      if (!currentParamType.empty()) {
                        currentParamType += " ";
                      }
                      currentParamType += tokenText;
                    }
                  } else {
                    // Other tokens (keywords, ::, *, &, etc.) - part of type
                    if (!currentParamType.empty() && !StringRef(currentParamType).ends_with("::") &&
                        tokenText != "::" && !tokenText.starts_with("::") && tokenText != "*" &&
                        tokenText != "&") {
                      currentParamType += " ";
                    }
                    currentParamType += tokenText;
                  }
                }

                // Add the last parameter
                if (!currentParamType.empty() && !currentParamName.empty()) {
                  MethodParameter param;
                  param.type = StringRef(currentParamType).trim().str();
                  param.name = StringRef(currentParamName).trim().str();
                  parameters.push_back(param);
                }
              }

              break;
            }
          }
        }
      }
      if (closeParenIdx >= tokenCount) {
        // Couldn't find closing paren, skip this
        continue;
      }

      // Check for 'const' after parameters but before end of declaration (';' or '{').
      bool isConst = false;
      size_t endIdx = closeParenIdx + 1;
      while (endIdx < tokenCount) {
        Token curr = tokens[endIdx];
        if (curr.is(tok::semi) || curr.is(tok::l_brace)) {
          break;
        }
        if (curr.is(tok::raw_identifier) && curr.getRawIdentifier() == "const") {
          isConst = true;
        }
        endIdx++;
      }

      // Create method struct
      if (!returnType.empty() && !methodName.empty()) {
        if (methods.contains(methodName)) {
          llvm::errs() << "Warning: Skipping overloaded method '" << methodName
                       << "' - C API does not support method overloading\n";
          methods[methodName] = std::nullopt;
        } else {
          ExtraMethod method;
          method.returnType = returnType;
          method.methodName = methodName;
          method.documentation = getDocumentation(returnTypeStart, tokens, sourceMgr);
          method.isConst = isConst;
          method.hasParameters = hasParameters;
          method.parameters = parameters;
          methods[methodName] = std::make_optional(method);
        }
      }

      // Skip to end of this declaration for the next iteration.
      i = endIdx;
    }
  }

  // Return valid methods, skipping overloaded names (nullopt entries).
  return llvm::to_vector(
      llvm::map_range(
          llvm::make_filter_range(methods, [](const auto &p) { return p.second.has_value(); }),
          [](const auto &p) { return p.second.value(); }
      )
  );
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
/// Check if conversion failed by seeing if the type is unchanged and not a known primitive
bool isValidTypeConversion(const std::string &capiReturnType, const std::string &cppReturnType) {
  if (capiReturnType == cppReturnType && !isPrimitiveType(capiReturnType)) {
    llvm::errs() << "Error: Unsupported type conversion from C++ type '" << cppReturnType
                 << "' to C API type\n";
    return false;
  }
  return true;
}
