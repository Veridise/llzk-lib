#pragma once

#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/SymbolTable.h>

#include <llvm/ADT/DenseMap.h>
#include <llvm/Support/raw_ostream.h>

#include <string>
#include <vector>

namespace llzk {
namespace debug {

namespace {
void append(llvm::raw_ostream &ss, const mlir::NamedAttribute &a) {
  ss << a.getName() << '=' << a.getValue();
}
template <typename A, typename B>
void append(llvm::raw_ostream &ss, const llvm::detail::DenseMapPair<A, B> &a) {
  ss << '(' << a.first << ',' << a.second << ')';
}
template <typename A, typename B> void append(llvm::raw_ostream &ss, const std::pair<A, B> &a) {
  ss << '(' << a.first << ',' << a.second << ')';
}
template <typename Any> void append(llvm::raw_ostream &ss, const Any &value) { ss << value; }
} // namespace

/// Generate a comma-separated string representation by traversing elements from `begin` to `end`
/// where the element type implements `operator<<`.
template <typename InputIt> std::string toStringList(InputIt begin, InputIt end) {
  std::string output;
  llvm::raw_string_ostream oss(output);
  oss << "[";
  for (auto it = begin; it != end; ++it) {
    append(oss, *it);
    if (std::next(it) != end) {
      oss << ", ";
    }
  }
  oss << "]";
  return output;
}

/// Generate a comma-separated string representation by traversing elements from
/// `collection.begin()` to `collection.end()` where the element type implements `operator<<`.
template <typename InputIt> inline std::string toStringList(const InputIt &collection) {
  return toStringList(collection.begin(), collection.end());
}

template <typename T> inline std::string toStringOne(const T &value) {
  std::string output;
  llvm::raw_string_ostream oss(output);
  append(oss, value);
  return output;
}

inline void dumpSymbolTableWalk(mlir::Operation *symbolTableOp) {
  std::string output; // buffer to avoid multi-threaded mess
  llvm::raw_string_ostream oss(output);
  oss << "Dumping symbol walk (self = [" << symbolTableOp << "]): \n";
  auto walkFn = [&](mlir::Operation *op, bool allUsesVisible) {
    oss << "  found op [" << op << "] " << op->getName() << " named "
        << op->getAttrOfType<mlir::StringAttr>(mlir::SymbolTable::getSymbolAttrName()) << "\n";
  };
  mlir::SymbolTable::walkSymbolTables(symbolTableOp, /*allSymUsesVisible=*/true, walkFn);
  llvm::outs() << output;
}

inline void
dumpSymbolTable(llvm::raw_ostream &stream, mlir::SymbolTable &symTab, unsigned indent = 0) {
  for (unsigned i = 0; i < indent; ++i) {
    stream << "  ";
  }
  stream << "Dumping SymbolTable [" << &symTab << "]: \n";
  auto *rawSymbolTablePtr = reinterpret_cast<char *>(&symTab);
  auto *privateFieldPtr =
      reinterpret_cast<llvm::DenseMap<mlir::Attribute, mlir::Operation *> *>(rawSymbolTablePtr + 8);
  for (llvm::detail::DenseMapPair<mlir::Attribute, mlir::Operation *> &p : *privateFieldPtr) {
    for (unsigned i = 0; i < indent; ++i) {
      stream << "  ";
    }
    mlir::Operation *op = p.second;
    stream << "  " << p.first << " -> [" << op << "] " << op->getName() << "\n";
  }
}

inline void dumpSymbolTable(mlir::SymbolTable &symTab) {
  std::string output; // buffer to avoid multi-threaded mess
  llvm::raw_string_ostream oss(output);
  dumpSymbolTable(oss, symTab);
  llvm::outs() << output;
}

inline void dumpSymbolTables(llvm::raw_ostream &stream, mlir::SymbolTableCollection &tables) {
  stream << "Dumping SymbolTableCollection [" << &tables << "]: \n";
  auto *rawObjectPtr = reinterpret_cast<char *>(&tables);
  auto *privateFieldPtr =
      reinterpret_cast<llvm::DenseMap<mlir::Operation *, std::unique_ptr<mlir::SymbolTable>> *>(
          rawObjectPtr + 0
      );
  for (llvm::detail::DenseMapPair<mlir::Operation *, std::unique_ptr<mlir::SymbolTable>> &p :
       *privateFieldPtr) {
    stream << "  [" << p.first << "] " << p.first->getName() << " -> " << "\n";
    dumpSymbolTable(stream, *p.second.get(), 2);
  }
}

inline void dumpSymbolTables(mlir::SymbolTableCollection &tables) {
  std::string output; // buffer to avoid multi-threaded mess
  llvm::raw_string_ostream oss(output);
  dumpSymbolTables(oss, tables);
  llvm::outs() << output;
}

inline void dumpToFile(mlir::Operation *op, llvm::StringRef filename) {
  std::error_code err;
  llvm::raw_fd_stream stream(filename, err);
  if (!err) {
    auto options = mlir::OpPrintingFlags().assumeVerified().useLocalScope();
    op->print(stream, options);
    stream << "\n";
  }
}

} // namespace debug
} // namespace llzk
