#pragma once

#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/SymbolTable.h>

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/raw_ostream.h>

#include <string>
#include <vector>

namespace llzk {
namespace debug {

namespace {

// Define this concept instead of `std::ranges::range` because certain classes (like OperandRange)
// do not work with `std::ranges::range`.
template <typename T>
concept Iterable = requires(T t) {
  std::begin(t);
  std::end(t);
};

struct Appender {
  llvm::raw_string_ostream stream;
  Appender(std::string &out) : stream(out) {}

  void append(const mlir::NamedAttribute &a);
  void append(const mlir::SymbolTable::SymbolUse &a);
  template <typename T> void append(const std::optional<T> &a);
  template <typename Any> void append(const Any &value);
  template <typename A, typename B> void append(const std::pair<A, B> &a);
  template <typename A, typename B> void append(const llvm::detail::DenseMapPair<A, B> &a);
  template <Iterable InputIt> void append(const InputIt &collection);
  template <typename InputIt> void appendList(InputIt begin, InputIt end);
};

void Appender::append(const mlir::NamedAttribute &a) {
  stream << a.getName() << '=' << a.getValue();
}

void Appender::append(const mlir::SymbolTable::SymbolUse &a) { stream << a.getUser()->getName(); }

template <typename T> inline void Appender::append(const std::optional<T> &a) {
  if (a.has_value()) {
    append(a.value());
  } else {
    stream << "NONE";
  }
}

template <typename Any> void Appender::append(const Any &value) { stream << value; }

template <typename A, typename B> void Appender::append(const std::pair<A, B> &a) {
  stream << '(';
  append(a.first);
  stream << ',';
  append(a.second);
  stream << ')';
}

template <typename A, typename B> void Appender::append(const llvm::detail::DenseMapPair<A, B> &a) {
  stream << '(';
  append(a.first);
  stream << ',';
  append(a.second);
  stream << ')';
}

template <Iterable InputIt> inline void Appender::append(const InputIt &collection) {
  appendList(std::begin(collection), std::end(collection));
}

template <typename InputIt> void Appender::appendList(InputIt begin, InputIt end) {
  stream << '[';
  llvm::interleave(begin, end, [this](const auto &n) { append(n); }, [this] { stream << ", "; });
  stream << ']';
}

} // namespace

/// Generate a comma-separated string representation by traversing elements from `begin` to `end`
/// where the element type implements `operator<<`.
template <typename InputIt> std::string toStringList(InputIt begin, InputIt end) {
  std::string output;
  Appender(output).appendList(begin, end);
  return output;
}

/// Generate a comma-separated string representation by traversing elements from
/// `collection.begin()` to `collection.end()` where the element type implements `operator<<`.
template <typename InputIt> inline std::string toStringList(const InputIt &collection) {
  return toStringList(collection.begin(), collection.end());
}

template <typename InputIt>
inline std::string toStringList(const std::optional<InputIt> &optionalCollection) {
  if (optionalCollection.has_value()) {
    return toStringList(optionalCollection.value());
  } else {
    return "NONE";
  }
}

template <typename T> inline std::string toStringOne(const T &value) {
  std::string output;
  Appender(output).append(value);
  return output;
}

inline void dumpSymbolTableWalk(mlir::Operation *symbolTableOp) {
  std::string output; // buffer to avoid multi-threaded mess
  llvm::raw_string_ostream oss(output);
  oss << "Dumping symbol walk (self = [" << symbolTableOp << "]): \n";
  auto walkFn = [&](mlir::Operation *op, bool allUsesVisible) {
    oss << "  found op [" << op << "] " << op->getName() << " named "
        << op->getAttrOfType<mlir::StringAttr>(mlir::SymbolTable::getSymbolAttrName()) << '\n';
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
      stream << '  ';
    }
    mlir::Operation *op = p.second;
    stream << '  ' << p.first << " -> [" << op << "] " << op->getName() << '\n';
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
    stream << "  [" << p.first << "] " << p.first->getName() << " -> " << '\n';
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
    stream << '\n';
  }
}

} // namespace debug
} // namespace llzk
