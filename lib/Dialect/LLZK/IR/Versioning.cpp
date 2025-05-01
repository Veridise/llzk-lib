

#include "llzk/Config/Config.h"
#include "llzk/Dialect/Shared/Versioning.h"

#include <mlir/Support/LLVM.h>

using namespace mlir;

namespace llzk {

//===------------------------------------------------------------------===//
// LLZKDialectVersion
//===------------------------------------------------------------------===//

const LLZKDialectVersion &LLZKDialectVersion::CurrentVersion() {
  static LLZKDialectVersion current(LLZK_VERSION_MAJOR, LLZK_VERSION_MINOR, LLZK_VERSION_PATCH);
  return current;
}

FailureOr<LLZKDialectVersion> LLZKDialectVersion::read(DialectBytecodeReader &reader) {
  LLZKDialectVersion v;
  if (failed(reader.readVarInt(v.majorVersion)) || failed(reader.readVarInt(v.minorVersion)) ||
      failed(reader.readVarInt(v.patchVersion))) {
    return failure();
  }
  return v;
}

void LLZKDialectVersion::write(DialectBytecodeWriter &writer) const {
  writer.writeVarInt(majorVersion);
  writer.writeVarInt(minorVersion);
  writer.writeVarInt(patchVersion);
}

std::string LLZKDialectVersion::str() const {
  return (Twine(majorVersion) + "." + Twine(minorVersion) + "." + Twine(patchVersion)).str();
}

std::strong_ordering LLZKDialectVersion::operator<=>(const LLZKDialectVersion &other) const {
  if (auto cmp = majorVersion <=> other.majorVersion; cmp != 0) {
    return cmp;
  }
  if (auto cmp = minorVersion <=> other.minorVersion; cmp != 0) {
    return cmp;
  }
  return patchVersion <=> other.patchVersion;
}

} // namespace llzk
