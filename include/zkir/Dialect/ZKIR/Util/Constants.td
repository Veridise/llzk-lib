#ifndef ZKIR_ZKIRCONSTANTS_TD
#define ZKIR_ZKIRCONSTANTS_TD

include "mlir/IR/EnumAttr.td"

/// This class can be used to avoid defining duplicate string constants for
/// string values that are used in both tablegen files and C++ files. In
/// tablegen, the value can be accessed via the ".result" field on the defs
/// created from this class. In C++, the value can be accessed via
/// "zkir::stringifyEnum(<cppName>::I)" where the "stringifyEnum" function is
/// generated for each concrete def created from this class.
class GeneratedStrConst<string cppName, string value>
    : StrFunc<value>,
      I32EnumAttr<cppName, "", [I32EnumAttrCase<"I", 0, value>]> {
  let cppNamespace = "zkir";
  let genSpecializedAttr = 0;
}

def InlineIncludesPassName
    : GeneratedStrConst<"InlineIncludesPassName", "zkir-inline-includes">;

def InlineIncludesPassSummary
    : GeneratedStrConst<
          "InlineIncludesPassSummary",
          "Replace all IncludeOp with contents of the referenced file. This "
          "pass must run from the top-level ModuleOp only.">;

#endif // ZKIR_ZKIRCONSTANTS_TD
