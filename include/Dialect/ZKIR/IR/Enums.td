#ifndef ZKIR_ENUMS
#define ZKIR_ENUMS

include "mlir/IR/EnumAttr.td"

/// Single-selection attributes
class ZKIR_I32Enum<string name, string desc, list<I32EnumAttrCase> cases>
    : I32EnumAttr<name, desc, cases> {
  let genSpecializedAttr = 0;
  let cppNamespace = "zkir";
}

/// Multiple selection attributes, i.e. bit flags
class ZKIR_I32BitEnum<string name, string desc, list<BitEnumAttrCaseBase> cases>
    : I32BitEnumAttr<name, desc, cases> {
  let genSpecializedAttr = 0;
  let cppNamespace = "zkir";
}

def ZKIR_TypeModifiers : ZKIR_I32BitEnum<"ModifiersEnum", "Type modifiers", [
  I32BitEnumAttrCaseNone<"TypeModNone", "none">,
  I32BitEnumAttrCaseBit<"TypeModPublic", 1, "pub">,
  I32BitEnumAttrCaseBit<"TypeModConstant", 2, "const">,
  I32BitEnumAttrCaseBit<"TypeModBoolean", 3, "bool">,
]>;

def ZKIR_FeltCmpPredicate
    : ZKIR_I32Enum<"FeltCmpPredicate", "Field element comparison predicate", [
      I32EnumAttrCase<"EQ", 0, "eq">,
      I32EnumAttrCase<"NE", 1, "ne">,
      I32EnumAttrCase<"LT", 2, "lt">,
      I32EnumAttrCase<"LE", 3, "le">,
      I32EnumAttrCase<"GT", 4, "gt">,
      I32EnumAttrCase<"GE", 5, "ge">,
    ]>;

#endif // ZKIR_ENUMS
