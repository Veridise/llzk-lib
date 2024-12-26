#ifndef LLZK_OPS_BASE
#define LLZK_OPS_BASE

include "llzk/Dialect/LLZK/IR/Types.td"
include "llzk/Dialect/LLZK/IR/OpInterfaces.td"

include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"

//===----------------------------------------------------------------------===//
// Custom traits that can be applied to Ops
//===----------------------------------------------------------------------===//

def ComputeOnly : NativeOpTrait<"ComputeOnly">, StructuralOpTrait {
  string cppNamespace = "::llzk";
}

def InStruct : NativeOpTrait<"InStruct">, StructuralOpTrait {
  string cppNamespace = "::llzk";
}

class InStructFunctionNamed<string name>
    : ParamNativeOpTrait<"InStructFunctionNamed", name>, StructuralOpTrait {
  string cppNamespace = "::llzk";
}

// Implementation of TypesMatchWith for Variadic `rhsArg` that returns success
// if `rhsArg` is empty.
class VariadicTypesMatchWith<string summary, string lhsArg, string rhsArg,
                             string transform,
                             string comparator = "std::equal_to<>()">
    : TypesMatchWith<
          summary, lhsArg, rhsArg, transform,
          "get"#snakeCaseToCamelCase<rhsArg>.ret#"().empty() || "#comparator>;

// Type constraint `llzk::typesUnify(transform(lhs.getType()), rhs.getType())`.
// If either parameter is `$result` it is recommended to use TypeUnifyWithResult
// instead as this is likely too restrictive when type variables are involved.
class TypesUnify<string lhsArg, string rhsArg, string lhsSummary = lhsArg,
                 string rhsSummary = rhsArg, string transform = "$_self">
    : TypesMatchWith<rhsSummary#" type matches with "#lhsSummary, lhsArg,
                     rhsArg, transform, "::llzk::typesUnify">;

// Returns success if `elementArg` unifies with the `arrayArg` element type.
class ArrayElemTypeUnifyWith<string arrayArg, string elementArg>
    : TypesUnify<arrayArg, elementArg, arrayArg#" element type", elementArg,
                 "::llvm::cast<::llzk::ArrayType>($_self).getElementType()">;

// Identical to `TypesMatchWith` with `rhsArg = result`. This should be used
// instead of `TypesMatchWith` when custom return type inference is necessary
// (via `InferTypeOpAdaptor*`) because MLIR has special handing for
// `TypesMatchWith` that results in "error: redefinition of 'inferReturnTypes'".
class TypeMatchResultWith<string lhsArg, string lhsSummary = lhsArg,
                          string transform,
                          string comparator = "std::equal_to<>()">
    : PredOpTrait<
          "result type matches with "#lhsSummary,
          CPred<comparator#"("#!subst("$_self", "$"#lhsArg#".getType()",
                                      transform)#", $result.getType())">> {
  string lhs = lhsArg;
  string rhs = "result";
  string transformer = transform;
}

// Like TypesUnify with `rhsArg = "result"`
class TypeUnifyWithResult<string lhsArg, string lhsSummary = lhsArg,
                          string transform = "$_self">
    : TypeMatchResultWith<lhsArg, lhsSummary, transform, "::llzk::typesUnify">;

// Returns success if `$result` unifies with the `arrayArg` element type.
class ArrayElemTypeUnifyWithResult<string arrayArg>
    : TypeMatchResultWith<
          arrayArg, arrayArg#" element type",
          "::llvm::cast<::llzk::ArrayType>($_self).getElementType()",
          "::llzk::typesUnify">;

// ArrayElemTypeUnifyWithResult + InferTypeOpAdaptorWithIsCompatible
class ArrayTypeElemsUnifyWithResultCustomInfer<string arrayArg>
    : TraitList<[ArrayElemTypeUnifyWithResult<arrayArg>,
                 InferTypeOpAdaptorWithIsCompatible]>;

//===----------------------------------------------------------------------===//
// Base Op classes
//===----------------------------------------------------------------------===//

class LLZK_Op<string mnemonic, list<Trait> traits = []>
    : Op<LLZKDialect, mnemonic, traits>;

class LLZK_FieldRefOp<string mnemonic, list<Trait> traits = []>
    : LLZK_Op<mnemonic,
              traits#[DeclareOpInterfaceMethods<SymbolUserOpInterface>,
                      DeclareOpInterfaceMethods<FieldRefOpInterface>]> {
  let extraClassDeclaration = [{
    /// Gets the `field` definition for the field referenced in this op.
    ::mlir::FailureOr<SymbolLookupResult<FieldDefOp>> getFieldDefOp(::mlir::SymbolTableCollection &symbolTable);
  }];
}

class LLZK_NaryOpBase<string mnemonic, string defaultTypeBuilder,
                      list<Trait> traits = []> : LLZK_Op<mnemonic, traits> {
  let extraClassDeclaration = [{
    static ::mlir::ParseResult parseInferredOrParsedType(
        ::mlir::AsmParser &parser, ::mlir::Type &opType, bool isFirst
    ) {
      if (mlir::succeeded(isFirst ? parser.parseOptionalColon() : parser.parseOptionalComma())) {
        // If there is a comma, parse the `opType`
        mlir::Type type;
        if (parser.parseCustomTypeWithFallback(type)) {
          return mlir::failure();
        }
        opType = type;
      } else {
        // Otherwise, build the default type
        opType = 
        }]#!subst("$_builder", "parser.getBuilder()", defaultTypeBuilder)#[{;
      }
      return mlir::success();
    }

    static void printInferredOrParsedType(::mlir::AsmPrinter &printer,
        ::mlir::Operation *op, ::mlir::Type opType, bool isFirst
    ) {
      printer << (isFirst ? " : " : ", ");
      printer.printStrippedAttrOrType(opType);
    }
  }];
}

// Note: `resultType.builderCall` must not be empty
class LLZK_BinaryOpBase<string mnemonic, Type resultType,
                        list<Trait> traits = []>
    : LLZK_NaryOpBase<mnemonic, resultType.builderCall,
                      traits#[Pure, TypeUnifyWithResult<"lhs">,
                              TypeUnifyWithResult<"rhs">]> {

  let arguments = (ins VarTypeOr<resultType>:$lhs, VarTypeOr<resultType>:$rhs);
  let results = (outs resultType:$result);

  let assemblyFormat = [{
      $lhs `,` $rhs
      `` custom<InferredOrParsedType>(type($lhs), "true") 
      `` custom<InferredOrParsedType>(type($rhs), "false") 
      attr-dict
  }];
}

// Note: `resultType.builderCall` must not be empty
class LLZK_UnaryOpBase<string mnemonic, Type resultType,
                       list<Trait> traits = []>
    : LLZK_NaryOpBase<mnemonic, resultType.builderCall,
                      traits#[Pure, TypeUnifyWithResult<"operand">]> {

  let arguments = (ins VarTypeOr<resultType>:$operand);
  let results = (outs resultType:$result);

  let assemblyFormat = [{
      $operand
      `` custom<InferredOrParsedType>(type($operand), "true") 
      attr-dict
  }];
}

#endif // LLZK_OPS_BASE
