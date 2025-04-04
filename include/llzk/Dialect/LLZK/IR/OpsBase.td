//===-- OpsBase.td -----------------------------------------*- tablegen -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

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

/// Verifies that an Op marked with this Trait cannot be used within a
/// "constrain()" function. However, this does not restrict it from being used
/// within a free function since those can be called by a "compute()" function.
/// If an Op must only be used within the immediate body of a "compute()"
/// function, then `InStructFunctionNamed<"::llzk::FUNC_NAME_COMPUTE">` should
/// be used instead.
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

// Implements verification for ops with an affine_map instantiation list. These
// ops are expected to contain the following in their `arguments` list:
//  - VariadicOfVariadic<Index, "mapOpGroupSizes">:$mapOperands
//  - DefaultValuedAttr<DenseI32ArrayAttr, "{}">:$numDimsPerMap
//  - DenseI32ArrayAttr:$mapOpGroupSizes
// Additionally, if the op also has the `AttrSizedOperandSegments` trait, the
// parameter of this trait specifies the index within the `operandSegmentSizes`
// attribute associated with the `$mapOperands` argument, otherwise the
// parameter is ignored. All of these attributes are necessary because MLIR
// stores all operands for an Op in a single list. These attributes specify how
// the list of operands is split into logical pieces for the operand components.
//
// For example, suppose the `CreateArrayOp` is used to create an array with type
//    `!llzk.array<affine_map<(d0)->(d0)>,affine_map<(d0)[s0]->(d0+s0)> x i1>`
//
// 1) `CreateArrayOp` requires the `AttrSizedOperandSegments` trait because it
// defines two variadic arguments: `$elements` and `$mapOperands` (in that
// order). Thus, the `operandSegmentSizes` attribute is automatically defined
// to specify the number of operands that belong to each variadic argument:
//    `operandSegmentSizes = array<i32: COUNT($elements), COUNT($mapOperands)>`
// In the case of `CreateArrayOp`, one of those sizes will always be 0 because
// its assembly format has `$elements` and `$mapOperands` as alternatives. In
// this example, `COUNT($elements) = 0` and `COUNT($mapOperands) = 3` (this is
// the sum of operand count for all affine_map that are used as array dimensions
// in the result array type).
//
// 2) The `$mapOpGroupSizes` attribute groups the `$mapOperands` per affine_map.
// This implies that their sum equals `COUNT($mapOperands)`. In the example, the
// first affine_map has 1 parameter and the second has 2 so:
//     `mapOpGroupSizes = array<i32: 1, 2>`
//
// 3) Finally, the `$numDimsPerMap` attribute splits the `$mapOperands` in each
// group into the dimensional and symbolic inputs for each affine_map.
// Dimensional inputs appear between the () and symbolic inputs appear between
// the []. LLZK mainly uses dimensional inputs and not symbolic inputs but both
// are fully supported. The length of `$numDimsPerMap` must equal the length of
// `$mapOpGroupSizes` and each element of `$numDimsPerMap` must be less than the
// corresponding element of `$mapOpGroupSizes`. In the example, the both
// affine_map instantiations in the array type have 1 dimensional input so:
//     `numDimsPerMap = array<i32: 1, 1>`
//
// It is also recomended to use `custom<AttrDictWithWarnings>(attr-dict)` in the
// assembly format (or the associated parse/print functions directly) to parse
// the attribute dictionary in these ops and present warnings if the
// aforementioned attributes are manually specified.
class VerifySizesForMultiAffineOps<int operandSegmentIndex>
    : ParamNativeOpTrait<"VerifySizesForMultiAffineOps",
                         ""#operandSegmentIndex>,
      StructuralOpTrait {
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
    : TypesMatchWith<rhsSummary#" type matches with "#lhsSummary#" type",
                     lhsArg, rhsArg, transform, "::llzk::typesUnify">;

// Returns success if `elementArg` unifies with the `arrayArg` element type.
class ArrayElemTypeUnifyWith<string arrayArg, string elementArg>
    : TypesUnify<arrayArg, elementArg, arrayArg#" element", elementArg,
                 "::llvm::cast<::llzk::ArrayType>($_self).getElementType()">;

// Identical to `TypesMatchWith` with `rhsArg = result`. This should be used
// instead of `TypesMatchWith` when custom return type inference is necessary
// (via `InferTypeOpAdaptor*`) because MLIR has special handing for
// `TypesMatchWith` that results in "error: redefinition of 'inferReturnTypes'".
class TypeMatchResultWith<string lhsArg, string lhsSummary = lhsArg,
                          string transform,
                          string comparator = "std::equal_to<>()">
    : PredOpTrait<
          "result type matches with "#lhsSummary#" type",
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
          arrayArg, arrayArg#" element",
          "::llvm::cast<::llzk::ArrayType>($_self).getElementType()",
          "::llzk::typesUnify">;

// ArrayElemTypeUnifyWithResult + InferTypeOpAdaptorWithIsCompatible (i.e.
// generate inferReturnTypes() and isCompatibleReturnTypes() functions)
class ArrayTypeElemsUnifyWithResultCustomInfer<string arrayArg>
    : TraitList<[ArrayElemTypeUnifyWithResult<arrayArg>,
                 InferTypeOpAdaptorWithIsCompatible]>;

//===----------------------------------------------------------------------===//
// Base Op classes
//===----------------------------------------------------------------------===//

class LLZK_Op<string mnemonic, list<Trait> traits = []>
    : Op<LLZKDialect, mnemonic, traits>;

class LLZK_GlobalRefOp<string mnemonic, list<Trait> traits = []>
    : LLZK_Op<mnemonic,
              traits#[DeclareOpInterfaceMethods<SymbolUserOpInterface>,
                      DeclareOpInterfaceMethods<GlobalRefOpInterface>]> {
  let extraClassDeclaration = [{
    /// Gets the definition for the `global` referenced in this op.
    inline ::mlir::FailureOr<SymbolLookupResult<GlobalDefOp>> getGlobalDefOp(::mlir::SymbolTableCollection &tables) {
      return ::llvm::cast<GlobalRefOpInterface>(getOperation()).getGlobalDefOp(tables);
    }
  }];
}

class LLZK_FieldRefOp<string mnemonic, list<Trait> traits = []>
    : LLZK_Op<mnemonic,
              traits#[DeclareOpInterfaceMethods<SymbolUserOpInterface>,
                      DeclareOpInterfaceMethods<FieldRefOpInterface>]> {
  let extraClassDeclaration = [{
    /// Gets the definition for the `field` referenced in this op.
    inline ::mlir::FailureOr<SymbolLookupResult<FieldDefOp>> getFieldDefOp(::mlir::SymbolTableCollection &tables) {
      return ::llvm::cast<FieldRefOpInterface>(getOperation()).getFieldDefOp(tables);
    }
  }];
}

class LLZK_NaryOpBase<string mnemonic, string defaultTypeBuilder,
                      list<Trait> traits = []> : LLZK_Op<mnemonic, traits> {
  let extraClassDeclaration = [{
    static ::mlir::ParseResult parseInferredOrParsedType(
        ::mlir::OpAsmParser &parser, ::mlir::Type &opType, bool isFirst
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

    static void printInferredOrParsedType(::mlir::OpAsmPrinter &printer,
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
