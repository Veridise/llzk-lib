//===-- OpHelper.td ----------------------------------------*- tablegen -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_OP_HELPER
#define LLZK_OP_HELPER

include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

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
//    `!array.array<affine_map<(d0)->(d0)>,affine_map<(d0)[s0]->(d0+s0)> x i1>`
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
    : TypesUnify<
          arrayArg, elementArg, arrayArg#" element", elementArg,
          "::llvm::cast<::llzk::array::ArrayType>($_self).getElementType()">;

// Returns success if `$result` unifies with the `arrayArg` element type.
class ArrayElemTypeUnifyWithResult<string arrayArg>
    : TypeMatchResultWith<
          arrayArg, arrayArg#" element",
          "::llvm::cast<::llzk::array::ArrayType>($_self).getElementType()",
          "::llzk::typesUnify">;

// ArrayElemTypeUnifyWithResult + InferTypeOpAdaptorWithIsCompatible (i.e.
// generate inferReturnTypes() and isCompatibleReturnTypes() functions)
class ArrayTypeElemsUnifyWithResultCustomInfer<string arrayArg>
    : TraitList<[ArrayElemTypeUnifyWithResult<arrayArg>,
                 InferTypeOpAdaptorWithIsCompatible]>;

//===------------------------------------------------------------------===//
// Op Classes
//===------------------------------------------------------------------===//

class NaryOpBase<Dialect dialect, string mnemonic, string defaultTypeBuilder,
                      list<Trait> traits = []> : Op<dialect, mnemonic, traits> {
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

#endif // LLZK_OP_HELPER
