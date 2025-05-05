/*
 @licstart  The following is the entire license notice for the JavaScript code in this file.

 The MIT License (MIT)

 Copyright (C) 1997-2020 by Dimitri van Heesch

 Permission is hereby granted, free of charge, to any person obtaining a copy of this software
 and associated documentation files (the "Software"), to deal in the Software without restriction,
 including without limitation the rights to use, copy, modify, merge, publish, distribute,
 sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all copies or
 substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
 BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
 DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

 @licend  The above is the entire license notice for the JavaScript code in this file
*/
var NAVTREE =
[
  [ "LLZK", "index.html", [
    [ "Overview", "index.html", "index" ],
    [ "What is LLZK?", "overview.html", [
      [ "Project Overview", "overview.html#autotoc_md0", [
        [ "Frontends", "overview.html#frontends", null ],
        [ "Passes", "overview.html#pass-overview", null ],
        [ "Backends", "overview.html#backends", null ]
      ] ]
    ] ],
    [ "Setup", "setup.html", [
      [ "Nix Setup", "setup.html#autotoc_md1", null ],
      [ "Manual Build Setup", "setup.html#autotoc_md2", null ],
      [ "Development Workflow", "setup.html#dev-workflow", null ]
    ] ],
    [ "Tool Guides", "tools.html", [
      [ "llzk-opt", "tools.html#llzk-opt", [
        [ "LLZK Pass Documentation", "tools.html#passes", [
          [ "Analysis Passes", "tools.html#autotoc_md4", [
            [ "-llzk-print-call-graph", "tools.html#autotoc_md5", null ],
            [ "-llzk-print-call-graph-sccs", "tools.html#autotoc_md6", null ],
            [ "-llzk-print-constraint-dependency-graphs", "tools.html#autotoc_md7", null ],
            [ "-llzk-print-interval-analysis", "tools.html#autotoc_md8", [
              [ "Options", "tools.html#autotoc_md9", null ]
            ] ]
          ] ],
          [ "General Transformation Passes", "tools.html#autotoc_md10", [
            [ "-llzk-duplicate-op-elim", "tools.html#autotoc_md11", null ],
            [ "-llzk-duplicate-read-write-elim", "tools.html#autotoc_md12", null ],
            [ "-llzk-unused-declaration-elim", "tools.html#autotoc_md13", [
              [ "Options", "tools.html#autotoc_md14", null ]
            ] ]
          ] ],
          [ "'array' Dialect Transformation Passes", "tools.html#autotoc_md15", [
            [ "-llzk-array-to-scalar", "tools.html#autotoc_md16", null ]
          ] ],
          [ "'polymorphic' Dialect Transformation Passes", "tools.html#autotoc_md17", [
            [ "-llzk-flatten", "tools.html#autotoc_md18", null ]
          ] ],
          [ "Validation Passes", "tools.html#autotoc_md19", [
            [ "-llzk-validate-field-writes", "tools.html#autotoc_md20", null ]
          ] ]
        ] ]
      ] ],
      [ "llzk-lsp-server", "tools.html#autotoc_md21", null ]
    ] ],
    [ "LLZK Language Specification", "syntax.html", [
      [ "Syntax", "syntax.html#autotoc_md22", null ],
      [ "Types", "syntax.html#autotoc_md23", null ],
      [ "Special Constructs", "syntax.html#autotoc_md24", null ],
      [ "Semantic Rules", "syntax.html#autotoc_md25", null ],
      [ "Translation Guidelines", "syntax.html#translation-guidelines", null ]
    ] ],
    [ "Contribution Guide", "contribution-guide.html", "contribution-guide" ],
    [ "LLZK Dialects", "dialects.html", [
      [ "'array' Dialect", "dialects.html#autotoc_md45", [
        [ "Types", "dialects.html#autotoc_md46", [
          [ "ArrayType", "dialects.html#autotoc_md47", [
            [ "Parameters:", "dialects.html#autotoc_md48", null ]
          ] ]
        ] ]
      ] ],
      [ "'bool' Dialect", "dialects.html#autotoc_md49", [
        [ "Operations", "dialects.html#autotoc_md50", [
          [ "bool.and (llzk::boolean::AndBoolOp)", "dialects.html#autotoc_md51", [
            [ "Operands:", "dialects.html#autotoc_md52", null ],
            [ "Results:", "dialects.html#autotoc_md53", null ]
          ] ],
          [ "bool.assert (llzk::boolean::AssertOp)", "dialects.html#autotoc_md54", [
            [ "Attributes:", "dialects.html#autotoc_md55", null ],
            [ "Operands:", "dialects.html#autotoc_md56", null ]
          ] ],
          [ "bool.cmp (llzk::boolean::CmpOp)", "dialects.html#autotoc_md57", [
            [ "Attributes:", "dialects.html#autotoc_md58", null ],
            [ "Operands:", "dialects.html#autotoc_md59", null ],
            [ "Results:", "dialects.html#autotoc_md60", null ]
          ] ],
          [ "bool.not (llzk::boolean::NotBoolOp)", "dialects.html#autotoc_md61", [
            [ "Operands:", "dialects.html#autotoc_md62", null ],
            [ "Results:", "dialects.html#autotoc_md63", null ]
          ] ],
          [ "bool.or (llzk::boolean::OrBoolOp)", "dialects.html#autotoc_md64", [
            [ "Operands:", "dialects.html#autotoc_md65", null ],
            [ "Results:", "dialects.html#autotoc_md66", null ]
          ] ],
          [ "bool.xor (llzk::boolean::XorBoolOp)", "dialects.html#autotoc_md67", [
            [ "Operands:", "dialects.html#autotoc_md68", null ],
            [ "Results:", "dialects.html#autotoc_md69", null ]
          ] ]
        ] ],
        [ "Attributes", "dialects.html#autotoc_md70", [
          [ "FeltCmpPredicateAttr", "dialects.html#autotoc_md71", [
            [ "Parameters:", "dialects.html#autotoc_md72", null ]
          ] ]
        ] ]
      ] ],
      [ "'cast' Dialect", "dialects.html#autotoc_md73", [
        [ "Operations", "dialects.html#autotoc_md74", [
          [ "cast.tofelt (llzk::cast::IntToFeltOp)", "dialects.html#autotoc_md75", [
            [ "Operands:", "dialects.html#autotoc_md76", null ],
            [ "Results:", "dialects.html#autotoc_md77", null ]
          ] ],
          [ "cast.toindex (llzk::cast::FeltToIndexOp)", "dialects.html#autotoc_md78", [
            [ "Operands:", "dialects.html#autotoc_md79", null ],
            [ "Results:", "dialects.html#autotoc_md80", null ]
          ] ]
        ] ]
      ] ],
      [ "'constrain' Dialect", "dialects.html#autotoc_md81", [
        [ "Operations", "dialects.html#autotoc_md82", [
          [ "constrain.eq (llzk::constrain::EmitEqualityOp)", "dialects.html#autotoc_md83", [
            [ "Operands:", "dialects.html#autotoc_md84", null ]
          ] ],
          [ "constrain.in (llzk::constrain::EmitContainmentOp)", "dialects.html#autotoc_md85", [
            [ "Operands:", "dialects.html#autotoc_md86", null ]
          ] ]
        ] ]
      ] ],
      [ "'felt' Dialect", "dialects.html#autotoc_md87", [
        [ "Operations", "dialects.html#autotoc_md88", [
          [ "felt.add (llzk::felt::AddFeltOp)", "dialects.html#autotoc_md89", [
            [ "Operands:", "dialects.html#autotoc_md90", null ],
            [ "Results:", "dialects.html#autotoc_md91", null ]
          ] ],
          [ "felt.bit_and (llzk::felt::AndFeltOp)", "dialects.html#autotoc_md92", [
            [ "Operands:", "dialects.html#autotoc_md93", null ],
            [ "Results:", "dialects.html#autotoc_md94", null ]
          ] ],
          [ "felt.bit_not (llzk::felt::NotFeltOp)", "dialects.html#autotoc_md95", [
            [ "Operands:", "dialects.html#autotoc_md96", null ],
            [ "Results:", "dialects.html#autotoc_md97", null ]
          ] ],
          [ "felt.bit_or (llzk::felt::OrFeltOp)", "dialects.html#autotoc_md98", [
            [ "Operands:", "dialects.html#autotoc_md99", null ],
            [ "Results:", "dialects.html#autotoc_md100", null ]
          ] ],
          [ "felt.bit_xor (llzk::felt::XorFeltOp)", "dialects.html#autotoc_md101", [
            [ "Operands:", "dialects.html#autotoc_md102", null ],
            [ "Results:", "dialects.html#autotoc_md103", null ]
          ] ],
          [ "felt.const (llzk::felt::FeltConstantOp)", "dialects.html#autotoc_md104", [
            [ "Attributes:", "dialects.html#autotoc_md105", null ],
            [ "Results:", "dialects.html#autotoc_md106", null ]
          ] ],
          [ "felt.div (llzk::felt::DivFeltOp)", "dialects.html#autotoc_md107", [
            [ "Operands:", "dialects.html#autotoc_md108", null ],
            [ "Results:", "dialects.html#autotoc_md109", null ]
          ] ],
          [ "felt.inv (llzk::felt::InvFeltOp)", "dialects.html#autotoc_md110", [
            [ "Operands:", "dialects.html#autotoc_md111", null ],
            [ "Results:", "dialects.html#autotoc_md112", null ]
          ] ],
          [ "felt.mod (llzk::felt::ModFeltOp)", "dialects.html#autotoc_md113", [
            [ "Operands:", "dialects.html#autotoc_md114", null ],
            [ "Results:", "dialects.html#autotoc_md115", null ]
          ] ],
          [ "felt.mul (llzk::felt::MulFeltOp)", "dialects.html#autotoc_md116", [
            [ "Operands:", "dialects.html#autotoc_md117", null ],
            [ "Results:", "dialects.html#autotoc_md118", null ]
          ] ],
          [ "felt.neg (llzk::felt::NegFeltOp)", "dialects.html#autotoc_md119", [
            [ "Operands:", "dialects.html#autotoc_md120", null ],
            [ "Results:", "dialects.html#autotoc_md121", null ]
          ] ],
          [ "felt.nondet (llzk::felt::FeltNonDetOp)", "dialects.html#autotoc_md122", [
            [ "Results:", "dialects.html#autotoc_md123", null ]
          ] ],
          [ "felt.shl (llzk::felt::ShlFeltOp)", "dialects.html#autotoc_md124", [
            [ "Operands:", "dialects.html#autotoc_md125", null ],
            [ "Results:", "dialects.html#autotoc_md126", null ]
          ] ],
          [ "felt.shr (llzk::felt::ShrFeltOp)", "dialects.html#autotoc_md127", [
            [ "Operands:", "dialects.html#autotoc_md128", null ],
            [ "Results:", "dialects.html#autotoc_md129", null ]
          ] ],
          [ "felt.sub (llzk::felt::SubFeltOp)", "dialects.html#autotoc_md130", [
            [ "Operands:", "dialects.html#autotoc_md131", null ],
            [ "Results:", "dialects.html#autotoc_md132", null ]
          ] ]
        ] ],
        [ "Attributes", "dialects.html#autotoc_md133", [
          [ "FeltConstAttr", "dialects.html#autotoc_md134", [
            [ "Parameters:", "dialects.html#autotoc_md135", null ]
          ] ]
        ] ],
        [ "Types", "dialects.html#autotoc_md136", [
          [ "FeltType", "dialects.html#autotoc_md137", null ]
        ] ]
      ] ],
      [ "'function' Dialect", "dialects.html#autotoc_md138", [
        [ "Operations", "dialects.html#autotoc_md139", [
          [ "function.call (llzk::function::CallOp)", "dialects.html#autotoc_md140", [
            [ "Attributes:", "dialects.html#autotoc_md141", null ],
            [ "Operands:", "dialects.html#autotoc_md142", null ],
            [ "Results:", "dialects.html#autotoc_md143", null ]
          ] ],
          [ "function.def (llzk::function::FuncDefOp)", "dialects.html#autotoc_md144", [
            [ "Attributes:", "dialects.html#autotoc_md145", null ]
          ] ],
          [ "function.return (llzk::function::ReturnOp)", "dialects.html#autotoc_md146", [
            [ "Operands:", "dialects.html#autotoc_md147", null ]
          ] ]
        ] ]
      ] ],
      [ "'global' Dialect", "dialects.html#autotoc_md148", [
        [ "Operations", "dialects.html#autotoc_md149", [
          [ "global.def (llzk::global::GlobalDefOp)", "dialects.html#autotoc_md150", [
            [ "Attributes:", "dialects.html#autotoc_md151", null ]
          ] ],
          [ "global.read (llzk::global::GlobalReadOp)", "dialects.html#autotoc_md152", [
            [ "Attributes:", "dialects.html#autotoc_md153", null ],
            [ "Results:", "dialects.html#autotoc_md154", null ]
          ] ],
          [ "global.write (llzk::global::GlobalWriteOp)", "dialects.html#autotoc_md155", [
            [ "Attributes:", "dialects.html#autotoc_md156", null ],
            [ "Operands:", "dialects.html#autotoc_md157", null ]
          ] ]
        ] ]
      ] ],
      [ "'include' Dialect", "dialects.html#autotoc_md158", [
        [ "Operations", "dialects.html#autotoc_md159", [
          [ "include.from (llzk::include::IncludeOp)", "dialects.html#autotoc_md160", [
            [ "Attributes:", "dialects.html#autotoc_md161", null ]
          ] ]
        ] ]
      ] ],
      [ "'llzk' Dialect", "dialects.html#autotoc_md162", [
        [ "Attributes", "dialects.html#autotoc_md163", [
          [ "LoopBoundsAttr", "dialects.html#autotoc_md164", [
            [ "Parameters:", "dialects.html#autotoc_md165", null ]
          ] ],
          [ "PublicAttr", "dialects.html#autotoc_md166", null ]
        ] ]
      ] ],
      [ "'poly' Dialect", "dialects.html#autotoc_md167", [
        [ "Operations", "dialects.html#autotoc_md168", [
          [ "poly.applymap (llzk::polymorphic::ApplyMapOp)", "dialects.html#autotoc_md169", [
            [ "Attributes:", "dialects.html#autotoc_md170", null ],
            [ "Operands:", "dialects.html#autotoc_md171", null ],
            [ "Results:", "dialects.html#autotoc_md172", null ]
          ] ],
          [ "poly.read_const (llzk::polymorphic::ConstReadOp)", "dialects.html#autotoc_md173", [
            [ "Attributes:", "dialects.html#autotoc_md174", null ],
            [ "Results:", "dialects.html#autotoc_md175", null ]
          ] ],
          [ "poly.unifiable_cast (llzk::polymorphic::UnifiableCastOp)", "dialects.html#autotoc_md176", [
            [ "Operands:", "dialects.html#autotoc_md177", null ],
            [ "Results:", "dialects.html#autotoc_md178", null ]
          ] ]
        ] ],
        [ "Types", "dialects.html#autotoc_md179", [
          [ "TypeVarType", "dialects.html#autotoc_md180", [
            [ "Parameters:", "dialects.html#autotoc_md181", null ]
          ] ]
        ] ]
      ] ],
      [ "'string' Dialect", "dialects.html#autotoc_md182", [
        [ "Operations", "dialects.html#autotoc_md183", [
          [ "string.new (llzk::string::LitStringOp)", "dialects.html#autotoc_md184", [
            [ "Attributes:", "dialects.html#autotoc_md185", null ],
            [ "Results:", "dialects.html#autotoc_md186", null ]
          ] ]
        ] ],
        [ "Types", "dialects.html#autotoc_md187", [
          [ "StringType", "dialects.html#autotoc_md188", null ]
        ] ]
      ] ],
      [ "'struct' Dialect", "dialects.html#autotoc_md189", [
        [ "Operations", "dialects.html#autotoc_md190", [
          [ "struct.def (llzk::component::StructDefOp)", "dialects.html#autotoc_md191", [
            [ "Attributes:", "dialects.html#autotoc_md192", null ]
          ] ],
          [ "struct.field (llzk::component::FieldDefOp)", "dialects.html#autotoc_md193", [
            [ "Attributes:", "dialects.html#autotoc_md194", null ]
          ] ],
          [ "struct.new (llzk::component::CreateStructOp)", "dialects.html#autotoc_md195", [
            [ "Results:", "dialects.html#autotoc_md196", null ]
          ] ],
          [ "struct.readf (llzk::component::FieldReadOp)", "dialects.html#autotoc_md197", [
            [ "Attributes:", "dialects.html#autotoc_md198", null ],
            [ "Operands:", "dialects.html#autotoc_md199", null ],
            [ "Results:", "dialects.html#autotoc_md200", null ]
          ] ],
          [ "struct.writef (llzk::component::FieldWriteOp)", "dialects.html#autotoc_md201", [
            [ "Attributes:", "dialects.html#autotoc_md202", null ],
            [ "Operands:", "dialects.html#autotoc_md203", null ]
          ] ]
        ] ],
        [ "Types", "dialects.html#autotoc_md204", [
          [ "StructType", "dialects.html#autotoc_md205", [
            [ "Parameters:", "dialects.html#autotoc_md206", null ]
          ] ]
        ] ]
      ] ],
      [ "'undef' Dialect", "dialects.html#autotoc_md207", [
        [ "Operations", "dialects.html#autotoc_md208", [
          [ "undef.undef (llzk::undef::UndefOp)", "dialects.html#autotoc_md209", [
            [ "Results:", "dialects.html#autotoc_md210", null ]
          ] ]
        ] ]
      ] ]
    ] ],
    [ "Namespaces", "namespaces.html", [
      [ "Namespace List", "namespaces.html", "namespaces_dup" ],
      [ "Namespace Members", "namespacemembers.html", [
        [ "All", "namespacemembers.html", "namespacemembers_dup" ],
        [ "Functions", "namespacemembers_func.html", "namespacemembers_func" ],
        [ "Variables", "namespacemembers_vars.html", null ],
        [ "Typedefs", "namespacemembers_type.html", null ],
        [ "Enumerations", "namespacemembers_enum.html", null ]
      ] ]
    ] ],
    [ "Concepts", "concepts.html", "concepts" ],
    [ "Classes", "annotated.html", [
      [ "Class List", "annotated.html", "annotated_dup" ],
      [ "Class Index", "classes.html", null ],
      [ "Class Hierarchy", "hierarchy.html", "hierarchy" ],
      [ "Class Members", "functions.html", [
        [ "All", "functions.html", "functions_dup" ],
        [ "Functions", "functions_func.html", "functions_func" ],
        [ "Variables", "functions_vars.html", "functions_vars" ],
        [ "Typedefs", "functions_type.html", "functions_type" ],
        [ "Enumerations", "functions_enum.html", null ],
        [ "Related Symbols", "functions_rela.html", null ]
      ] ]
    ] ],
    [ "Files", "files.html", [
      [ "File List", "files.html", "files_dup" ],
      [ "File Members", "globals.html", [
        [ "All", "globals.html", "globals_dup" ],
        [ "Functions", "globals_func.html", null ],
        [ "Variables", "globals_vars.html", null ],
        [ "Macros", "globals_defs.html", null ]
      ] ]
    ] ]
  ] ]
];

var NAVTREEINDEX =
[
"_a_p_int_helper_8cpp.html",
"_felt_2_i_r_2_dialect_8h_8inc.html",
"_polymorphic_2_transforms_2_transformation_passes_8h.html#a50fd6f4ec277edd1b62f2efe4a35eede",
"classllzk_1_1_constrain_ref.html#a14c3650d390437a49268cc7b810f7c0b",
"classllzk_1_1_expression_value.html#ad140e3732fab6c5e228379a9856c6dbc",
"classllzk_1_1_module_builder.html#aa2a7258718c32845d2bcbafe43e03f79",
"classllzk_1_1array_1_1_array_index_gen.html#a6b8e07a161e4089149fc9fba13f88978",
"classllzk_1_1array_1_1_extract_array_op.html#a6989a6eb9ac668108ed6e438f7b05968",
"classllzk_1_1array_1_1_write_array_op.html#a9a22ddc3a4286d0c16c196f3e80551e0",
"classllzk_1_1boolean_1_1_and_bool_op_adaptor.html#a0e23192e825e37defebb0ae33b774641",
"classllzk_1_1boolean_1_1_not_bool_op_generic_adaptor.html#ae3f6e6f1a076812d1dce362806e4082a",
"classllzk_1_1cast_1_1_cast_dialect.html#a4f25a4f77bb6ce6214612acc1e0170ac",
"classllzk_1_1component_1_1_field_def_op.html#a997d4dcbc0f08a20b4c299d9384f60d2",
"classllzk_1_1component_1_1_field_write_op_generic_adaptor.html",
"classllzk_1_1component_1_1detail_1_1_field_write_op_generic_adaptor_base.html",
"classllzk_1_1dataflow_1_1_abstract_dense_forward_data_flow_analysis.html#a9c8420d402c7e59d9070020c0817c7ef",
"classllzk_1_1felt_1_1_div_felt_op.html#aaa0799756f9144af230849a267c2109a",
"classllzk_1_1felt_1_1_mod_felt_op.html#a2af1431be31c2e5f09c90aa847c616ab",
"classllzk_1_1felt_1_1_or_felt_op.html#a1d7dae2c3e4e81c6f8fdd1b80db2aceb",
"classllzk_1_1felt_1_1_sub_felt_op_generic_adaptor.html#a86edec80ed272d4fe980558515aecf08",
"classllzk_1_1felt_1_1detail_1_1_or_felt_op_generic_adaptor_base.html#a5bbbf6ac1b6593d46f4ab8a3b31262f4",
"classllzk_1_1function_1_1_func_def_op.html#a5e340f3ca8deb3c212244693b9c115be",
"classllzk_1_1global_1_1_global_def_op.html#a0d5cc21e0ad2fe2a1126be3e8cb25603",
"classllzk_1_1global_1_1detail_1_1_global_def_op_generic_adaptor_base.html#a047c4023a9b7861261fb040dcb4246f3",
"classllzk_1_1impl_1_1_unused_declaration_elimination_pass_base.html#ac85f58efe907e43fe2ab155e750aa9b4",
"classllzk_1_1polymorphic_1_1_const_read_op.html#aa120b8527bb41fed66aab9446869cb3c",
"classllzk_1_1string_1_1_lit_string_op.html#afb9b2d00f3f7b872c1691396ec798691",
"dialects.html#autotoc_md178",
"functions_j.html",
"namespacellzk.html#ab3aea79f1ec3c694d5d0bcf1f949c5fe",
"structllvm_1_1_dense_map_info_3_1_1llzk_1_1boolean_1_1_felt_cmp_predicate_01_4.html#a6650f4630f2296703915536dff0dc203",
"structllzk_1_1component_1_1detail_1_1_field_read_op_generic_adaptor_base_1_1_properties.html#a63d935ae2e006e051462f7047cb0a73b",
"structllzk_1_1global_1_1detail_1_1_global_ref_op_interface_interface_traits_1_1_concept.html#ab0fef470c316c846f32e0c2c8580b2f8"
];

var SYNCONMSG = 'click to disable panel synchronization';
var SYNCOFFMSG = 'click to enable panel synchronization';