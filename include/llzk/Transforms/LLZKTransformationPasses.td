//===-- LLZKTransformationPasses.td ------------------------*- tablegen -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_TRANSFORMATION_PASSES_TD
#define LLZK_TRANSFORMATION_PASSES_TD

include "llzk/Pass/PassBase.td"

def RedundantReadAndWriteEliminationPass
    : LLZKPass<"llzk-duplicate-read-write-elim"> {
  let summary = "Remove redundant reads and writes";
  let description = [{
    Remove read and write operations to struct fields and arrays that are redundant or unnecessary.
  }];
  let constructor = "llzk::createRedundantReadAndWriteEliminationPass()";
}

def RedundantOperationEliminationPass : LLZKPass<"llzk-duplicate-op-elim"> {
  let summary = "Remove redundant operations";
  let description = [{
    Remove llzk and arith dialect operations that produce the same results
    as previously executed operations.

    Pass should be run after llzk-duplicate-read-write-elim for maximum effect.
  }];
  let constructor = "llzk::createRedundantOperationEliminationPass()";
}

def UnusedDeclarationEliminationPass
    : LLZKPass<"llzk-unused-declaration-elim"> {
  let summary = "Remove unused field and struct declarations";
  let description = [{
    Remove field and struct declarations that are unused within the current compilation
    unit. Note that this pass may cause linking issues with external modules that
    depend on any unused field and struct declarations from this compilation unit.

    Pass should be run after llzk-duplicate-read-write-elim
    and llzk-duplicate-op-elim for maximum effect.
  }];
  let constructor = "llzk::createUnusedDeclarationEliminationPass()";
  let options = [Option<"removeStructs", "remove-structs", "bool",
                        /* default */ "false",
                        "Whether to remove unused struct definitions as well. "
                        "Requires module to declare a Main component, "
                        "otherwise all components will appear unused.">,
  ];
}

def PolyLoweringPass
    : LLZKPass<"llzk-poly-lowering-pass"> {
  let summary = "Lowers the degree of all polynomial equations to a specified maximum";
  let description = [{
    Remove field and struct declarations that are unused within the current compilation
    unit. Note that this pass may cause linking issues with external modules that
    depend on any unused field and struct declarations from this compilation unit.

    Pass should be run after llzk-duplicate-read-write-elim
    and llzk-duplicate-op-elim for maximum effect.
  }];
  let constructor = "llzk::createPolyLoweringPass()";
  let options = [Option<"maxDegree", "lower-poly", "unsigned",
                        /* default */ "2",
                        "Rewrite constraints to equisatisfiable system "
                        "where the maximum degree of any polynomial"
                        "is less than or equal to the specified degree. ">,
  ];
}

#endif // LLZK_TRANSFORMATION_PASSES_TD
