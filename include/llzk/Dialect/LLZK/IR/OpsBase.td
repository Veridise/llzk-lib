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
include "llzk/Util/OpHelper.td"

include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"

//===----------------------------------------------------------------------===//
// Custom traits that can be applied to Ops
//===----------------------------------------------------------------------===//

/// Only valid/implemented for StructDefOp. Sets the proper
/// AllowConstraintsAttrs on the functions defined within the StructDefOp.
def InitAllowConstraintsAttrs : NativeOpTrait<"InitAllowConstraintsAttrs">,
                                StructuralOpTrait {
  string cppNamespace = "::llzk";
}

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

//===----------------------------------------------------------------------===//
// Base Op classes
//===----------------------------------------------------------------------===//

// Note: `resultType.builderCall` must not be empty
class BinaryOpBase<Dialect dialect, string mnemonic, Type resultType,
                        list<Trait> traits = []>
    : NaryOpBase<dialect, mnemonic, resultType.builderCall,
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
class UnaryOpBase<Dialect dialect, string mnemonic, Type resultType,
                       list<Trait> traits = []>
    : NaryOpBase<dialect, mnemonic, resultType.builderCall,
                      traits#[Pure, TypeUnifyWithResult<"operand">]> {

  let arguments = (ins VarTypeOr<resultType>:$operand);
  let results = (outs resultType:$result);

  let assemblyFormat = [{
      $operand
      `` custom<InferredOrParsedType>(type($operand), "true")
      attr-dict
  }];
}


class LLZK_Op<string mnemonic, list<Trait> traits = []>
    : Op<LLZKDialect, mnemonic, traits>;

class LLZK_GlobalRefOp<string mnemonic, list<Trait> traits = []>
    : LLZK_Op<mnemonic,
              traits#[DeclareOpInterfaceMethods<GlobalRefOpInterface>,
                      DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let extraClassDeclaration = [{
    /// Gets the definition for the `global` referenced in this op.
    inline ::mlir::FailureOr<SymbolLookupResult<GlobalDefOp>> getGlobalDefOp(::mlir::SymbolTableCollection &tables) {
      return ::llvm::cast<GlobalRefOpInterface>(getOperation()).getGlobalDefOp(tables);
    }
  }];
}

class LLZK_FieldRefOp<string mnemonic, list<Trait> traits = []>
    : LLZK_Op<mnemonic,
              traits#[DeclareOpInterfaceMethods<FieldRefOpInterface>,
                      DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  bit isRead = ?; // read(1) vs write(0) ops

  let extraClassDeclaration = [{
    /// Gets the definition for the `field` referenced in this op.
    inline ::mlir::FailureOr<SymbolLookupResult<FieldDefOp>> getFieldDefOp(::mlir::SymbolTableCollection &tables) {
      return ::llvm::cast<FieldRefOpInterface>(getOperation()).getFieldDefOp(tables);
    }
  }];

  let extraClassDefinition = [{
    /// Return `true` if the op is a read, `false` if it's a write.
    bool $cppClass::isRead() {
      return }]#!if(isRead, "true", "false")#[{;
    }
  }];
}

// Note: `resultType.builderCall` must not be empty
class LLZK_BinaryOpBase<string mnemonic, Type resultType,
                        list<Trait> traits = []>
    : BinaryOpBase<LLZKDialect, mnemonic, resultType, traits>;

// Note: `resultType.builderCall` must not be empty
class LLZK_UnaryOpBase<string mnemonic, Type resultType,
                       list<Trait> traits = []>
    : UnaryOpBase<LLZKDialect, mnemonic, resultType, traits>;

#endif // LLZK_OPS_BASE
