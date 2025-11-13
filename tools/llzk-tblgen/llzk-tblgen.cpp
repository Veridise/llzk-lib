//===-- llzk-tblgen.cpp - LLZK tblgen tool ----------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the main entry point for the llzk-tblgen tool.
/// The tool extends mlir-tblgen with additional C API generators that are
/// registered by the other source files in this directory (OpCAPIGen.cpp,
/// AttrCAPIGen.cpp, TypeCAPIGen.cpp, etc.). These generators provide more
/// comprehensive C API coverage than the default MLIR tablegen tool.
///
//===----------------------------------------------------------------------===//

#include <mlir/Tools/mlir-tblgen/MlirTblgenMain.h>

int main(int argc, char **argv) { return mlir::MlirTblgenMain(argc, argv); }
