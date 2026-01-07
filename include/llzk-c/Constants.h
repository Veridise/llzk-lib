//===-- Constants.h - LLZK constants ------------------------------*- C -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// This header declares string constants used by LLZK. The actual values are defined in
// llzk/Utils/Constants.h
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_C_CONSTANTS_H
#define LLZK_C_CONSTANTS_H

#ifdef __cplusplus
extern "C" {
#endif

/// Symbol name for the main entry point struct/component (if any). There are additional
/// restrictions on the struct with this name:
/// 1. It cannot have struct parameters.
/// 2. The parameter types of its functions (besides the required "self" parameter) can
///     only be `felt` or `array<.. x felt>`.
extern const char *LLZK_COMPONENT_NAME_MAIN;

/// Symbol name for the witness generation (and resp. constraint generation) functions within a
/// component.
extern const char *LLZK_FUNC_NAME_COMPUTE;
extern const char *LLZK_FUNC_NAME_CONSTRAIN;

/// Name of the attribute on the top-level ModuleOp that specifies the IR language name.
extern const char *LLZK_LANG_ATTR_NAME;

#ifdef __cplusplus
}
#endif

#endif
