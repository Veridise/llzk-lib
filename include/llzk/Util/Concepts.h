//===-- Concepts.h ----------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include <concepts>

/// Restricts a template parameter to `Op` classes that have the given Trait.
template <typename OpT, template <typename> class Trait>
concept HasTrait = OpT::template hasTrait<Trait>();

/// Restricts a template parameter to `Op` classes that implement the given OpInterface.
template <typename OpT, typename Iface>
concept HasInterface = HasTrait<OpT, Iface::template Trait>;
