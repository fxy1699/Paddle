// Copyright (c) 2024 CINN Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
#include "paddle/cinn/ir/ir.h"

namespace cinn {
namespace optim {

/**
 *
 * This pass ensures that operations using `int64_t` in IR are cast down to
 * `int32_t` where overflow is not a risk.
 *
 * This pass is applicable in scenarios where computations in a loop or tensor
 * manipulation rely on 64-bit integer types (`int64_t`), but the operations are
 * safe within 32-bit integer limits (`int32_t`). It is especially useful in
 * architectures or compilers where using 32-bit integers yields better
 * performance or reduces memory overhead. For example, large-scale tensor
 * computations or loop iterations that would safely fit within the range of a
 * 32-bit integer can benefit from this pass.
 *
 * When applied, this pass modifies the IR by:
 * - Casting loop bounds (`min` and `extent`) and variable bounds from `int64_t`
 * to `int32_t`.
 * - Adjusting tensor shape dimensions and buffer metadata from `int64_t` to
 * `int32_t`.
 * - Ensuring all index expressions used in `Load` and `Store` operations are
 * converted to `int32_t`.
 * - Traversing schedule blocks and their variables, converting types from
 * `int64_t` to `int32_t` where applicable.
 *
 * Performance impact: This pass addresses potential inefficiencies due to the
 * use of 64-bit integers on architectures optimized for 32-bit operations. It
 * can reduce memory usage and improve computation speed by minimizing the size
 * of integers processed in critical sections like loops or tensor access
 * patterns.
 *
 */

// Try to change the type of longlong to int in the expr.
void TryCastLonglong2Int(Expr* expr);
}  // namespace optim
}  // namespace cinn
