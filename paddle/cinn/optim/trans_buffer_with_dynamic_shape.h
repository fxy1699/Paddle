// Copyright (c) 2023 CINN Authors. All Rights Reserved.
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
#include <string>

#include "paddle/cinn/ir/ir.h"

namespace cinn {
namespace optim {

/**
 *
 * This pass processes buffers with dynamic shapes to ensure their validity on
 * target hardware (especially GPUs) and checks whether shared memory usage
 * adheres to hardware constraints.
 *
 * This pass is applicable in scenarios where tensors or buffers in the IR have
 * dynamic shapes that need runtime evaluation or simplification, particularly
 * in environments like CUDA GPU computations where shared or local memory has
 * strict size limits. Typical cases include dynamic shape handling in GPU
 * kernels.
 *
 * When applied, this pass makes the following modifications to the IR:
 * - Performs symbolic analysis and simplifies expressions related to tensor or
 * buffer shapes.
 * - Ensures that dynamic shapes can be upper-bounded and verifies that the
 * resulting expressions are constants.
 * - Calculates the size of buffers allocated in shared memory and checks
 * whether the size exceeds the hardware's maximum shared memory capacity.
 * - Throws runtime or compile-time errors if shape expressions cannot be
 * simplified or validated.
 *
 * Performance impact: This pass improves program stability and execution
 * efficiency in dynamic shape scenarios by ensuring shape expressions are valid
 * and shared memory allocation is reasonable, avoiding runtime crashes or
 * inefficiencies.
 *
 */

void CudaTransBufferWithDynamicShape(ir::Expr* expr);

}  // namespace optim
}  // namespace cinn
