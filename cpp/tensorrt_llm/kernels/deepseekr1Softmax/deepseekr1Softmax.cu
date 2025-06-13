/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cuda_fp16.h>
#if ENABLE_BF16
#include <cuda_bf16.h>
#endif
#include <cuda_runtime_api.h>
#include <type_traits>

#include "tensorrt_llm/common/assert.h"

#include "tensorrt_llm/kernels/deepseekr1Softmax/deepseekr1Softmax.h"
#include "tensorrt_llm/kernels/deepseekr1Softmax/deepseekr1TypeConversion.cuh"

namespace tensorrt_llm
{
namespace kernels
{
namespace deepseekr1_softmax
{
// Helper function for warp-wide max reduction
inline __device__ float warpReduceMax(float val)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    uint32_t mask = 0xffffffffu;
    asm volatile("redux.sync.max.f32 %0, %0, %1;\n" : "+f"(val) : "r"(mask));
#else
    for (int offset = 16; offset > 0; offset /= 2)
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
#endif
    return val;
}

// Helper function for warp-wide sum reduction
inline __device__ float warpReduceSum(float val)
{
    // With _xor_ all the threads have the reduced value in the end.
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;
}

// Helper function for block-wide max reduction
inline __device__ float blockReduceMax(float* shared, float val)
{
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    val = warpReduceMax(val);

    if (lane == 0)
        shared[wid] = val;
    __syncthreads();

    // All the warps have the max this way.
    val = (lane < blockDim.x / 32) ? shared[lane] : -INFINITY;
    return warpReduceMax(val);
}

// Helper function for block-wide sum reduction
inline __device__ float blockReduceSum(float* shared, float val)
{
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    val = warpReduceSum(val);

    if (lane == 0)
        shared[wid] = val;
    __syncthreads();

    // All the warps have the sum this way.
    val = (lane < blockDim.x / 32) ? shared[lane] : 0.f;
    return warpReduceSum(val);
}

template <typename scalar_t, typename acc_scalar_t = float, int num_load_elements = 8>
__global__ void deepseekr1SoftmaxKernel(
    scalar_t const* input, scalar_t* output, acc_scalar_t* output_se, int rows, int cols)
{
    // Shared memory for the reductions - we have at most 64 warps per block.
    __shared__ acc_scalar_t s_max[64], s_sum[64];

    int const row = blockIdx.x;
    if (row >= rows)
        return;
    int const col = threadIdx.x * num_load_elements;

    // Initialize reduction variables
    acc_scalar_t max_val = -INFINITY;
    acc_scalar_t float_vals[num_load_elements];

    // Process each row in chunks of num_load_elements elements
    if (col + num_load_elements - 1 < cols)
    {
        uint4 data = reinterpret_cast<uint4 const*>(&input[row * cols + col])[0];

        // Convert to float.
        float2 f0 = to_float2(reinterpret_cast<__nv_bfloat162 const&>(data.x));
        float2 f1 = to_float2(reinterpret_cast<__nv_bfloat162 const&>(data.y));
        float2 f2 = to_float2(reinterpret_cast<__nv_bfloat162 const&>(data.z));
        float2 f3 = to_float2(reinterpret_cast<__nv_bfloat162 const&>(data.w));

        // Scale with LOG2E.
        f0 = __fmul2_rn(f0, make_float2(float(M_LOG2E), float(M_LOG2E)));
        f1 = __fmul2_rn(f1, make_float2(float(M_LOG2E), float(M_LOG2E)));
        f2 = __fmul2_rn(f2, make_float2(float(M_LOG2E), float(M_LOG2E)));
        f3 = __fmul2_rn(f3, make_float2(float(M_LOG2E), float(M_LOG2E)));

        // Store in the array of regs.
        float_vals[0] = f0.x;
        float_vals[1] = f0.y;
        float_vals[2] = f1.x;
        float_vals[3] = f1.y;
        float_vals[4] = f2.x;
        float_vals[5] = f2.y;
        float_vals[6] = f3.x;
        float_vals[7] = f3.y;

        // Compute the local max.
        f0.x = fmaxf(f0.x, f0.y);
        f1.x = fmaxf(f1.x, f1.y);
        f2.x = fmaxf(f2.x, f2.y);
        f3.x = fmaxf(f3.x, f3.y);
        f0.x = fmaxf(f0.x, f1.x);
        f2.x = fmaxf(f2.x, f3.x);
        f0.x = fmaxf(f0.x, f2.x);

        // Update the max.
        max_val = fmaxf(max_val, f0.x);
    }
    else
    {
// Handle remaining elements when cols is not divisible by 8
#pragma unroll
        for (int i = 0; i < num_load_elements; ++i)
        {
            float_vals[i] = -INFINITY;
            if (col + i < cols)
            {
                float_vals[i] = to_float(input[row * cols + col + i]) * float(M_LOG2E);
            }
            max_val = fmaxf(max_val, float_vals[i]);
        }
    }

    // Perform block-wide max reduction
    max_val = blockReduceMax(s_max, max_val);

    // Compute sum of exponentials using the loaded values
    float2 sum2{0.f, 0.f};
#pragma unroll
    for (int i = 0; i < 4; i++)
    {
        float2 vals = make_float2(float_vals[i * 2 + 0], float_vals[i * 2 + 1]);
        vals = __fadd2_rn(vals, make_float2(-max_val, -max_val));
        float_vals[i * 2 + 0] = (vals.x = exp2f(vals.x));
        float_vals[i * 2 + 1] = (vals.y = exp2f(vals.y));
        sum2 = __fadd2_rn(sum2, vals);
    }

    // Perform block-wide sum reduction
    float sum = blockReduceSum(s_sum, sum2.x + sum2.y) + 1e-5f;
    if (output_se && threadIdx.x == 0)
    {
        output_se[row] = sum;
    }

    // Compute softmax and store results using the same loaded values
    float inv_sum = 1.f / sum;
    if (col + num_load_elements - 1 < cols)
    {
        float2 v0 = make_float2(float_vals[0], float_vals[1]);
        float2 v1 = make_float2(float_vals[2], float_vals[3]);
        float2 v2 = make_float2(float_vals[4], float_vals[5]);
        float2 v3 = make_float2(float_vals[6], float_vals[7]);

        v0 = __fmul2_rn(v0, make_float2(inv_sum, inv_sum));
        v1 = __fmul2_rn(v1, make_float2(inv_sum, inv_sum));
        v2 = __fmul2_rn(v2, make_float2(inv_sum, inv_sum));
        v3 = __fmul2_rn(v3, make_float2(inv_sum, inv_sum));

        uint4 data;
        asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(data.x) : "f"(v0.y), "f"(v0.x));
        asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(data.y) : "f"(v1.y), "f"(v1.x));
        asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(data.z) : "f"(v2.y), "f"(v2.x));
        asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(data.w) : "f"(v3.y), "f"(v3.x));

        // Make sure the pointer is aligned on 16B.
        reinterpret_cast<uint4*>(&output[row * cols + col])[0] = data;
    }
    else
    {
// Process remaining elements
#pragma unroll
        for (int i = 0; i < num_load_elements; i++)
        {
            float softmax_val = float_vals[i] * inv_sum;
            uint16_t data;
            asm volatile("cvt.rn.bf16.f32 %0, %1;\n" : "=h"(data) : "f"(softmax_val));
            if (col + i < cols)
            {
                output[row * cols + col + i] = data;
            }
        }
    }
}

template <typename scalar_t, typename accscalar_t>
__global__ void basicSoftmaxKernel(
    scalar_t const* input, scalar_t* output, accscalar_t* output_se, int64_t num_groups, int64_t softmax_size)
{
    // Shared memory for the reductions - we have at most 64 warps per block
    __shared__ accscalar_t s_max[64], s_sum[64];

    int const row = blockIdx.x;
    if (row >= num_groups)
        return;
    int const col = threadIdx.x;

    // Initialize reduction variables
    accscalar_t max_val = -INFINITY;
    accscalar_t sum_val = 0.0f;

    // First pass: find max value
    for (int i = col; i < softmax_size; i += blockDim.x)
    {
        accscalar_t val = static_cast<accscalar_t>(input[row * softmax_size + i]);
        max_val = fmaxf(max_val, val);
    }

    // Perform block-wide max reduction
    max_val = blockReduceMax(s_max, max_val);

    // Second pass: compute exp and sum
    for (int i = col; i < softmax_size; i += blockDim.x)
    {
        accscalar_t val = static_cast<accscalar_t>(input[row * softmax_size + i]);
        val = expf(val - max_val);
        sum_val += val;
        output[row * softmax_size + i] = static_cast<scalar_t>(val);
    }

    // Perform block-wide sum reduction
    sum_val = blockReduceSum(s_sum, sum_val) + 1e-5f;
    if (output_se && threadIdx.x == 0)
    {
        output_se[row] = sum_val;
    }

    // Third pass: normalize
    for (int i = col; i < softmax_size; i += blockDim.x)
    {
        accscalar_t val = static_cast<accscalar_t>(output[row * softmax_size + i]);
        output[row * softmax_size + i] = static_cast<scalar_t>(val / sum_val);
    }
}

template <typename scalar_t, typename accscalar_t>
void launchDeepseekr1SoftmaxKernel(scalar_t const* input, scalar_t* output, accscalar_t* output_se, int64_t num_groups,
    int64_t softmax_size, cudaStream_t stream)
{
    // Calculate grid and block dimensions
    int grid_size = num_groups;
    int block_size = 256;
    TLLM_CHECK_WITH_INFO(std::is_floating_point<accscalar_t>::value, "accscalar_t must be float");
    // Launch the kernel
    if (softmax_size == 2048)
    {
        deepseekr1SoftmaxKernel<scalar_t, accscalar_t>
            <<<grid_size, block_size, 0, stream>>>(input, output, output_se, num_groups, softmax_size);
    }
    else
    {
        basicSoftmaxKernel<scalar_t, accscalar_t>
            <<<grid_size, block_size, 0, stream>>>(input, output, output_se, num_groups, softmax_size);
    }
}

// Explicit template instantiations for standard floating types
template void launchDeepseekr1SoftmaxKernel<float, float>(float const*, float*, float*, int64_t, int64_t, cudaStream_t);
// Explicit template instantiations for Half and BFloat16 (using float for accumulation)
template void launchDeepseekr1SoftmaxKernel<half, float>(half const*, half*, float*, int64_t, int64_t, cudaStream_t);
#ifdef ENABLE_BF16
template void launchDeepseekr1SoftmaxKernel<__nv_bfloat16, float>(
    __nv_bfloat16 const*, __nv_bfloat16*, float*, int64_t, int64_t, cudaStream_t);
#endif

} // namespace deepseekr1_softmax
} // namespace kernels
} // namespace tensorrt_llm
