/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
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
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/deepseekr1Softmax/deepseekr1Softmax.h"
#include "tensorrt_llm/thop/thUtils.h"
#include <cuda_fp16.h>
#if ENABLE_BF16
#include <cuda_bf16.h>
#endif

namespace torch_ext
{

template <typename scalar_t, typename accscalar_t = float>
torch::Tensor deepseekr1SoftmaxImpl(torch::Tensor const& input)
{
    torch::Tensor output = torch::empty_like(input);
    // Get tensor information
    int64_t input_dim = input.dim();
    int64_t softmax_size = input.size(input_dim - 1);
    int64_t num_groups = 1;
    for (int64_t i = 0; i < input_dim - 1; ++i)
    {
        num_groups *= input.size(i);
    }

    // Get pointers to data
    scalar_t const* input_ptr = reinterpret_cast<scalar_t const*>(input.data_ptr());
    scalar_t* output_ptr = reinterpret_cast<scalar_t*>(output.data_ptr());

    // Get CUDA stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    // Launch kernel
    tensorrt_llm::kernels::deepseekr1_softmax::launchDeepseekr1SoftmaxKernel<scalar_t, accscalar_t>(
        input_ptr, output_ptr, nullptr, num_groups, softmax_size, stream);

    sync_check_cuda_error(stream);
    return output;
}

template <typename scalar_t, typename accscalar_t = float>
std::tuple<torch::Tensor, torch::Tensor> deepseekr1SoftmaxWithSEImpl(torch::Tensor const& input)
{
    torch::Tensor output = torch::empty_like(input);
    // Get tensor information
    int64_t input_dim = input.dim();
    int64_t softmax_size = input.size(input_dim - 1);
    int64_t num_groups = 1;
    for (int64_t i = 0; i < input_dim - 1; ++i)
    {
        num_groups *= input.size(i);
    }
    torch::Tensor output_se
        = torch::empty(num_groups, torch::TensorOptions().device(input.device()).dtype(torch::kFloat32));

    // Get pointers to data
    scalar_t const* input_ptr = reinterpret_cast<scalar_t const*>(input.data_ptr());
    scalar_t* output_ptr = reinterpret_cast<scalar_t*>(output.data_ptr());
    accscalar_t* output_se_ptr = reinterpret_cast<accscalar_t*>(output_se.data_ptr());

    // Get CUDA stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    // Launch kernel
    tensorrt_llm::kernels::deepseekr1_softmax::launchDeepseekr1SoftmaxKernel<scalar_t, accscalar_t>(
        input_ptr, output_ptr, output_se_ptr, num_groups, softmax_size, stream);

    sync_check_cuda_error(stream);
    return std::make_tuple(output, output_se);
}

torch::Tensor deepseekr1Softmax(torch::Tensor const& input, int64_t dim = -1)
{
    // Check that both tensors are on CUDA
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA");

    // Normalize dimension
    if (dim < 0)
    {
        dim += input.dim();
    }
    TORCH_CHECK(dim >= 0 && dim < input.dim(), "Dimension out of range");
    TORCH_CHECK(dim == input.dim() - 1, "Dimension must be the last dimension");
    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16, "Input tensor must be bfloat16");

    // Handle different input types explicitly
    switch (input.scalar_type())
    {
    case torch::kFloat: return deepseekr1SoftmaxImpl<float, float>(input);
    case torch::kHalf: return deepseekr1SoftmaxImpl<half, float>(input);
    case torch::kBFloat16: return deepseekr1SoftmaxImpl<__nv_bfloat16, float>(input);
    default: // Handle other data types
        throw std::invalid_argument("Invalid dtype, only supports float16, float32, and bfloat16");
    }
}

std::tuple<torch::Tensor, torch::Tensor> deepseekr1SoftmaxWithSE(torch::Tensor const& input, int64_t dim = -1)
{
    // Check that both tensors are on CUDA
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA");

    // Normalize dimension
    if (dim < 0)
    {
        dim += input.dim();
    }
    TORCH_CHECK(dim >= 0 && dim < input.dim(), "Dimension out of range");
    TORCH_CHECK(dim == input.dim() - 1, "Dimension must be the last dimension");
    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16, "Input tensor must be bfloat16");

    // Handle different input types explicitly
    switch (input.scalar_type())
    {
    case torch::kFloat: return deepseekr1SoftmaxWithSEImpl<float, float>(input);
    case torch::kHalf: return deepseekr1SoftmaxWithSEImpl<half, float>(input);
    case torch::kBFloat16: return deepseekr1SoftmaxWithSEImpl<__nv_bfloat16, float>(input);
    default: // Handle other data types
        throw std::invalid_argument("Invalid dtype, only supports float16, float32, and bfloat16");
    }
}

} // namespace torch_ext

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def("deepseekr1_softmax(Tensor input, int dim) -> Tensor");
    m.def("deepseekr1_softmax_with_se(Tensor input, int dim) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("deepseekr1_softmax", &torch_ext::deepseekr1Softmax);
    m.impl("deepseekr1_softmax_with_se", &torch_ext::deepseekr1SoftmaxWithSE);
}
