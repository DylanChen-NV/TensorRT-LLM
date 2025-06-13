#pragma once

#include "tensorrt_llm/common/cudaUtils.h"
#include <cstdint>

namespace tensorrt_llm
{
namespace kernels
{
namespace deepseekr1_softmax
{

template <typename T, typename accscalar_t>
void launchDeepseekr1SoftmaxKernel(T const* input, T* output, accscalar_t* output_se, int64_t num_groups,
    int64_t softmax_size, cudaStream_t stream = nullptr);

} // namespace deepseekr1_softmax
} // namespace kernels
} // namespace tensorrt_llm
