/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

// #include "tensorrt_llm/common/cublasMMWrapper.h"
#include "tensorrt_llm/common/opUtils.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"
// #include "tensorrt_llm/common/quantization.h"
#include "tensorrt_llm/kernels/contextFusedMultiHeadAttention/fused_multihead_attention_common.h"
// #include "tensorrt_llm/kernels/decoderMaskedMultiheadAttention/decoderXQARunner.h"
// #include "tensorrt_llm/kernels/fmhaDispatcher.h"
// #include "tensorrt_llm/kernels/gptKernels.h"
// #include "tensorrt_llm/kernels/internal_cutlass_kernels/include/fp8_blockscale_gemm.h"
// #include "tensorrt_llm/kernels/kvCacheUtils.h"
// #include "tensorrt_llm/kernels/mlaKernels.h"
// #include "tensorrt_llm/kernels/xqaDispatcher.h"
// #include <cassert>
#include <set>
// #include <string>
// #include <vector>
#if ENABLE_MULTI_DEVICE
#include <nccl.h>
#endif // ENABLE_MULTI_DEVICE

namespace tensorrt_llm::common::op
{

class UlyssesOp
{
public:
    UlyssesOp() = default;
    UlyssesOp(int cp_size, int cp_rank, const std::set<int32_t>& comm_group, int head_size, int num_heads, int num_attn_heads, int num_attn_kv_heads,
              int num_kv_heads_origin, int attn_tp_size, int attn_tp_rank, int attn_cp_size, int attn_cp_rank,
              int ulysses_mqa_broadcast, nvinfer1::DataType type, bool is_mla_enabled, int qk_nope_head_dim, int qk_rope_head_dim)
        : mCpSize(cp_size), mCpRank(cp_rank), mCommGroup(comm_group), mHeadSize(head_size), mNumHeads(num_heads), mNumAttnHeads(num_attn_heads),
          mNumAttnKVHeads(num_attn_kv_heads), mNumKVHeadsOrigin(num_kv_heads_origin), mAttnTpSize(attn_tp_size),
          mAttnTpRank(attn_tp_rank), mAttnCpSize(attn_cp_size), mAttnCpRank(attn_cp_rank), mUlyssesMQABroadcast(ulysses_mqa_broadcast),
          mType(type), mIsMLAEnabled(is_mla_enabled), mQkNopeHeadDim(qk_nope_head_dim), mQkRopeHeadDim(qk_rope_head_dim)
    {
    }
    ~UlyssesOp() = default;

    int initialize()
    {
#if ENABLE_MULTI_DEVICE
    if (mCpSize != mAttnCpSize && COMM_SESSION.getSize() > 1)
    {
        TLLM_LOG_TRACE("%s start for rank %d", __PRETTY_FUNCTION__, COMM_SESSION.getRank());
        mNcclComm = getComm(mCommGroup);
        TLLM_LOG_TRACE("%s stop for rank %d", __PRETTY_FUNCTION__, COMM_SESSION.getRank());
    }
#endif // ENABLE_MULTI_DEVICE
    return 0;
    }

    template <typename T>
    int ulyssesContextCP2TP(T const* input, T* output, T* buffer, int32_t batch_size, int32_t const* host_context_lengths,
        int const* context_lengths, int const* cu_q_seqlens, int const* cu_cp_partial_seqlens, int rank, int size, bool isPreprocess,
        cudaStream_t stream);

    template <typename T>
    int ulyssesContextTP2CP(T const* input, T* output, T* buffer, int32_t batch_size, int32_t const* host_context_lengths,
        int const* context_lengths, int const* cu_q_seqlens, int const* cu_cp_partial_seqlens, int rank, int size, bool isPreprocess,
        cudaStream_t stream);

    template <typename T>
    int ulyssesGenerationPreprocess(T const* input, T* output, T* buffer, int32_t batch_beam, cudaStream_t stream);

    template <typename T>
    int ulyssesGenerationPostprocess(T const* input, T* output, T* buffer, int32_t batch_beam, cudaStream_t stream);

    // [[nodiscard]] std::string toString() const;

    nvinfer1::DataType mType;
    bool mIsMLAEnabled = false;
    int mCpSize = 1;
    int mCpRank = 0;
    int mTpSize = 1;
    int mTpRank = 0;
    std::set<int32_t> mCommGroup = {};
    // These parameters are used to specifically configure the attention attributes when cp/tp_size are different
    // between Attention and FFN(such as Ulysses)
    int mHeadSize = -1;
    int mNumHeads = -1;
    int mNumAttnHeads = -1;
    int mNumAttnKVHeads = -1;
    // for XQA
    int mNumKVHeadsOrigin = -1;
    // for MLA
    int mQkNopeHeadDim = -1;
    int mQkRopeHeadDim = -1;
    int mAttnTpSize = -1;
    int mAttnTpRank = 0;
    int mAttnCpSize = -1;
    int mAttnCpRank = 0;
    int mUlyssesMQABroadcast = 1;

    // [[nodiscard]] auto data() const
    // {
    //     return std::make_tuple(mCpSize, mCpRank, mCpGroup, mNumAttnHeads, mNumAttnKVHeads, mNumKVHeadsOrigin,
    //         mAttnTpSize, mAttnTpRank, mAttnCpSize, mAttnCpRank, mUlyssesMQABroadcast);
    // };

private:
#if ENABLE_MULTI_DEVICE
    std::shared_ptr<ncclComm_t> mNcclComm;
#endif // ENABLE_MULTI_DEVICE
};

} // namespace tensorrt_llm::common::op
