/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
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

#include "tensorrt_llm/common/ulyssesOp.h"
#include "tensorrt_llm/common/dataType.h"
// #include "tensorrt_llm/kernels/gptKernels.h"
// #include "tensorrt_llm/kernels/mlaKernels.h"
#include "tensorrt_llm/runtime/torchUtils.h"
// #include "tensorrt_llm/runtime/utils/debugUtils.h"
// #include "tensorrt_llm/thop/thUtils.h"
// #include <cstdint>
// #include <functional>
#include <torch/extension.h>
// #include <unordered_set>

namespace torch_ext
{
using tensorrt_llm::common::op::UlyssesOp;
// using tensorrt_llm::common::op::hash;
// using tensorrt_llm::runtime::RequestType;


// enum class UlyssesType : int8_t
// {
//     ContextTP2CP,
//     ContextCP2TP,
// };

struct UlyssesCommParams
{
    // torch::ScalarType out_dtype;
    int64_t tp_size;
    int64_t tp_rank;
    int64_t cp_size;
    int64_t cp_rank;
    int64_t attn_tp_size;
    int64_t attn_cp_size;
    std::vector<int64_t> comm_group;

    // bool is_cp_attn;
    bool is_preprocess;
    int64_t batch_size;

    UlyssesCommParams() = default;

    static UlyssesCommParams fromDict(c10::Dict<std::string, c10::IValue> const& dict)
    {
        UlyssesCommParams params;
        params.tp_size = dict.at("tp_size").toInt();
        params.tp_rank = dict.at("tp_rank").toInt();
        params.cp_size = dict.at("cp_size").toInt();
        params.cp_rank = dict.at("cp_rank").toInt();
        params.attn_tp_size = dict.at("attn_tp_size").toInt();
        params.attn_cp_size = dict.at("attn_cp_size").toInt();
        params.comm_group = dict.at("comm_group").toIntList().vec();
        // params.out_dtype = dict.at("out_dtype").toScalarType();

        // params.is_cp_attn = dict.at("is_cp_attn").toBool();
        params.is_preprocess = dict.at("is_preprocess").toBool();
        params.batch_size = dict.at("batch_size").toInt();
        return params;
    }
};

struct UlyssesModelParams
{
    int64_t head_size;
    int64_t num_heads;
    int64_t num_attn_heads;
    int64_t num_attn_kv_heads;
    int64_t num_kv_heads_origin;
    int64_t qk_nope_head_dim;
    int64_t qk_rope_head_dim;
    bool is_mla_enable;

    UlyssesModelParams() = default;

    static UlyssesModelParams fromDict(c10::Dict<std::string, c10::IValue> const& dict)
    {
        UlyssesModelParams params;
        params.head_size = dict.at("head_size").toInt();
        params.num_heads = dict.at("num_heads").toInt();
        params.num_attn_heads = dict.at("num_attn_heads").toInt();
        params.num_attn_kv_heads = dict.at("num_attn_kv_heads").toInt();
        params.num_kv_heads_origin = dict.at("num_kv_heads_origin").toInt();
        params.qk_nope_head_dim = dict.at("qk_nope_head_dim").toInt();
        params.qk_rope_head_dim = dict.at("qk_rope_head_dim").toInt();
        params.is_mla_enable = dict.at("is_mla_enable").toBool();
        return params;
    }
};

template<typename T>
void run_ulysses(const std::shared_ptr<UlyssesOp>& op, const torch::Tensor& input, const torch::Tensor& output, const torch::Tensor& buffer,
    const torch::Tensor& host_context_lengths, const torch::Tensor& context_lengths, const torch::Tensor& cu_q_seqlens, const torch::Tensor& cu_cp_partial_seqlens,
    int batch_size, bool isCPAttn, bool isPreprocess, cudaStream_t stream)
{
    // 可以用↓？
    // T* input_ptr = input.data_ptr<T>();
    T* input_ptr = static_cast<T*>(input.data_ptr());
    T* output_ptr = static_cast<T*>(output.data_ptr());
    T* buffer_ptr = static_cast<T*>(buffer.data_ptr());
    // TODO: check这里int类型是否正确
    int* host_context_lengths_ptr = static_cast<int*>(host_context_lengths.data_ptr());
    int* context_lengths_ptr = static_cast<int*>(context_lengths.data_ptr());
    int* cu_q_seqlens_ptr = static_cast<int*>(cu_q_seqlens.data_ptr());
    int* cu_cp_partial_seqlens_ptr = static_cast<int*>(cu_cp_partial_seqlens.data_ptr());

    if (isCPAttn)
    {
        auto rank = op->mTpRank;
        auto size = op->mTpSize;
        // target: CP Attn
        if (isPreprocess)
        {
            // TP2CP
            TLLM_LOG_TRACE("Ulysses op TP2CP"); // TODO: add more details?
            op->ulyssesContextTP2CP(input_ptr, output_ptr, buffer_ptr, batch_size, host_context_lengths_ptr, context_lengths_ptr, cu_q_seqlens_ptr, cu_cp_partial_seqlens_ptr, rank, size, isPreprocess, stream);
        }
        else
        {
            // CP2TP
            // op->ulyssesContextCP2TP(input, output, buffer, host_context_lengths, context_lengths, cu_q_seqlens, cu_cp_partial_seqlens, rank, size, isPreprocess, stream);
        }
    }
    else
    {
        auto rank = op->mCpRank;
        auto size = op->mCpSize;
        if (isPreprocess)
        {
            // op->ulyssesContextCP2TP(input, output, buffer, host_context_lengths, context_lengths, cu_q_seqlens, cu_cp_partial_seqlens, rank, size, isPreprocess, stream);
        }
        else
        {
            // op->ulyssesContextTP2CP(input, output, buffer, host_context_lengths, context_lengths, cu_q_seqlens, cu_cp_partial_seqlens, rank, size, isPreprocess, stream);
        }
    }
}


// Input: input(token_num, head_num/tp, head_size), output(token_num/tp, head_num, head_size)
//        buffer(max_token_num * head_num / tp * head_size + 2 * (batch_size + 1))
// Output: output and cu_q_seqlens+cu_cp_partial_seqlens (2*(batch_size + 1))
torch::Tensor ulysses(torch::Tensor input, torch::Tensor output, torch::Tensor buffer,
    torch::Tensor host_context_lengths, torch::Tensor context_lengths, torch::Tensor cu_q_seqlens, torch::Tensor cu_cp_partial_seqlens,
    c10::Dict<std::string, c10::IValue> ulysses_comm_params_dict, c10::Dict<std::string, c10::IValue> ulysses_model_params_dict)
{
    TLLM_LOG_TRACE("Ulysses op starts"); // TODO: add more details?
    auto stream = at::cuda::getCurrentCUDAStream(input.get_device());

    auto op = std::make_shared<UlyssesOp>();
    auto const dtype = tensorrt_llm::runtime::TorchUtils::dataType(input.scalar_type());
    // TODO: not necessary, 外部指定T即可 但是要两边一起改下
    op->mType = dtype;
    // op->mType = tensorrt_llm::runtime::TorchUtils::dataType(ulysses_comm_params_dict.at("out_dtype").toScalarType());
    op->mCpSize = ulysses_comm_params_dict.at("cp_size").toInt();
    op->mCpRank = ulysses_comm_params_dict.at("cp_rank").toInt();
    op->mTpSize = ulysses_comm_params_dict.at("tp_size").toInt();
    op->mTpRank = ulysses_comm_params_dict.at("tp_rank").toInt();
    auto const& commGroupVec = ulysses_comm_params_dict.at("comm_group").toIntList().vec();
    op->mCommGroup = std::set<int>(commGroupVec.begin(), commGroupVec.end());
    op->mAttnTpSize = ulysses_comm_params_dict.at("attn_tp_size").toInt();
    // op->mAttnTpRank = 0;
    op->mAttnCpSize = ulysses_comm_params_dict.at("attn_cp_size").toInt();
    // op->mAttnCpRank = 0;

    op->mHeadSize = ulysses_model_params_dict.at("head_size").toInt();
    op->mNumHeads = ulysses_model_params_dict.at("num_heads").toInt();
    op->mNumAttnHeads = ulysses_model_params_dict.at("num_attn_heads").toInt();
    op->mNumAttnKVHeads = ulysses_model_params_dict.at("num_attn_kv_heads").toInt();
    // for XQA
    // op->mNumKVHeadsOrigin = -1; // no use
    //是否总是正确？
    // mUlyssesMQABroadcast = (mAttnTpSize + mNumKVHeadsOrigin - 1) / mNumKVHeadsOrigin;
    op->mUlyssesMQABroadcast = 1; //TODO
    // for MLA
    op->mIsMLAEnabled = ulysses_model_params_dict.at("is_mla_enable").toBool();
    op->mQkNopeHeadDim = ulysses_model_params_dict.at("qk_nope_head_dim").toInt();
    op->mQkRopeHeadDim = ulysses_model_params_dict.at("qk_rope_head_dim").toInt();

    op->initialize();

    TLLM_CHECK(op->mCpSize * op->mTpSize == op->mAttnCpSize * op->mAttnTpSize);
    TLLM_CHECK(op->mCpSize == 1 || op->mTpSize == 1);
    TLLM_CHECK(op->mAttnCpSize == 1 || op->mAttnTpSize == 1);
    TLLM_CHECK(op->mCpSize != op->mAttnCpSize);

    // bool isCPAttn = ulysses_comm_params_dict.at("is_cp_attn").toBool();
    bool isCPAttn = op->mCpSize < op->mAttnCpSize;
    bool isPreprocess = ulysses_comm_params_dict.at("is_preprocess").toBool();
    int batch_size = ulysses_comm_params_dict.at("batch_size").toInt();

    if (dtype == nvinfer1::DataType::kHALF)
    {
        run_ulysses<half>(op, input, output, buffer, host_context_lengths, context_lengths, cu_q_seqlens, cu_cp_partial_seqlens, batch_size, isCPAttn, isPreprocess, stream);
    } else if (dtype == nvinfer1::DataType::kBF16) {
        run_ulysses<__nv_bfloat16>(op, input, output, buffer, host_context_lengths, context_lengths, cu_q_seqlens, cu_cp_partial_seqlens, batch_size, isCPAttn, isPreprocess, stream);
    } else if (dtype == nvinfer1::DataType::kFLOAT) {
        run_ulysses<float>(op, input, output, buffer, host_context_lengths, context_lengths, cu_q_seqlens, cu_cp_partial_seqlens, batch_size, isCPAttn, isPreprocess, stream);
    } else {
        TLLM_CHECK_WITH_INFO(false, "Unsupported data type for Ulysses");
    }

    TLLM_LOG_TRACE("Ulysses op stops"); // TODO: add more details?

    return output;
}

} // namespace torch_ext

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "ulysses("
        "Tensor input"
        ", Tensor output"
        ", Tensor buffer"
        ", Tensor host_context_lengths"
        ", Tensor context_lengths"
        ", Tensor cu_q_seqlens"
        ", Tensor cu_cp_partial_seqlens"
        ", Dict(str, Any) ulysses_comm_params_dict"
        ", Dict(str, Any) ulysses_model_params_dict"
        ") -> Tensor");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("ulysses", &torch_ext::ulysses);
}
