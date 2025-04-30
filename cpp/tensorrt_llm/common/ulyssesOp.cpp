/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION &
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
#include "ulyssesOp.h"
// #include "tensorrt_llm/common/assert.h"
// #include "tensorrt_llm/common/envUtils.h"
// #include "tensorrt_llm/common/memoryUtils.h"
// #include "tensorrt_llm/kernels/decoderMaskedMultiheadAttention.h"
// #include "tensorrt_llm/kernels/flashMLA/flash_mla.h"
// #include "tensorrt_llm/kernels/gptKernels.h"
// #include "tensorrt_llm/kernels/kvCacheUtils.h"
#include "tensorrt_llm/kernels/unfusedAttentionKernels.h"
// #include "tensorrt_llm/runtime/iBuffer.h"
// #include "tensorrt_llm/runtime/utils/debugUtils.h"
// #include "tensorrt_llm/runtime/utils/mpiUtils.h"
// #include <algorithm>
// #include <cstdint>
// #include <type_traits>

using namespace tensorrt_llm::kernels;
namespace tc = tensorrt_llm::common;
using tensorrt_llm::common::op::UlyssesOp;

template <typename T>
int UlyssesOp::ulyssesContextCP2TP(T const* input, T* output, T* buffer, int batch_size, int const* host_context_lengths, 
    int const* context_lengths, int const* cu_q_seqlens, int const* cu_cp_partial_seqlens, int rank, int size, bool isPreprocess,
    cudaStream_t stream)
{
    // 这里的mCpSize理论上应该是 mCpSize / mAttnCpSize,即需要转TP的CP数，不过mAttnCpSize暂时只能是1 or mCpSize
    // rank 应该是 mCpRank % (mCpSize / mAttnCpSize)
    int32_t partialTokenNum = 0;
    int32_t maxPartialLength = 0;
    for (int32_t batchIdx = 0; batchIdx < batch_size; ++batchIdx)
    {
        int32_t partialLength = (host_context_lengths[batchIdx] + size - 1) / size;
        maxPartialLength = std::max(maxPartialLength, partialLength);
        partialTokenNum += partialLength;
    }

    int64_t numHeadsOutput, partialHeads, partialHiddenSize, headSizeQ, headSizeK, headSizeV;
    if (isPreprocess)
    {
        numHeadsOutput = mNumAttnHeads;
        if (mIsMLAEnabled) // TODO adhoc 默认 MLA 是 TP GEMM + CP Attn, 后续把入口集中下有个dispatcher
        {
            TLLM_CHECK_WITH_INFO(false, "czq CP2TP-preprocess-MLA not implemented");
        }
        else
        {
            partialHeads = numHeadsOutput + 2 * mNumAttnKVHeads;
            partialHiddenSize = partialHeads * mHeadSize;
            headSizeQ = mHeadSize;
            headSizeK = mHeadSize;
            headSizeV = mHeadSize;
        }
    }
    else
    {
        numHeadsOutput = mNumHeads;
        if (mIsMLAEnabled)
        {
            // is postprocess
            partialHeads = mQkNopeHeadDim;
            partialHiddenSize = partialHeads * mQkNopeHeadDim;
            headSizeQ = mQkNopeHeadDim;
            headSizeK = 0;
            headSizeV = 0;
        }
        else
        {
            TLLM_CHECK_WITH_INFO(false, "czq CP2TP-postprocess-MHA not implemented");
        }
    }

    // full request: [bs, seqlen, head, headSize]
    //
    // input of cp: [bs, partialLength, head, headSize]
    // view_1 as [bs, partialLength, cpSize_Head, partialHead, headSize]
    // transpose_1 as [cpSize_Head, bs, partialLenth, partialHead, headSize]
    // all-to-all to get [cpSize_Length, bs, partialLength, partialHead, headSize]
    // transpose_2 to [bs, cpSize_Length, partialLength, partialHead, headSize]
    // view_2 as [bs, totalLength, partialHead, headSize]
    // and this is same to the input under TP.
    //
    // when we use remove_input_padding, bs and length are fused into numTokens. So, we need to
    // insert the cpSize_Length dimension of transpose_2 into numTokens directly like
    // input of cp: [partialNumTokens, head, headSize]
    // view_1 as [partialNumTokens, cpSize_Head, partialHead, headSize]
    // transpose_1 as [cpSize_Head, partialNumTokens, partialHead, headSize]
    // all-to-all to get [cpSize_Length, partialNumTokens, partialHead, headSize]
    // transpose_2 as [NumTokens, partialHead, headSize]
    // and this is same to the input under TP.

            // if (rank == 0)
            // {
            //     printf("czq attn cp2tp input %p", input);
            //     for (int idx = 0; idx < 5 * 32; ++idx)
            //     {
            //         const int tid = idx / 32;
            //         const int hid = idx % 32;

            //         const int bufSize = 128;
            //         const int line = 128;
            //         const int lineN = 128;
            //         std::vector<T> host_buf(bufSize, 0);
            //         cudaMemcpyAsync(host_buf.data(), input + idx * bufSize, bufSize * sizeof(T), cudaMemcpyDeviceToHost, stream);
            //         sync_check_cuda_error(stream);
            //         for (int i=0;i<host_buf.size();++i)
            //         {
            //             if (i % line == 0) printf("\ntid%d,hid%d,line%d:", tid, hid, i / line);
            //             if (i % line < lineN) 
            //                 printf("%f ", float(host_buf[i]));
            //         }
            //         printf("end \n");
            //         sync_check_cuda_error(stream);
            //     }
            // }

    // view_1 + transpose_1
    invokeCpTranspose(output, buffer, input, partialTokenNum, size, numHeadsOutput, mNumAttnKVHeads,
        mUlyssesMQABroadcast, headSizeQ, headSizeK, headSizeV, rank, stream);
    sync_check_cuda_error(stream);

            // if (rank == 0)
            // {
            //     printf("czq attn cp2tp trans output %p", output);
            //     for (int idx = 0; idx < 5 * 32; ++idx)
            //     {
            //         const int tid = idx / 32;
            //         const int hid = idx % 32;

            //         const int bufSize = 128;
            //         const int line = 128;
            //         const int lineN = 128;
            //         std::vector<T> host_buf(bufSize, 0);
            //         cudaMemcpyAsync(host_buf.data(), output + idx * bufSize, bufSize * sizeof(T), cudaMemcpyDeviceToHost, stream);
            //         sync_check_cuda_error(stream);
            //         for (int i=0;i<host_buf.size();++i)
            //         {
            //             if (i % line == 0) printf("\ntid%d,hid%d,line%d:", tid, hid, i / line);
            //             if (i % line < lineN) 
            //                 printf("%f ", float(host_buf[i]));
            //         }
            //         printf("end \n");
            //         sync_check_cuda_error(stream);
            //     }
            // }
            // if (rank == 0)
            // {
            //     printf("czq attn cp2tp trans buffer %p", buffer);
            //     for (int idx = 0; idx < 5 * 32; ++idx)
            //     {
            //         const int tid = idx / 32;
            //         const int hid = idx % 32;

            //         const int bufSize = 128;
            //         const int line = 128;
            //         const int lineN = 128;
            //         std::vector<T> host_buf(bufSize, 0);
            //         cudaMemcpyAsync(host_buf.data(), buffer + idx * bufSize, bufSize * sizeof(T), cudaMemcpyDeviceToHost, stream);
            //         sync_check_cuda_error(stream);
            //         for (int i=0;i<host_buf.size();++i)
            //         {
            //             if (i % line == 0) printf("\ntid%d,hid%d,line%d:", tid, hid, i / line);
            //             if (i % line < lineN) 
            //                 printf("%f ", float(host_buf[i]));
            //         }
            //         printf("end \n");
            //         sync_check_cuda_error(stream);
            //     }
            // }

    // Do all to all
#if ENABLE_MULTI_DEVICE
    ncclGroupStart();
    for (int cpIdx = 0; cpIdx < size; cpIdx++)
    {
        if (cpIdx != rank)
        {
            NCCLCHECK(ncclSend(output + cpIdx * (partialTokenNum * partialHiddenSize),
                (partialTokenNum * partialHiddenSize), (*getDtypeMap())[mType], cpIdx, *mNcclComm, stream));
            NCCLCHECK(ncclRecv(buffer + cpIdx * (partialTokenNum * partialHiddenSize),
                (partialTokenNum * partialHiddenSize), (*getDtypeMap())[mType], cpIdx, *mNcclComm, stream));
        }
    }
    ncclGroupEnd();
    sync_check_cuda_error(stream);
#endif // ENABLE_MULTI_DEVICE

    // transpose_2 + view_2
    invokeCpTranspose2(output, buffer, context_lengths, cu_q_seqlens, cu_cp_partial_seqlens, size,
        maxPartialLength, batch_size, partialHiddenSize, stream);

    // if (rank == 0)
    // {
    //     const int bufSize = 16*16*128;
    //     const int line = 64;
    //     const int lineN = 4;
    //     std::vector<T> host_buf(bufSize, 0);
    //     cudaMemcpyAsync(host_buf.data(), output, bufSize * sizeof(T), cudaMemcpyDeviceToHost, stream);
    //     sync_check_cuda_error(stream);
    //     printf("czq attn output final");
    //     for (int i=0;i<host_buf.size();++i)
    //     {
    //         if (i % line == 0) printf("\nline%d:", i / line);
    //         if (i % line < lineN)
    //             printf("%f ", float(host_buf[i]));
    //     }
    //     printf("end \n");
    // }

    return 0;
}

// template <typename T>
// int UlyssesOp::ulyssesContextPreprocess(T const* input, T* output, T* buffer,
//     int batch_size, int const* host_context_lengths, int const* context_lengths,
//     int const* cu_q_seqlens, int const* cu_cp_partial_seqlens, cudaStream_t stream)
// {
//     int32_t partialTokenNum = 0;
//     int32_t maxPartialLength = 0;
//     for (int32_t batchIdx = 0; batchIdx < batch_size; ++batchIdx)
//     {
//         int32_t partialLength = (host_context_lengths[batchIdx] + mCpSize - 1) / mCpSize;
//         maxPartialLength = std::max(maxPartialLength, partialLength);
//         partialTokenNum += partialLength;
//     }
//     auto const partialHeads = mNumAttnHeads + 2 * mNumAttnKVHeads;

//     // full request: [bs, seqlen, head, headSize]
//     //
//     // input of cp: [bs, partialLength, head, headSize]
//     // view_1 as [bs, partialLength, cpSize_Head, partialHead, headSize]
//     // transpose_1 as [cpSize_Head, bs, partialLenth, partialHead, headSize]
//     // all-to-all to get [cpSize_Length, bs, partialLength, partialHead, headSize]
//     // transpose_2 to [bs, cpSize_Length, partialLength, partialHead, headSize]
//     // view_2 as [bs, totalLength, partialHead, headSize]
//     // and this is same to the input under TP.
//     //
//     // when we use remove_input_padding, bs and length are fused into numTokens. So, we need to
//     // insert the cpSize_Length dimension of transpose_2 into numTokens directly like
//     // input of cp: [partialNumTokens, head, headSize]
//     // view_1 as [partialNumTokens, cpSize_Head, partialHead, headSize]
//     // transpose_1 as [cpSize_Head, partialNumTokens, partialHead, headSize]
//     // all-to-all to get [cpSize_Length, partialNumTokens, partialHead, headSize]
//     // transpose_2 as [NumTokens, partialHead, headSize]
//     // and this is same to the input under TP.

//     // view_1 + transpose_1
//     invokeCpTranspose(output, buffer, input, partialTokenNum, mCpSize, mNumAttnHeads, mNumAttnKVHeads,
//         mUlyssesMQABroadcast, mHeadSize, mCpRank, stream);
//     sync_check_cuda_error(stream);

//     // Do all to all
// #if ENABLE_MULTI_DEVICE
//     ncclGroupStart();
//     for (int cpIdx = 0; cpIdx < mCpSize; cpIdx++)
//     {
//         if (cpIdx != mCpRank)
//         {
//             NCCLCHECK(ncclSend(output + cpIdx * (partialTokenNum * mHeadSize * partialHeads),
//                 (partialTokenNum * mHeadSize * partialHeads), (*getDtypeMap())[mType], cpIdx, *mCpNcclComm,
//                 stream));
//             NCCLCHECK(ncclRecv(buffer + cpIdx * (partialTokenNum * mHeadSize * partialHeads),
//                 (partialTokenNum * mHeadSize * partialHeads), (*getDtypeMap())[mType], cpIdx, *mCpNcclComm,
//                 stream));
//         }
//     }
//     ncclGroupEnd();
//     sync_check_cuda_error(stream);
// #endif // ENABLE_MULTI_DEVICE

//     // transpose_2 + view_2
//     invokeCpTranspose2(output, buffer, context_lengths, cu_q_seqlens, cu_cp_partial_seqlens, mCpSize,
//         maxPartialLength, batch_size, partialHeads, mHeadSize, stream);

//     return 0;
// }

template <typename T>
int UlyssesOp::ulyssesContextTP2CP(T const* input, T* output, T* buffer, int batch_size, int const* host_context_lengths,
    int const* context_lengths, int const* cu_q_seqlens, int const* cu_cp_partial_seqlens, int rank, int size, bool isPreprocess,
    cudaStream_t stream)
{
    // After FMHA, we get result [numTokens(bs, cp, paritalLength), partialHead, headSize]
    // transpose_2_reverse: [cpSize_Length, partialTokens(bs, partialLength), partialHead, headSize]
    // all-to-all: [cpSize_Head, partialTokens, partialHead, headSize]
    // transpose_1_reverse: [partialTokens, cpSize_Head, partialHead, headSize]
    // view: [partialTokens, head, headSize]

            // if (rank == 0)
            // {
            //     printf("czq attn tp2cp input %p", input);
            //     for (int idx = 0; idx < 10 * 16; ++idx)
            //     {
            //         const int tid = idx / 16;
            //         const int hid = idx % 16;

            //         const int bufSize = 512;
            //         const int line = 64;
            //         const int lineN = 64;
            //         std::vector<T> host_buf(bufSize, 0);
            //         cudaMemcpyAsync(host_buf.data(), input + idx * bufSize, bufSize * sizeof(T), cudaMemcpyDeviceToHost, stream);
            //         sync_check_cuda_error(stream);
            //         for (int i=0;i<host_buf.size();++i)
            //         {
            //             if (i % line == 0) printf("\ntid%d,hid%d,line%d:", tid, hid, i / line);
            //             if (i % line < lineN) 
            //                 printf("%f ", float(host_buf[i]));
            //         }
            //         printf("end \n");
            //         sync_check_cuda_error(stream);
            //     }
            // }

    int32_t maxPartialLength = 0;
    int32_t partialTokenNum = 0;
    for (int32_t batchIdx = 0; batchIdx < batch_size; ++batchIdx)
    {
        int32_t partialLength = (host_context_lengths[batchIdx] + size - 1) / size;
        maxPartialLength = std::max(maxPartialLength, partialLength);
        partialTokenNum += partialLength;
    }

    int64_t numHeadsInput, hiddenSize;
    if (isPreprocess)
    {
        numHeadsInput = mNumHeads;
        if (mIsMLAEnabled) // TODO adhoc 默认 MLA 是 TP GEMM + CP Attn, 后续把入口集中下有个dispatcher
        {
            hiddenSize = (mQkNopeHeadDim * 3 + mQkRopeHeadDim * 2) * numHeadsInput;
            TLLM_LOG_ERROR(
                "czq TP2CP-preprocess-MLA hiddenSize %d, mQkNopeHeadDim %d, mQkRopeHeadDim %d, numHeadsInput %d",
                hiddenSize, mQkNopeHeadDim, mQkRopeHeadDim, numHeadsInput);
        }
        else
        {
            TLLM_CHECK_WITH_INFO(false, "czq TP2CP-preprocess-MHA not implemented");
        }
    }
    else
    {
        numHeadsInput = mNumAttnHeads;
        if (mIsMLAEnabled)
        {
            TLLM_CHECK_WITH_INFO(false, "czq TP2CP-postprocess-MLA not implemented");
        }
        else
        {
            hiddenSize = numHeadsInput * mHeadSize;
        }
    }

    // transpose_2_reverse
    if (mType == nvinfer1::DataType::kFP8)
    {
        invokeCpTransposeToSeqMajor2(reinterpret_cast<__nv_fp8_e4m3*>(buffer), reinterpret_cast<__nv_fp8_e4m3*>(output),
            reinterpret_cast<__nv_fp8_e4m3 const*>(input), context_lengths, cu_q_seqlens, cu_cp_partial_seqlens,
            size, maxPartialLength, batch_size, hiddenSize, rank, stream);
    }
    else
    {
        invokeCpTransposeToSeqMajor2(buffer, output, input, context_lengths, cu_q_seqlens, cu_cp_partial_seqlens,
            size, maxPartialLength, batch_size, hiddenSize, rank, stream);
    }
    // if (rank == 0)
    // {
    //     printf("czq attn qkv input after transpose_2_reverse %p", input);
    //     for (int idx = 0; idx < 10 * 16; ++idx)
    //     {
    //         const int tid = idx / 16;
    //         const int hid = idx % 16;

    //         const int bufSize = 512;
    //         const int line = 64;
    //         const int lineN = 64;
    //         std::vector<T> host_buf(bufSize, 0);
    //         cudaMemcpyAsync(host_buf.data(), input + idx * bufSize, bufSize * sizeof(T), cudaMemcpyDeviceToHost,
    //         stream); sync_check_cuda_error(stream); for (int i=0;i<host_buf.size();++i)
    //         {
    //             if (i % line == 0) printf("\ntid%d,hid%d,line%d:", tid, hid, i / line);
    //             if (i % line < lineN)
    //                 printf("%f ", float(host_buf[i]));
    //         }
    //         printf("end \n");
    //         sync_check_cuda_error(stream);
    //     }
    // }
    // if (rank == 0)
    // {
    //     printf("czq attn qkv buffer after transpose_2_reverse %p", buffer);
    //     for (int idx = 0; idx < 10 * 16; ++idx)
    //     {
    //         const int tid = idx / 16;
    //         const int hid = idx % 16;

    //         const int bufSize = 512;
    //         const int line = 64;
    //         const int lineN = 64;
    //         std::vector<T> host_buf(bufSize, 0);
    //         cudaMemcpyAsync(host_buf.data(), buffer + idx * bufSize, bufSize * sizeof(T), cudaMemcpyDeviceToHost,
    //         stream); sync_check_cuda_error(stream); for (int i=0;i<host_buf.size();++i)
    //         {
    //             if (i % line == 0) printf("\ntid%d,hid%d,line%d:", tid, hid, i / line);
    //             if (i % line < lineN)
    //                 printf("%f ", float(host_buf[i]));
    //         }
    //         printf("end \n");
    //         sync_check_cuda_error(stream);
    //     }
    // }
    // return 0;

    // all-to-all
#if ENABLE_MULTI_DEVICE
    size_t elementNum = partialTokenNum * hiddenSize;
    ncclGroupStart();
    for (int cpIdx = 0; cpIdx < size; cpIdx++)
    {
        if (cpIdx != rank)
        {
            if (mType == nvinfer1::DataType::kFP8)
            {
                NCCLCHECK(ncclSend(reinterpret_cast<__nv_fp8_e4m3*>(output) + cpIdx * elementNum, elementNum, ncclInt8,
                    cpIdx, *mNcclComm, stream));
                NCCLCHECK(ncclRecv(reinterpret_cast<__nv_fp8_e4m3*>(buffer) + cpIdx * elementNum, elementNum, ncclInt8,
                    cpIdx, *mNcclComm, stream));
            }
            else
            {
                NCCLCHECK(ncclSend(
                    output + cpIdx * elementNum, elementNum, (*getDtypeMap())[mType], cpIdx, *mNcclComm, stream));
                NCCLCHECK(ncclRecv(
                    buffer + cpIdx * elementNum, elementNum, (*getDtypeMap())[mType], cpIdx, *mNcclComm, stream));
            }
        }
    }
    ncclGroupEnd();
#endif // ENABLE_MULTI_DEVICE

    // transpose_1_reverse + view
    if (mType == nvinfer1::DataType::kFP8)
    {
        invokeCpTransposeToSeqMajor<__nv_fp8_e4m3>(reinterpret_cast<__nv_fp8_e4m3*>(output),
            reinterpret_cast<__nv_fp8_e4m3 const*>(buffer), reinterpret_cast<__nv_fp8_e4m3 const*>(buffer),
            partialTokenNum, size, hiddenSize, mCpRank, stream);
    }
    else
    {
        invokeCpTransposeToSeqMajor<T>((T*) output, buffer, buffer, partialTokenNum, size, hiddenSize, mCpRank, stream);
    }
            // if (rank == 0)
            // {
            //     printf("czq attn tp2cp output %p", output);
            //     for (int idx = 0; idx < 5 * 32; ++idx)
            //     {
            //         const int tid = idx / 32;
            //         const int hid = idx % 32;

            //         const int bufSize = 512;
            //         const int line = 64;
            //         const int lineN = 64;
            //         std::vector<T> host_buf(bufSize, 0);
            //         cudaMemcpyAsync(host_buf.data(), output + idx * bufSize, bufSize * sizeof(T), cudaMemcpyDeviceToHost, stream);
            //         sync_check_cuda_error(stream);
            //         for (int i=0;i<host_buf.size();++i)
            //         {
            //             if (i % line == 0) printf("\ntid%d,hid%d,line%d:", tid, hid, i / line);
            //             if (i % line < lineN) 
            //                 printf("%f ", float(host_buf[i]));
            //         }
            //         printf("end \n");
            //         sync_check_cuda_error(stream);
            //     }
            // }
    return 0;
}

template <typename T>
int UlyssesOp::ulyssesGenerationPreprocess(
    T const* input, T* output, T* buffer, int32_t batch_beam, cudaStream_t stream)
{
    if (mCpSize <= 1)
        return 0;

    auto const partialTokenNum = (batch_beam + mCpSize - 1) / mCpSize;
    int64_t headSizeQ = mHeadSize;
    int64_t headSizeK = mHeadSize;
    int64_t headSizeV = mHeadSize;

    // attention_input shape: [partialTokenNum, numHeads, headSize]
    // view_1: [partialTokenNum, cpSize_Head, partialHeads, headSize]
    // transpose_1: [cpSize_Head, partialTokenNum, partialHeads, headSize]
    // all-to-all to get [cpSize_Length, partialTokenNum, partialHead, headSize]
    // view_2 as [tokens, partialHead, headSize]

    // do transpose_1
    // [1, mNumHeads + 2*mNumKVHeads, headSize]
    // -> (view) [1, cpSize * mNumAttnHeads + cpSize * mNumAttnKVHeads + cpSize * partilKVHeads,
    // headSize]
    // -> (transpose) [cpSize, 1, mNumAttnHeads + mNumAttnKVHeads + mNumAttnKVHeads, headSize]
    invokeCpTranspose(buffer, output, input, partialTokenNum, mCpSize, mNumAttnHeads, mNumAttnKVHeads,
        mUlyssesMQABroadcast, headSizeQ, headSizeK, headSizeV, mCpRank, stream);
    sync_check_cuda_error(stream);

    // Do all to all
#if ENABLE_MULTI_DEVICE
    auto const partialHeads = mNumAttnHeads + 2 * mNumAttnKVHeads;

    ncclGroupStart();
    for (int cpIdx = 0; cpIdx < mCpSize; cpIdx++)
    {
        if (cpIdx != mCpRank)
        {
            NCCLCHECK(ncclSend(buffer + cpIdx * (partialTokenNum * mHeadSize * partialHeads),
                (partialTokenNum * mHeadSize * partialHeads), (*getDtypeMap())[mType], cpIdx, *mNcclComm, stream));
            NCCLCHECK(ncclRecv(output + cpIdx * (partialTokenNum * mHeadSize * partialHeads),
                (partialTokenNum * mHeadSize * partialHeads), (*getDtypeMap())[mType], cpIdx, *mNcclComm, stream));
        }
    }
    ncclGroupEnd();
    sync_check_cuda_error(stream);
#endif // ENABLE_MULTI_DEVICE
    return 0;
}

template <typename T>
int UlyssesOp::ulyssesGenerationPostprocess(T const* input, T* output, T* buffer, int32_t batch_beam, cudaStream_t stream)
{
    if (mCpSize <= 1)
        return 0;

    // mmha output shape: [tokens, partialHead, headSize]
    // view: [cpSize_Length, partialTokens, partialHead, headSize]
    // all-to-all: [cpSize_Head, partialTokens, partialHead, headSize]
    // transpose_1_reverse: [partialTokens, cpSize_Head, partialHead, headSize]
    // view: [partialTokens, head, headSize]

    auto const partialTokenNum = (batch_beam + mCpSize - 1) / mCpSize;
    int64_t hiddenSize = mNumAttnHeads * mHeadSize;
    // if (mIsMLAEnabled)
    // {
    //     hiddenSize = mNumAttnHeads * getHeadSize();
    // }

    // do all-to-all
#if ENABLE_MULTI_DEVICE
    size_t const elementNum = partialTokenNum * mHeadSize * mNumAttnHeads;
    ncclGroupStart();
    for (int cpIdx = 0; cpIdx < mCpSize; cpIdx++)
    {
        if (cpIdx != mCpRank)
        {
            if (mType == nvinfer1::DataType::kFP8)
            {
                NCCLCHECK(ncclSend(reinterpret_cast<__nv_fp8_e4m3 const*>(input) + cpIdx * elementNum, elementNum, ncclInt8,
                    cpIdx, *mNcclComm, stream));
                NCCLCHECK(ncclRecv(reinterpret_cast<__nv_fp8_e4m3*>(buffer) + cpIdx * elementNum, elementNum, ncclInt8,
                    cpIdx, *mNcclComm, stream));
            }
            else
            {
                NCCLCHECK(ncclSend(
                    input + cpIdx * elementNum, elementNum, (*getDtypeMap())[mType], cpIdx, *mNcclComm, stream));
                NCCLCHECK(ncclRecv(
                    buffer + cpIdx * elementNum, elementNum, (*getDtypeMap())[mType], cpIdx, *mNcclComm, stream));
            }
        }
    }
    ncclGroupEnd();
#endif // ENABLE_MULTI_DEVICE

    // do transpose_1_reverse
    if (mType == nvinfer1::DataType::kFP8)
    {
        invokeCpTransposeToSeqMajor<__nv_fp8_e4m3>(reinterpret_cast<__nv_fp8_e4m3*>(output),
            reinterpret_cast<__nv_fp8_e4m3 const*>(input), reinterpret_cast<__nv_fp8_e4m3 const*>(buffer),
            partialTokenNum, mCpSize, hiddenSize, mCpRank, stream);
    }
    else
    {
        invokeCpTransposeToSeqMajor<T>(
            (T*) output, input, buffer, partialTokenNum, mCpSize, hiddenSize, mCpRank, stream);
    }
    return 0;
}

#define INSTANTIATE_ulyssesContextCP2TP(T)                                                                     \
    template int UlyssesOp::ulyssesContextCP2TP<T>(T const* input, T* output, T* buffer, int batch_size, int const* host_context_lengths, \
        int const* context_lengths, int const* cu_q_seqlens, int const* cu_cp_partial_seqlens, int rank, int size, bool isPreprocess,     \
        cudaStream_t stream)
INSTANTIATE_ulyssesContextCP2TP(float);
INSTANTIATE_ulyssesContextCP2TP(half);
INSTANTIATE_ulyssesContextCP2TP(__nv_bfloat16);
INSTANTIATE_ulyssesContextCP2TP(__nv_fp8_e4m3);
#undef INSTANTIATE_ulyssesContextCP2TP

#define INSTANTIATE_ulyssesContextTP2CP(T)                                                                     \
    template int UlyssesOp::ulyssesContextTP2CP<T>(T const* input, T* output, T* buffer, int batch_size, int const* host_context_lengths, \
        int const* context_lengths, int const* cu_q_seqlens, int const* cu_cp_partial_seqlens, int rank, int size, bool isPreprocess,     \
        cudaStream_t stream)
INSTANTIATE_ulyssesContextTP2CP(float);
INSTANTIATE_ulyssesContextTP2CP(half);
INSTANTIATE_ulyssesContextTP2CP(__nv_bfloat16);
INSTANTIATE_ulyssesContextTP2CP(__nv_fp8_e4m3);
#undef INSTANTIATE_ulyssesContextTP2CP

#define INSTANTIATE_ulyssesGenerationPreprocess(T)                                                                     \
    template int UlyssesOp::ulyssesGenerationPreprocess<T>(T const* input, T* output, T* buffer, int32_t batch_beam, cudaStream_t stream)
INSTANTIATE_ulyssesGenerationPreprocess(float);
INSTANTIATE_ulyssesGenerationPreprocess(half);
INSTANTIATE_ulyssesGenerationPreprocess(__nv_bfloat16);
INSTANTIATE_ulyssesGenerationPreprocess(__nv_fp8_e4m3);
#undef INSTANTIATE_ulyssesGenerationPreprocess

#define INSTANTIATE_ulyssesGenerationPostprocess(T)                                                                     \
    template int UlyssesOp::ulyssesGenerationPostprocess<T>(T const* input, T* output, T* buffer, int32_t batch_beam, cudaStream_t stream)
INSTANTIATE_ulyssesGenerationPostprocess(float);
INSTANTIATE_ulyssesGenerationPostprocess(half);
INSTANTIATE_ulyssesGenerationPostprocess(__nv_bfloat16);
INSTANTIATE_ulyssesGenerationPostprocess(__nv_fp8_e4m3);
#undef INSTANTIATE_ulyssesGenerationPostprocess