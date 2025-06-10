# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import pickle
import sys
import traceback

import cloudpickle
import numpy as np
import pytest
import torch
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor
# from utils.util import skip_pre_blackwell

import tensorrt_llm
from tensorrt_llm._torch.distributed import (AllReduce, AllReduceFusionOp,
                                             AllReduceParams, MoEAllReduce)
# from tensorrt_llm._torch.modules.linear import Linear, TensorParallelMode
# from tensorrt_llm._torch.modules.rms_norm import RMSNorm
# from tensorrt_llm.mapping import Mapping


sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(
    cloudpickle.dumps,
    cloudpickle.loads,
    pickle.HIGHEST_PROTOCOL,
)

# TODO: refine arguments
def run_ulysses_single_rank(rank):
    parallel_size = 2
    seq_lens = [16, 32, 64] # TODO加不能整除的情况
    batch_size = len(seq_lens)
    token_num = sum(seq_lens)
    partial_seq_lens = [(seq_len + parallel_size - 1) // parallel_size for seq_len in seq_lens]

    input_tensor = torch.randn(token_num, 1024, dtype=torch.half, device="cuda")
    # TODO: create buffer inside of OP (not equal size as input_tensor)
    output_tensor = torch.randn(token_num, 1024, dtype=torch.half, device="cuda")
    # TODO: create buffer inside of OP
    buffer_tensor = torch.randn(token_num, 1024, dtype=torch.half, device="cuda")

    host_context_lengths = torch.tensor(seq_lens, dtype=torch.int32)
    context_lengths = host_context_lengths.cuda()
    cu_q_seqlens = torch.tensor([0] + np.cumsum(seq_lens).tolist(), dtype=torch.int32, device="cuda")
    cu_cp_partial_seqlens = torch.tensor([0] + np.cumsum(partial_seq_lens).tolist(), dtype=torch.int32, device="cuda")
    # print("host_context_lengths", host_context_lengths)
    # print("context_lengths", context_lengths)
    # print("cu_q_seqlens", cu_q_seqlens)
    # print("cu_cp_partial_seqlens", cu_cp_partial_seqlens)

    # test CP Attn
    ulysses_comm_params_dict = {
        "tp_size": parallel_size,
        "tp_rank": rank,
        "cp_size": 1,
        "cp_rank": 0,
        "attn_tp_size": 1,
        "attn_cp_size": parallel_size,
        "comm_group": [0, 1],
        "is_preprocess": True,
        "batch_size": batch_size,
    }
    # context phase
    ulysses_model_params_dict = {
        "head_size": 128,
        "num_heads": 16,
        "num_attn_heads": 16,
        "num_attn_kv_heads": 16,
        "num_kv_heads_origin": 16,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "is_mla_enable": True,
    }
    from torch.ops import trtllm as trtllmOps
    # preprocess: TP2CP
    middle_tensor = trtllmOps.ulysses(
        input_tensor,
        output_tensor,
        buffer_tensor,
        host_context_lengths,
        context_lengths,
        cu_q_seqlens,
        cu_cp_partial_seqlens,
        ulysses_comm_params_dict,
        ulysses_model_params_dict,
    )

    ulysses_comm_params_dict = {
        "tp_size": 1,
        "tp_rank": 0,
        "cp_size": parallel_size,
        "cp_rank": rank,
        "attn_tp_size": parallel_size,
        "attn_cp_size": 1,
        "comm_group": [0, 1],
        "is_preprocess": True,
        "batch_size": batch_size,
    }
    output = trtllmOps.ulysses(
        middle_tensor,
        output_tensor,
        buffer_tensor,
        host_context_lengths,
        context_lengths,
        cu_q_seqlens,
        cu_cp_partial_seqlens,
        ulysses_comm_params_dict,
        ulysses_model_params_dict,
    )
    assert output is not None
    return True

def run_single_rank(func):
    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    try:
        func(rank)
    except Exception:
        traceback.print_exc()
        raise
    return True

@pytest.mark.skipif(torch.cuda.device_count() < 2,
                    reason="Requires at least 2 GPUs for this test")
def test_ulysses():
    parallel_size = 2
    with MPIPoolExecutor(max_workers=parallel_size) as executor:
        results = executor.map(run_single_rank, [run_ulysses_single_rank for _ in range(parallel_size)])
        for r in results:
            assert r is True

if __name__ == "__main__":
    test_ulysses()
