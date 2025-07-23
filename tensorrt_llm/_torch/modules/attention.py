import math
import weakref
from typing import Optional, Union, cast

import torch
from torch import nn

from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

from ..attention_backend import (AttentionInputType, AttentionMetadata,
                                 TrtllmAttention, TrtllmAttentionMetadata)
from ..attention_backend.interface import (PositionalEmbeddingParams,
                                           PredefinedAttentionMask)
from ..attention_backend.utils import create_attention, get_attention_backend
from ..distributed import AllReduceParams
from ..model_config import ModelConfig
from ..peft.lora.layer import LoraLayer, LoraModuleType
from ..utils import Fp4QuantizedTensor, get_model_extra_attrs
from .linear import Linear, TensorParallelMode, WeightMode, WeightsLoadingConfig
from .multi_stream_utils import maybe_execute_in_parallel
from .rms_norm import RMSNorm
from .rotary_embedding import RotaryEmbedding

import os
FLASH_DECODING = os.getenv("FLASH_DECODING", "0")
czq_idx = 0
class Attention(nn.Module):

    def __init__(
        self,
        *,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        max_position_embeddings: int,
        bias: bool,
        pos_embd_params: Optional[PositionalEmbeddingParams] = None,
        rope_fusion: Optional[bool] = None,
        layer_idx: Optional[int] = None,
        dtype: torch.dtype = None,
        dense_bias: Optional[bool] = None,
        config: Optional[ModelConfig] = None,
        q_scaling: float = 1.0,
        attention_chunk_size: Optional[int] = None,
    ):
        """
        Initialize the Attention module.

        Args:
            hidden_size (int): The size of the hidden dimension.
            num_attention_heads (int): The number of attention heads.
            num_key_value_heads (int): The number of key value heads.
            max_position_embeddings (int): The maximum position embeddings.
            bias (bool): Whether to use bias in the linear layers.
            pos_embd_params (Optional[PositionalEmbeddingParams]): The positional embedding parameters.
            rope_fusion (Optional[bool]): Whether to fuse RoPE into the attention OP and skip applying unfused RoPE. If None, whether to fuse is decided by the capability of the attention backend.
            layer_idx (Optional[int]): The layer index.
            dtype (torch.dtype): The data type.
            dense_bias (Optional[bool]): Whether to use bias in the output projection layer.
            config (Optional[ModelConfig]): The model configuration.
            q_scaling (float): The scaling factor for the qk_scale. The definition is $O = softmax(QK^T * qk_scale) * V, qk_scale = 1 / (sqrt(head_dim) * q_scaling)$. The default value is 1.0.
            attention_chunk_size (Optional[int]): See [Chunked Attention] below.
        """
        super().__init__()
        self.layer_idx = layer_idx

        config = config or ModelConfig()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.head_dim = getattr(config.pretrained_config, "head_dim",
                                self.hidden_size // self.num_heads)
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = max_position_embeddings
        self.pos_embd_params = pos_embd_params
        self.dense_bias = dense_bias
        self.q_scaling = q_scaling

        # [Chunked Attention]
        # Chunked attention is applied to context requests only. Chunked attention will be
        # applied when this field is specified and mMaskType == CAUSAL.
        #
        # In chunked attention, we break context requests into chunks of a specified size. Tokens can only
        # attend to tokens in the same chunk. So, for example, if the chunk size is 3, we might have a mask
        # that looks like this:
        #
        # 1 0 0 0 0 0
        # 1 1 0 0 0 0
        # 1 1 1 0 0 0
        # 0 0 0 1 0 0
        # 0 0 0 1 1 0
        # 0 0 0 1 1 1
        self.attention_chunk_size = attention_chunk_size

        if dense_bias is None:
            self.dense_bias = bias

        # tensor parallel
        tp_size = config.mapping.tp_size
        pp_size = config.mapping.pp_size
        if config.mapping.enable_attention_dp:
            tp_size = 1

        mapping = Mapping(
            world_size=tp_size * pp_size,
            tp_size=tp_size,
            pp_size=pp_size,
            rank=config.mapping.rank,
            gpus_per_node=config.mapping.gpus_per_node,
            enable_attention_dp=config.mapping.enable_attention_dp,
        )
        assert self.num_heads % tp_size == 0
        self.num_heads = self.num_heads // tp_size
        self.num_key_value_heads = (self.num_key_value_heads + tp_size -
                                    1) // tp_size
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_key_value_heads * self.head_dim

        self.qkv_proj = Linear(
            self.hidden_size,
            tp_size * self.q_size + 2 * tp_size * self.kv_size,
            bias=bias,
            dtype=dtype,
            mapping=mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            weights_loading_config=WeightsLoadingConfig(
                weight_mode=WeightMode.FUSED_QKV_LINEAR),
            quant_config=config.get_quant_config(),
            skip_create_weights_in_init=config.skip_create_weights_in_init,
            allreduce_strategy=config.allreduce_strategy,
            force_dynamic_quantization=config.force_dynamic_quantization)
        self.o_lora = LoraLayer([LoraModuleType.ATTENTION_DENSE],
                                [self.hidden_size])

        self.o_proj = Linear(
            tp_size * self.q_size,
            self.hidden_size,
            bias=self.dense_bias,
            dtype=dtype,
            mapping=mapping,
            tensor_parallel_mode=TensorParallelMode.ROW,
            quant_config=config.get_quant_config(),
            skip_create_weights_in_init=config.skip_create_weights_in_init,
            lora=self.o_lora,
            allreduce_strategy=config.allreduce_strategy,
            force_dynamic_quantization=config.force_dynamic_quantization)

        self.quant_config = config.get_quant_config()
        self.attn_backend = config.attn_backend
        attn_cls = get_attention_backend(self.attn_backend)

        # These two modules are mutually exclusive - either splitted_qkv_lora or fused_qkv_lora will be used,
        # but never both at the same time. splitted_qkv_lora handles Q,K,V separately while fused_qkv_lora
        # handles them as a single fused operation.
        self.splitted_qkv_lora = LoraLayer([
            LoraModuleType.ATTENTION_Q, LoraModuleType.ATTENTION_K,
            LoraModuleType.ATTENTION_V
        ], [self.q_size, self.kv_size, self.kv_size])
        self.fused_qkv_lora = LoraLayer([LoraModuleType.ATTENTION_QKV],
                                        [self.q_size + 2 * self.kv_size])

        self.o_lora = LoraLayer([LoraModuleType.ATTENTION_DENSE],
                                [self.hidden_size])

        # Whether to fuse RoPE into the attention OP.
        # If true, RoPE will be applied in self.attn.forward.
        # If false, RoPE will be applied in self.apply_rope.
        self.rope_fusion = rope_fusion
        if self.rope_fusion and not attn_cls.support_fused_rope():
            logger.warning(
                "rope_fusion is true but the attention backend does not support it. Will disable rope_fusion."
            )
            self.rope_fusion = False
        # If rope_fusion is not specified, enable if the attention backend supports it.
        if self.rope_fusion is None:
            self.rope_fusion = attn_cls.support_fused_rope()

        self.rotary_emb = None
        if not self.rope_fusion and self.pos_embd_params is not None:
            self.rotary_emb = RotaryEmbedding(
                self.pos_embd_params.rope,
                head_dim=self.head_dim,
                is_neox=self.pos_embd_params.is_neox,
            )

        self.attn = create_attention(
            self.attn_backend,
            self.layer_idx,
            self.num_heads,
            self.head_dim,
            self.num_key_value_heads,
            pos_embd_params=self.pos_embd_params if self.rope_fusion else None,
            quant_config=self.quant_config,
            skip_create_weights_in_init=config.skip_create_weights_in_init,
            q_scaling=self.q_scaling,
            attention_chunk_size=self.attention_chunk_size,
        )

        self.support_fused_qkv = self.attn.support_fused_qkv()
        self.support_nvfp4_output = self.attn.support_nvfp4_output()

        if not config.skip_create_weights_in_init:
            self.create_weights()

    def create_weights(self):
        # self.attn has no weights but has states that are related to quant_config,
        # which could be modified after __init__
        self.attn.update_quant_config(self.quant_config)

    def split_qkv(self, q, k=None, v=None):
        if k is None and v is None:
            q, k, v = q.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        return q, k, v

    def convert_qkv(self, q, k, v):
        if k is None and v is None and not self.support_fused_qkv:
            q, k, v = self.split_qkv(q)
        elif k is not None and v is not None and self.support_fused_qkv:
            qkv = torch.concat([q, k, v], dim=-1)
            q, k, v = qkv, None, None
        return q, k, v

    def forward(
        self,
        position_ids: Optional[torch.IntTensor],
        hidden_states: Union[torch.Tensor, Fp4QuantizedTensor],
        attn_metadata: AttentionMetadata,
        attention_mask: PredefinedAttentionMask = PredefinedAttentionMask.
        CAUSAL,
        mrope_config: Optional[dict] = None,
        all_reduce_params: Optional[AllReduceParams] = None,
        lora_params: Optional[dict] = None,
        attention_window_size: Optional[int] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass for the Attention module.

        Args:
            position_ids (Optional[torch.IntTensor]): The position IDs.
            hidden_states (torch.Tensor): The hidden states.
            attn_metadata (AttentionMetadata): The attention metadata.
            attention_mask (PredefinedAttentionMask): The attention mask type.
            mrope_config (Optional[dict]): The MROPE configuration.
            all_reduce_params (Optional[AllReduceParams]): The all reduce parameters.
            lora_params (Optional[dict]): The LoRA parameters.
            attention_window_size (Optional[int]): The attention window size.

        Returns:
            torch.Tensor: The output tensor.
        """
        qkv = self.qkv_proj(hidden_states)

        if bool(lora_params):
            qkv_lora = self.splitted_qkv_lora(hidden_states, lora_params,
                                              self.layer_idx)
            if qkv_lora is not None:
                qkv = qkv + qkv_lora

            qkv_lora = self.fused_qkv_lora(hidden_states, lora_params,
                                           self.layer_idx)
            if qkv_lora is not None:
                qkv = qkv + qkv_lora

        q, k, v = qkv, None, None

        q, k, v = self.apply_rope(q, k, v, position_ids)

        out_scale = None
        out_scale_sf = None
        if self.o_proj.has_fp8_qdq or self.o_proj.has_nvfp4 or self.o_proj.has_fp8_block_scales or self.o_proj.has_fp8_rowwise:
            out_scale = self.o_proj.inv_input_scale
        if self.o_proj.has_nvfp4 and self.support_nvfp4_output:
            out_scale_sf = self.o_proj.input_scale

        q, k, v = self.convert_qkv(q, k, v)
        attn_output = self.attn.forward(
            q,
            k,
            v,
            attn_metadata,
            out_scale=out_scale,
            out_scale_sf=out_scale_sf,
            attention_mask=attention_mask,
            mrope_config=mrope_config,
            attention_window_size=attention_window_size)
        hidden_states = attn_output
        attn_output = self.o_proj(attn_output,
                                  all_reduce_params=all_reduce_params,
                                  lora_params=lora_params,
                                  layer_idx=self.layer_idx)
        return attn_output

    def apply_rope(self, q: torch.Tensor, k: Optional[torch.Tensor],
                   v: Optional[torch.Tensor], position_ids: torch.Tensor):
        """
        Apply RoPE to the query and key.
        Depending on the implementation, q, k, v could be either fused (q, k, v = concat(q, k, v), None, None) or unfused (none of q, k, v is None).
        Before self.attn.forward, convert_qkv will be called to make sure that the format of (q, k, v) satisfies the requirement of self.attn.
        This method could be overridden in the subclass, in which extra functionalities such as q_norm/k_norm could be added.
        Args:
            q (torch.Tensor): The query tensor.
            k (Optional[torch.Tensor]): The key tensor.
            v (Optional[torch.Tensor]): The value tensor.
            position_ids (torch.Tensor): The position IDs of each token for RoPE.
        Returns:
            tuple: A tuple of (q, k, v).
        """
        q, k, v = self.split_qkv(q, k, v)
        # If RoPE is fused into the attention OP, do not apply RoPE here.
        if not self.rope_fusion and position_ids is not None:
            q, k = self.rotary_emb(position_ids, [q, k])
        return q, k, v


def extract_extra_attrs(layer_idx: str):
    extra_attrs = get_model_extra_attrs()
    assert extra_attrs is not None, "Model extra attrs is not set"

    metadata_ref = extra_attrs.get("attention_metadata", None)
    assert metadata_ref is not None, "Attention metadata is not set"
    metadata = metadata_ref()
    assert isinstance(
        metadata,
        TrtllmAttentionMetadata,
    )

    mla_layers = extra_attrs.get("mla_layers", None)
    assert mla_layers is not None, "MLA layers is not registered"
    mla_layer_ref = mla_layers.get(layer_idx, None)
    assert mla_layer_ref is not None, f"Cannot find MLA layer for layer {layer_idx}"
    mla_layer = mla_layer_ref()
    assert isinstance(
        mla_layer,
        MLA), "MLA layer must be a subclass of MLA or an instance of MLA"

    return metadata, mla_layer


@torch.library.custom_op("trtllm::mla_custom_op_inplace",
                         mutates_args=("output", ))
def mla_custom_op_inplace(
    hidden_states: torch.Tensor,
    position_ids: Optional[torch.Tensor],
    layer_idx: str,
    output: torch.Tensor,
) -> None:
    metadata, mla_layer = extract_extra_attrs(layer_idx)
    mla_layer.forward_impl(position_ids, hidden_states, metadata, output=output)


def fp8_block_scaling_bmm_out(
    mat1: torch.Tensor,
    mat2_fp8: torch.Tensor,
    mat2_scale: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    sm_version = get_sm_version()
    if sm_version == 90:
        mat1_fp8, mat1_scale = torch.ops.trtllm.fp8_batched_quantize_1x128_permute102(
            mat1)
        torch.ops.trtllm.fp8_block_scaling_bmm_out(mat1_fp8, mat2_fp8,
                                                   mat1_scale, mat2_scale, out)
    elif sm_version == 100:
        low_latency = True
        use_deep_seek_fp8 = True
        tile_size = 8
        epilogue_tile_m = 64 if use_deep_seek_fp8 else 128
        m_size = mat1.shape[0]
        if m_size % tile_size != 0:
            tiled_shape = ((m_size + tile_size - 1) // tile_size) * tile_size
            mat1 = torch.nn.functional.pad(
                mat1, (0, 0, 0, 0, 0, tiled_shape - m_size), "constant", 0)

        mat1_fp8, mat1_scale = torch.ops.trtllm.fp8_batched_quantize_1x128_permute102(
            mat1)
        output, output_sf = torch.ops.trtllm.fp8_batched_gemm_trtllmgen(
            mat1_fp8,
            mat2_fp8,
            tile_size=tile_size,
            epilogue_tile_m=epilogue_tile_m,
            use_deep_seek_fp8=use_deep_seek_fp8,
            low_latency=low_latency,
            dq_sfs_a=mat1_scale.reshape(mat1.shape[-1] // 128, -1),
            dq_sfs_b=mat2_scale,
            out_dtype=out.dtype,
        )
        out.copy_(output[:, :m_size])
    else:
        raise NotImplementedError(f"SM{sm_version} is not supported")


class MLA(nn.Module):

    def __init__(
        self,
        *,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        predicted_tokens_per_seq: int,
        max_position_embeddings: int,
        bias: bool,
        aux_stream: Optional[torch.cuda.Stream] = None,
        pos_embd_params: Optional[PositionalEmbeddingParams] = None,
        layer_idx: Optional[int] = None,
        dtype: torch.dtype = None,
        dense_bias: Optional[bool] = None,
        config: Optional[ModelConfig] = None,
    ):
        """
        Initialize the MLA module.

        Args:
            hidden_size (int): The size of the hidden dimension.
            num_attention_heads (int): The number of attention heads.
            num_key_value_heads (int): The number of key value heads.
            qk_nope_head_dim (int): The dimension of the query and key without Rope.
            qk_rope_head_dim (int): The dimension of the Rope of query and key.
            v_head_dim (int): The dimension of the value.
            q_lora_rank (int): The dimension of the compressed query.
            kv_lora_rank (int): The dimension of the compressed key and value.
            predicted_tokens_per_seq (int): The number of predicted tokens per sequence.
            max_position_embeddings (int): The maximum position embeddings.
            bias (bool): Whether to use bias in the linear layers.
            aux_stream (Optional[torch.cuda.Stream]): The auxiliary CUDA stream for running operations in two parallel streams.
            pos_embd_params (PositionalEmbeddingParams): The positional embedding parameters.
            layer_idx (int): The layer index.
            dtype (torch.dtype): The data type.
            dense_bias (bool): Whether to use bias in the output projection layer.
            config (ModelConfig): The model configuration.
        """
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_idx_str = str(layer_idx)
        self.dtype = dtype

        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.predicted_tokens_per_seq = predicted_tokens_per_seq
        self.max_position_embeddings = max_position_embeddings
        self.pos_embd_params = pos_embd_params
        self.dense_bias = dense_bias
        if dense_bias is None:
            self.dense_bias = bias

        if self.q_lora_rank is None:
            self.q_lora_rank = hidden_size
            self.is_lite = True
        else:
            self.is_lite = False

        assert pos_embd_params is not None, "pos_embd_params must be provided in MLA"

        self.register_to_config = False
        if config is not None:
            if "mla_layers" not in config.extra_attrs:
                config.extra_attrs["mla_layers"] = {}
            config.extra_attrs["mla_layers"][self.layer_idx_str] = weakref.ref(
                self)
            self.register_to_config = True

        # tensor parallel
        config = config or ModelConfig()
        tp_size = config.mapping.tp_size
        pp_size = config.mapping.pp_size
        if config.mapping.enable_attention_dp:
            tp_size = 1

        mapping = Mapping(
            world_size=tp_size * pp_size,
            tp_size=tp_size,
            pp_size=pp_size,
            rank=config.mapping.rank,
            gpus_per_node=config.mapping.gpus_per_node,
            enable_attention_dp=config.mapping.enable_attention_dp,
        )
        self.tp_size = mapping.tp_size
        self.tp_rank = mapping.tp_rank
        self.mapping = mapping

        assert self.num_heads % tp_size == 0
        self.global_num_heads = self.num_heads
        self.num_heads = self.num_heads // tp_size
        self.num_key_value_heads = (self.num_key_value_heads + tp_size -
                                    1) // tp_size

        rms_norm_eps = config.pretrained_config.rms_norm_eps
        quant_config = config.get_quant_config()
        self.quant_config = quant_config

        if not self.is_lite:
            self.fused_a = Linear(
                hidden_size,
                self.q_lora_rank + self.kv_lora_rank + self.qk_rope_head_dim,
                bias=bias,
                dtype=dtype,
                quant_config=quant_config,
                skip_create_weights_in_init=config.skip_create_weights_in_init,
                use_custom_cublas_mm=True,
                force_dynamic_quantization=config.force_dynamic_quantization)

            self.q_a_layernorm = RMSNorm(hidden_size=self.q_lora_rank,
                                         eps=rms_norm_eps,
                                         dtype=dtype)

            self.q_b_proj = Linear(
                self.q_lora_rank,
                tp_size * self.num_heads * self.qk_head_dim,
                bias=bias,
                dtype=dtype,
                mapping=mapping,
                tensor_parallel_mode=TensorParallelMode.COLUMN,
                quant_config=quant_config,
                skip_create_weights_in_init=config.skip_create_weights_in_init,
                allreduce_strategy=config.allreduce_strategy,
                force_dynamic_quantization=config.force_dynamic_quantization)
        else:
            self.fused_a = Linear(
                hidden_size,
                self.kv_lora_rank + self.qk_rope_head_dim,
                bias=bias,
                dtype=dtype,
                quant_config=quant_config,
                skip_create_weights_in_init=config.skip_create_weights_in_init,
                use_custom_cublas_mm=True,
                force_dynamic_quantization=config.force_dynamic_quantization)

            self.q_proj = Linear(
                self.q_lora_rank,
                tp_size * self.num_heads * self.qk_head_dim,
                bias=bias,
                dtype=dtype,
                mapping=mapping,
                tensor_parallel_mode=TensorParallelMode.COLUMN,
                quant_config=quant_config,
                skip_create_weights_in_init=config.skip_create_weights_in_init,
                allreduce_strategy=config.allreduce_strategy,
                force_dynamic_quantization=config.force_dynamic_quantization)
            self.q_b_proj = self.q_proj

        self.kv_a_layernorm = RMSNorm(hidden_size=kv_lora_rank,
                                      dtype=dtype,
                                      eps=rms_norm_eps)

        self.kv_b_proj = Linear(
            self.kv_lora_rank,
            tp_size * self.num_heads *
            (self.qk_nope_head_dim + self.v_head_dim),
            bias=bias,
            dtype=dtype,
            mapping=mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            quant_config=quant_config,
            skip_create_weights_in_init=config.skip_create_weights_in_init,
            allreduce_strategy=config.allreduce_strategy,
            force_dynamic_quantization=config.force_dynamic_quantization)
        # This parameter will view into self.kv_b_proj.weight after loading weights.
        # For dummy weight initialization, this parameter is initialized with empty tensor.
        # Used in forward_generation only
        self.v_b_proj = nn.Parameter(
            torch.empty(
                (self.num_heads, self.v_head_dim, self.kv_lora_rank),
                dtype=dtype,
            ),
            requires_grad=False,
        )

        self.o_proj = Linear(
            self.num_key_value_heads * self.v_head_dim * tp_size,
            self.hidden_size,
            bias=self.dense_bias,
            dtype=dtype,
            mapping=mapping,
            tensor_parallel_mode=TensorParallelMode.ROW,
            quant_config=quant_config,
            skip_create_weights_in_init=config.skip_create_weights_in_init,
            allreduce_strategy=config.allreduce_strategy,
            force_dynamic_quantization=config.force_dynamic_quantization)

        def yarn_get_mscale(scale=1, mscale=1):
            if scale <= 1:
                return 1.0
            return 0.1 * mscale * math.log(scale) + 1.0

        mscale_all_dim = pos_embd_params.rope.mscale_all_dim
        scaling_factor = pos_embd_params.rope.scale
        mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
        q_scaling = 1.0 / (mscale * mscale)

        self.mha = create_attention(
            config.attn_backend,
            self.layer_idx,
            self.num_heads,
            head_dim=self.qk_head_dim,
            num_kv_heads=self.num_key_value_heads,
            pos_embd_params=pos_embd_params,
            quant_config=quant_config,
            q_scaling=q_scaling,
            is_mla_enable=True,
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            v_head_dim=self.v_head_dim,
            predicted_tokens_per_seq=self.predicted_tokens_per_seq,
            skip_create_weights_in_init=config.skip_create_weights_in_init,
        )

        num_heads_gen = self.num_heads
        global FLASH_DECODING
        if FLASH_DECODING == "1":
            num_heads_gen = self.global_num_heads
        print(f"czq target3 {num_heads_gen}")
        self.mqa = create_attention(
            config.attn_backend,
            self.layer_idx,
            num_heads_gen,
            head_dim=self.kv_lora_rank + self.qk_rope_head_dim,
            num_kv_heads=1,
            pos_embd_params=pos_embd_params,
            quant_config=quant_config,
            q_scaling=q_scaling,
            is_mla_enable=True,
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            v_head_dim=self.kv_lora_rank,
            predicted_tokens_per_seq=self.predicted_tokens_per_seq,
            skip_create_weights_in_init=config.skip_create_weights_in_init,
        )

        self.aux_stream = aux_stream
        self.ln_events = [torch.cuda.Event(), torch.cuda.Event()]

        self.rope_fusion = self.mha.support_fused_rope()
        self.support_fused_qkv = self.mha.support_fused_qkv()
        self.rotary_emb = None
        self.apply_rotary_emb = not self.rope_fusion
        if self.apply_rotary_emb:
            self.rotary_emb = RotaryEmbedding(
                pos_embd_params.rope,
                head_dim=self.qk_rope_head_dim,
                is_neox=pos_embd_params.is_neox,
            )

        if not config.skip_create_weights_in_init:
            self.create_weights()

    def create_weights(self):
        # self.mha/mqa has no weights but has states that are related to quant_config,
        # which could be modified after __init__
        self.mha.update_quant_config(self.quant_config)
        self.mqa.update_quant_config(self.quant_config)

        # k_b_proj_trans's dtype must be consistent with self.kv_b_proj,
        # which can be modified after __init__
        has_fp8_block_scales = (
            self.kv_b_proj.quant_config
            and self.kv_b_proj.quant_config.quant_mode.has_fp8_block_scales())

        mla_weight_dtype = torch.float8_e4m3fn if has_fp8_block_scales else self.dtype
        self.k_b_proj_trans = nn.Parameter(
            torch.empty(
                (self.num_heads, self.kv_lora_rank, self.qk_nope_head_dim),
                dtype=mla_weight_dtype,
            ),
            requires_grad=False,
        )

        if has_fp8_block_scales:
            self.k_b_proj_trans_scale = nn.Parameter(
                torch.empty(
                    (
                        self.num_heads,
                        self.kv_lora_rank // 128,
                        self.qk_nope_head_dim // 128,
                    ),
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            # This parameter will view into self.kv_b_proj.weight_scale after loading weights.
            # For dummy weight initialization, this parameter is initialized with empty tensor.
            self.v_b_proj_scale = nn.Parameter(
                torch.empty(
                    (
                        self.num_heads,
                        self.v_head_dim // 128,
                        self.kv_lora_rank // 128,
                    ),
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
        else:
            self.k_b_proj_trans_scale = None
            self.v_b_proj_scale = None

    def apply_rope(
        self,
        q: torch.Tensor,
        k_pe: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        q = q.view(-1, self.num_heads, self.qk_head_dim)
        q_pe = q[..., self.qk_nope_head_dim:].reshape(
            -1, self.num_heads * self.qk_rope_head_dim)
        q_pe, k_pe = self.rotary_emb(position_ids, [q_pe, k_pe])
        q[..., self.qk_nope_head_dim:] = q_pe.view(-1, self.num_heads,
                                                   self.qk_rope_head_dim)
        return k_pe

    def create_output(self, hidden_states: torch.Tensor):
        num_tokens = hidden_states.shape[0]
        hidden_size = self.o_proj.in_features
        return hidden_states.new_empty([num_tokens, hidden_size],
                                       dtype=hidden_states.dtype)

    def forward_impl(self,
                     position_ids: Optional[torch.Tensor],
                     hidden_states: torch.Tensor,
                     attn_metadata: AttentionMetadata,
                     output: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for the MLA module.

        Args:
            position_ids (Optional[torch.IntTensor]): The position IDs.
            hidden_states (torch.Tensor): The hidden states.
            attn_metadata (AttentionMetadata): The attention metadata.
            all_reduce_params (Optional[AllReduceParams]): The all reduce parameters.

        Returns:
            torch.Tensor: The output tensor.
        """
        if self.is_lite:
            compressed_kv, k_pe = self.fused_a(hidden_states).split(
                [self.kv_lora_rank, self.qk_rope_head_dim], -1)
            compressed_kv = self.kv_a_layernorm(compressed_kv)
            q = hidden_states
        else:
            q, compressed_kv, k_pe = self.fused_a(hidden_states).split(
                [self.q_lora_rank, self.kv_lora_rank, self.qk_rope_head_dim],
                -1)

            q, compressed_kv = maybe_execute_in_parallel(
                lambda: self.q_a_layernorm(q),
                lambda: self.kv_a_layernorm(compressed_kv),
                self.ln_events[0],
                self.ln_events[1],
                self.aux_stream,
            )

        q, latent_cache = maybe_execute_in_parallel(
            lambda: self.q_b_proj(q),
            lambda: torch.concat([compressed_kv, k_pe], dim=-1),
            self.ln_events[0],
            self.ln_events[1],
            self.aux_stream,
        )

        # split q, k, v into context and gen batches
        num_contexts = attn_metadata.num_contexts
        num_generations = attn_metadata.num_generations
        num_ctx_tokens = attn_metadata.num_ctx_tokens
        num_tokens = attn_metadata.num_tokens

        assert q.shape[
            0] == num_tokens, f"Expect q.shape[0] to be {num_tokens}, but got {q.shape[0]}"

        if num_contexts > 0:
            q_ctx = q[:num_ctx_tokens, ...]
            compressed_kv_ctx = compressed_kv[:num_ctx_tokens, ...]
            k_pe_ctx = k_pe[:num_ctx_tokens, ...]
            latent_cache_ctx = latent_cache[:num_ctx_tokens, ...]
            if self.apply_rotary_emb:
                assert position_ids is not None
                k_pe_ctx = self.apply_rope(q_ctx, k_pe_ctx, position_ids)

            attn_output_context = self.forward_context(
                q_ctx,
                compressed_kv_ctx,
                k_pe_ctx,
                attn_metadata,
                latent_cache_ctx,
                output=output if num_generations == 0 else None)
            if num_generations == 0:
                return attn_output_context
        else:
            attn_output_context = None

        if num_generations > 0:
            q_gen = q[num_ctx_tokens:, ...]
            compressed_kv_gen = compressed_kv[num_ctx_tokens:, ...]
            k_pe_gen = k_pe[num_ctx_tokens:, ...]
            latent_cache_gen = latent_cache[num_ctx_tokens:, ...]
            if self.apply_rotary_emb:
                assert position_ids is not None
                k_pe_gen = self.apply_rope(q_gen, k_pe_gen, position_ids)

            attn_output_gen = self.forward_generation(
                q_gen,
                compressed_kv_gen,
                k_pe_gen,
                attn_metadata,
                latent_cache_gen,
                output=output if num_contexts == 0 else None)
            if num_contexts == 0:
                return attn_output_gen
        else:
            attn_output_gen = None

        # release pytorch activation memory
        q = None
        compressed_kv = None
        k_pe = None

        assert attn_output_context is not None and attn_output_gen is not None
        assert (
            len(attn_output_context.shape) == 2
        ), f"attn_output_context must be rank 2, not {len(attn_output_context.shape)}"
        assert (
            len(attn_output_gen.shape) == 2
        ), f"attn_output_gen must be rank 2, not {len(attn_output_gen.shape)}"
        output = output if output is not None else torch.empty(
            (num_tokens, attn_output_context.shape[1]),
            dtype=attn_output_context.dtype,
            device=attn_output_context.device)
        output[:attn_output_context.shape[0], :] = attn_output_context
        output[attn_output_context.shape[0]:, :] = attn_output_gen
        attn_output_context = None
        attn_output_gen = None
        return output

    def _maybe_concat_qkv(self, q, k, v):
        if k is not None and v is not None and self.support_fused_qkv:
            qkv = torch.concat([q, k, v], dim=-1)
            q, k, v = qkv, None, None
        return q, k, v

    def forward_context_for_flash_decoding(
            self,
            q: torch.Tensor,
            compressed_kv: torch.Tensor,
            latent_cache: Optional[torch.Tensor],
            attn_metadata: AttentionMetadata,
            output: Optional[torch.Tensor] = None) -> torch.Tensor:
        global czq_idx
        czq_idx += 1
        print(f"czq_idx: {czq_idx}")
        TARGET = 91 #91
        ##############
        # 0. context 部分改动只是为了快速得到存部分kv效果，具体改法还需商榷

        # 1. 存全量latent cache 计算全量q_pe k_pe, 主要是为了全量k_pe
        # apply RoPE, append compressed_kv + k_pe to paged kv cache and assign q_pe to q
        origin_latent_cache = latent_cache.clone()
        trtllm_attention = cast(TrtllmAttention, self.mha)
        trtllm_attention.mla_rope_append_paged_kv_assign_q(
            q, latent_cache, attn_metadata)

        k_pe = latent_cache[:, -self.qk_rope_head_dim:]

        # print(f"czq rank {self.tp_rank} get origin kv: {latent_cache[:,:512].sum(dim=1)}")

        # if czq_idx == TARGET and self.tp_rank == 0:
        #     # print(f"full_compressed_kv: {full_compressed_kv.shape}")
        #     import pdb; pdb.set_trace()

        # if self.rope_fusion:
        #     # copy full_compressed_kv and full_k_pe from paged kv cache
        #     full_compressed_kv, k_pe = trtllm_attention.load_paged_kv_cache_for_mla(
        #         attn_metadata, q.dtype)
        #     # print(f"czq rank {self.tp_rank} get kv: {full_compressed_kv.sum(dim=1)}")
        # if czq_idx == TARGET and self.tp_rank == 0:
        #     # print(f"full_compressed_kv: {full_compressed_kv.shape}")
        #     import pdb; pdb.set_trace()
        #     print(f"czq rank {self.tp_rank} get origin kv: {latent_cache[:,:512].sum(dim=1)}")

        ##############重新写入partial latent cache 此处重复操作
        num_contexts = attn_metadata.num_contexts
        seq_global_start = attn_metadata.ctx_kv_indptr
        seq_lens = seq_global_start[1:num_contexts+1] - seq_global_start[0:num_contexts]

        partial_latent_cache_chunks = []
        parital_latent_cache_chunks_seq_lens = [0]
        for i, seq_len in enumerate(seq_lens):
            start = seq_global_start[i]
            end = start + seq_len
            partial_latent_cache_chunk = torch.chunk(origin_latent_cache[start:end, ...], self.tp_size)[self.tp_rank]
            partial_latent_cache_chunks.append(partial_latent_cache_chunk)
            # parital_latent_cache_chunks_seq_len_acc = parital_latent_cache_chunks_seq_lens[-1] if parital_latent_cache_chunks_seq_lens else 0
            # parital_latent_cache_chunks_seq_lens.append(partial_latent_cache_chunk.shape[0] + parital_latent_cache_chunks_seq_len_acc)
            parital_latent_cache_chunks_seq_lens.append(partial_latent_cache_chunk.shape[0] + parital_latent_cache_chunks_seq_lens[-1])
        latent_cache_chunks_seq_lens = torch.tensor(parital_latent_cache_chunks_seq_lens, dtype=seq_global_start.dtype, device=seq_global_start.device)
        partial_latent_cache = torch.cat(partial_latent_cache_chunks, dim=0)
        print(f"czq rank {self.tp_rank} get seq_lens: {seq_lens}")
        print(f"czq rank {self.tp_rank} get latent_cache_chunks_seq_lens: {latent_cache_chunks_seq_lens}")
        print(f"czq rank {self.tp_rank} get partial_latent_cache: {partial_latent_cache.shape}")

        # 里面k_pe index可能不对，还是搞一个直接平移拷贝的kernel 或者 在generation中get出来然后slice
        setattr(attn_metadata, "ctx_kv_seq_lens", latent_cache_chunks_seq_lens)
        trtllm_attention.mla_rope_append_paged_kv_assign_q(
            torch.empty_like(q), partial_latent_cache, attn_metadata)
        delattr(attn_metadata, "ctx_kv_seq_lens")


        # # debug
        # origin_ctx_kv_seq_lens = getattr(attn_metadata, "ctx_kv_indptr")
        # setattr(attn_metadata, "ctx_kv_indptr", latent_cache_chunks_seq_lens)
        # full_compressed_kv, k_pe = trtllm_attention.load_paged_kv_cache_for_mla(
        #     attn_metadata, q.dtype)
        # setattr(attn_metadata, "ctx_kv_indptr", origin_ctx_kv_seq_lens)

        # print(f"czq rank {self.tp_rank} get kv: {full_compressed_kv.sum(dim=1)}")
        # if czq_idx == TARGET and self.tp_rank == 0:
        #     # print(f"full_compressed_kv: {full_compressed_kv.shape}")
        #     import pdb; pdb.set_trace()
        ##############
            

        kv = self.kv_b_proj(compressed_kv)
        k_nope, v = kv.split(
            [
                self.num_heads * self.qk_nope_head_dim,
                self.num_heads * self.v_head_dim
            ],
            -1,
        )

        k = torch.empty_like(q).view(-1, self.num_heads, self.qk_head_dim)
        k[..., :self.qk_nope_head_dim] = k_nope.view(-1, self.num_heads,
                                                     self.qk_nope_head_dim)
        # if self.apply_rotary_emb:
        if True:
            k[..., self.qk_nope_head_dim:] = k_pe.view(-1, 1,
                                                       self.qk_rope_head_dim)
        k = k.view(-1, self.num_heads * self.qk_head_dim)

        # if czq_idx == TARGET:
        #     import pdb; pdb.set_trace()
        # May concat q(including q_pe), k + k_pe, v together
        q, k, v = self._maybe_concat_qkv(q, k, v)

        # out_scale = getattr(self.o_proj, "inv_input_scale", None)
        out_scale = None  # Currently we use BF16 MHA for context phase

        attn_output = self.mha.forward(
            q,
            k,
            v,
            attn_metadata,
            attention_input_type=AttentionInputType.context_only,
            latent_cache=None,
            out_scale=out_scale,
            output=output,
        )

        return attn_output

    def forward_context_default(
            self,
            q: torch.Tensor,
            compressed_kv: torch.Tensor,
            k_pe: torch.Tensor,
            attn_metadata: AttentionMetadata,
            latent_cache: Optional[torch.Tensor] = None,
            output: Optional[torch.Tensor] = None) -> torch.Tensor:
        kv = self.kv_b_proj(compressed_kv)
        k_nope, v = kv.split(
            [
                self.num_heads * self.qk_nope_head_dim,
                self.num_heads * self.v_head_dim
            ],
            -1,
        )

        k = torch.empty_like(q).view(-1, self.num_heads, self.qk_head_dim)
        k[..., :self.qk_nope_head_dim] = k_nope.view(-1, self.num_heads,
                                                     self.qk_nope_head_dim)
        if self.apply_rotary_emb:
            k[..., self.qk_nope_head_dim:] = k_pe.view(-1, 1,
                                                       self.qk_rope_head_dim)
        k = k.view(-1, self.num_heads * self.qk_head_dim)

        # May concat q(including q_pe), k + k_pe, v together
        q, k, v = self._maybe_concat_qkv(q, k, v)

        # out_scale = getattr(self.o_proj, "inv_input_scale", None)
        out_scale = None  # Currently we use BF16 MHA for context phase

        attn_output = self.mha.forward(
            q,
            k,
            v,
            attn_metadata,
            attention_input_type=AttentionInputType.context_only,
            latent_cache=latent_cache,
            out_scale=out_scale,
            output=output,
        )

        return attn_output

    def forward_context_with_cached_kv(
        self,
        q: torch.Tensor,
        latent_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert latent_cache is not None
        trtllm_attention = cast(TrtllmAttention, self.mha)

        # apply RoPE, append compressed_kv + k_pe to paged kv cache and assign q_pe to q
        trtllm_attention.mla_rope_append_paged_kv_assign_q(
            q, latent_cache, attn_metadata)

        # copy full_compressed_kv and full_k_pe from paged kv cache
        full_compressed_kv, full_k_pe = trtllm_attention.load_paged_kv_cache_for_mla(
            attn_metadata, q.dtype)
        assert full_compressed_kv.shape[
            0] == attn_metadata.num_ctx_cached_tokens + attn_metadata.num_ctx_tokens
        assert full_compressed_kv.shape[1] == self.kv_lora_rank
        assert full_k_pe.shape[
            0] == attn_metadata.num_ctx_cached_tokens + attn_metadata.num_ctx_tokens
        assert full_k_pe.shape[1] == self.qk_rope_head_dim
        assert full_compressed_kv.is_contiguous()
        assert full_k_pe.is_contiguous()

        # compute full_k_nope and full_v from full_compressed_kv
        full_kv = self.kv_b_proj(full_compressed_kv)
        full_k_nope, full_v = full_kv.split(
            [
                self.num_heads * self.qk_nope_head_dim,
                self.num_heads * self.v_head_dim
            ],
            -1,
        )
        full_k_nope = full_k_nope.view(-1, self.num_heads,
                                       self.qk_nope_head_dim)
        full_v = full_v.view(-1, self.num_heads, self.v_head_dim)

        # build paged_full_kv
        tokens_per_block = attn_metadata.kv_cache_manager.tokens_per_block
        # paged_full_kv will be initialized to 0 in the kernel to avoid NaN
        paged_full_kv = torch.empty([
            attn_metadata.num_contexts, 2,
            (attn_metadata.max_ctx_kv_len + tokens_per_block - 1) //
            tokens_per_block, self.num_heads, tokens_per_block,
            max(self.qk_nope_head_dim + self.qk_rope_head_dim, self.v_head_dim)
        ],
                                    dtype=q.dtype,
                                    device=q.device)
        mla_context_kv_cache_block_offsets = trtllm_attention.set_paged_kv_cache_for_mla(
            paged_full_kv,
            full_k_nope,
            full_v,
            full_k_pe,
            attn_metadata,
        )

        # release pytorch activation memory
        full_compressed_kv = None
        full_k_pe = None
        full_kv = None
        full_k_nope = None
        full_v = None

        # out_scale = getattr(self.o_proj, "inv_input_scale", None)
        out_scale = None  # Currently we use BF16 MHA for context phase

        attn_output = self.mha.forward(
            q,
            None,
            None,
            attn_metadata,
            attention_input_type=AttentionInputType.context_only,
            latent_cache=None,
            out_scale=out_scale,
            mla_context_paged_kv=paged_full_kv,
            mla_context_kv_cache_block_offsets=
            mla_context_kv_cache_block_offsets,
            output=output,
        )

        return attn_output

    def forward_context_with_chunked_prefill(
        self,
        q: torch.Tensor,
        compressed_kv: torch.Tensor,
        latent_cache: torch.
        Tensor,  # compressed_kv + k_pe [context_tokens, 1, lora_size + rope_size]
        attn_metadata: TrtllmAttentionMetadata,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        trtllm_attention = cast(TrtllmAttention, self.mha)
        # apply RoPE, append compressed_kv + k_pe to paged kv cache and assign q_pe to q
        trtllm_attention.mla_rope_append_paged_kv_assign_q(
            q, latent_cache, attn_metadata)

        # determine the number of loop
        # currently we assume that the chunk size is the same as the max_num_tokens
        chunk_size = attn_metadata.runtime_features.chunk_size
        chunked_loop_num = attn_metadata.chunked_loop_num

        # [toal_token_q, num_heads, 2] -> [toal_token_q, num_heads] float2
        self.softmax_stats_tensor = torch.empty(
            (attn_metadata.num_ctx_tokens, self.num_heads, 2),
            dtype=torch.float,
            device='cuda',
        )
        self.temp_softmax_stats_tensor = torch.empty(
            (attn_metadata.num_ctx_tokens, self.num_heads, 2),
            dtype=torch.float,
            device='cuda',
        )
        if output is None:
            attn_output = q.new_empty(
                (q.size(0), self.num_heads * self.v_head_dim), dtype=q.dtype)
        else:
            attn_output = output
        temp_attn_output = q.new_empty(
            (q.size(0), self.num_heads * self.v_head_dim), dtype=q.dtype)

        # use fake cached_cu_seq_len for chunked loop
        origin_kv_lens_cuda_runtime = attn_metadata.kv_lens_cuda_runtime
        origin_kv_lens_runtime = attn_metadata.kv_lens_runtime

        # chunk context 问题:
        #   1. 为啥这里全循环完了，chunk context难道不是全程只能跑一个chunk？
        #   2. 
        # MISC:
        #   1. 必须先正常做完mha（其中先写完整kv，然后fmha kernel需要读paged kv）然后 hack
        #   2. [TEST0] 写paged kv 和 mha 是解耦的，v0把写paged kv拆到外面，结果对齐，v1只写部分kv
        #   3. [TEST1] 不如直接每个rank 一个chunk，然后reduce看下结果
        # flash decoding 做法:
        # 1. 【先不改】context 做完，用 set_chunked_kv_cache_for_mla 把当前rank的kv chunk顶到最前面
        #    - 可能的问题：需要改造，set_chunked_kv_cache_for_mla，目前他只支持写到tensor？
        # 2. generation 
        #   a. 【先不改】每次需要修改kv len 到实际需要值(考虑到上一次，每个rank都加了新token，这次只挑一个rank+1)
        #     - kv len怎么改
        #   a. kv 全存，每个rank读一部分出来（或者全读，然后split）(先把后面走通)
        #   b. torch allGather q_pe q_nope
        #   c. 每个rank正常写kv，attentionOp内基本不改
        #     - 可能的问题：q尺寸不对，kv len怎么给
        #   d. torch all2all每个rank拿部分head size
        #     - 第一步可以all gather + split走通
        #   e. reduce, 用 merge_attention_for_mla

        for loop_idx in range(chunked_loop_num):
            # {b, chunked_unit_size, h, kv_lora_rank + qk_rope_head_dim} zero padded
            # fetch `loop_idx` chunk from kv cache
            temp_cu_chunked_seq_len = attn_metadata.cu_chunked_seq_len[loop_idx]
            total_ctx_chunked_tokens = attn_metadata.host_cu_chunked_seq_len[
                loop_idx, attn_metadata.num_contexts]
            chunked_compressed_kv, chunked_k_pe = trtllm_attention.load_chunked_kv_cache_for_mla(
                metadata=attn_metadata,
                chunked_idx=loop_idx,
                num_ctx_cached_tokens=total_ctx_chunked_tokens,
                cu_chunked_seq_len=temp_cu_chunked_seq_len,
                out_dtype=q.dtype)

            # up proj to uncompressed kv
            # [tokens, 2, h, kv_dim], without rope_dim
            chunked_kv = self.kv_b_proj(chunked_compressed_kv)

            # build full_kv
            # full_kv {B, 2, chunk_size / tokens_per_block, h, tokens_per_block, kv_dim + rope_dim}
            tokens_per_block = attn_metadata.kv_cache_manager.tokens_per_block
            full_kv = torch.zeros([
                attn_metadata.num_contexts, 2,
                (chunk_size + tokens_per_block - 1) // tokens_per_block,
                self.num_heads, tokens_per_block,
                max(self.qk_nope_head_dim + self.qk_rope_head_dim,
                    self.v_head_dim)
            ],
                                  dtype=q.dtype,
                                  device=q.device)
            mla_kv_cache_block_offsets = trtllm_attention.set_chunked_kv_cache_for_mla(
                full_kv,
                chunked_kv,
                chunked_k_pe,
                cu_chunked_seq_len=temp_cu_chunked_seq_len,
                cached=True,
                metadata=attn_metadata)

            # copy chunked_seq_len to replace kv_lens_runtime
            attn_metadata.kv_lens_runtime = attn_metadata.host_chunked_seq_len[
                loop_idx]
            attn_metadata.kv_lens_cuda_runtime = attn_metadata.chunked_seq_len[
                loop_idx]
            out_scale = None
            # do not apply mask for attention within loop
            temp_attn_output = self.mha.forward(
                q,
                None,
                None,
                attn_metadata,
                attention_input_type=AttentionInputType.context_only,
                latent_cache=None,
                out_scale=out_scale,
                attention_mask=PredefinedAttentionMask.FULL,
                mla_context_paged_kv=full_kv, # 新建一个tensor作为primary pool
                mla_context_kv_cache_block_offsets=mla_kv_cache_block_offsets,
                softmax_stats_tensor=self.temp_softmax_stats_tensor,
                output=temp_attn_output,
            )
            # merge attn result
            temp_merge_op = attn_metadata.merge_op_tensor[loop_idx]
            trtllm_attention.merge_attention_for_mla(
                attn_output, temp_attn_output, self.softmax_stats_tensor,
                self.temp_softmax_stats_tensor, temp_merge_op, attn_metadata)

        # deal with the uncached kv
        kv = self.kv_b_proj(compressed_kv)
        _, k_pe = latent_cache.view([
            -1, self.kv_lora_rank + self.qk_rope_head_dim
        ]).split([self.kv_lora_rank, self.qk_rope_head_dim], -1)
        k_pe = k_pe.contiguous()
        # final round of attention

        # out_scale = getattr(self.o_proj, "inv_input_scale", None)
        out_scale = None  # Currently we use BF16 MHA for context phase

        tokens_per_block = attn_metadata.kv_cache_manager.tokens_per_block
        full_kv = torch.zeros([
            attn_metadata.num_contexts, 2,
            (attn_metadata.max_ctx_seq_len + tokens_per_block - 1) //
            tokens_per_block, self.num_heads, tokens_per_block,
            max(self.qk_nope_head_dim + self.qk_rope_head_dim, self.v_head_dim)
        ],
                              dtype=q.dtype,
                              device=q.device)
        mla_kv_cache_block_offsets = trtllm_attention.set_chunked_kv_cache_for_mla(
            full_kv,
            kv,
            k_pe,
            cu_chunked_seq_len=None,
            cached=False,
            metadata=attn_metadata)
        # copy q_lens to replace kv_lens_runtime
        attn_metadata.kv_lens_runtime = attn_metadata.prompt_lens_cpu_runtime
        attn_metadata.kv_lens_cuda_runtime = attn_metadata.prompt_lens_cuda_runtime
        temp_attn_output = self.mha.forward(
            q,
            None,
            None,
            attn_metadata,
            attention_input_type=AttentionInputType.context_only,
            latent_cache=None,
            out_scale=out_scale,
            mla_context_paged_kv=full_kv,
            mla_context_kv_cache_block_offsets=mla_kv_cache_block_offsets,
            softmax_stats_tensor=self.temp_softmax_stats_tensor,
            output=temp_attn_output,
        )
        temp_merge_op = attn_metadata.merge_op_tensor[chunked_loop_num]
        trtllm_attention.merge_attention_for_mla(attn_output, temp_attn_output,
                                                 self.softmax_stats_tensor,
                                                 self.temp_softmax_stats_tensor,
                                                 temp_merge_op, attn_metadata)
        # copy back kv_lens_runtime and kv_lens_cuda_runtime
        attn_metadata.kv_lens_runtime = origin_kv_lens_runtime
        attn_metadata.kv_lens_cuda_runtime = origin_kv_lens_cuda_runtime

        return attn_output

    def forward_context(
        self,
        q: torch.Tensor,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        attn_metadata: AttentionMetadata,
        latent_cache: Optional[torch.Tensor] = None,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # print(f"czq context latent_cache: {latent_cache.shape} \n {latent_cache[:, :512].sum(dim=1)}")
        # if True:
        #     return self.forward_context_for_flash_decoding(
        #         q, compressed_kv, latent_cache, attn_metadata, output)
        if isinstance(self.mha, TrtllmAttention):
            assert isinstance(attn_metadata, TrtllmAttentionMetadata)
            trtllm_attention = cast(TrtllmAttention, self.mha)
            # chunked context
            if trtllm_attention.is_chunked_prefill_for_mla_context(
                    attn_metadata):
                # print("czq 2")
                return self.forward_context_with_chunked_prefill(
                    q, compressed_kv, latent_cache, attn_metadata, output)
            # kv cache reuse
            elif trtllm_attention.has_cached_kv_for_mla_context(attn_metadata):
                # print("czq 3")
                return self.forward_context_with_cached_kv(
                    q, latent_cache, attn_metadata, output)
        # normal
        return self.forward_context_default(q, compressed_kv, k_pe,
                                            attn_metadata, latent_cache, output)

    def forward_generation(
            self,
            q: torch.Tensor,
            compressed_kv: torch.Tensor,
            k_pe: torch.Tensor,
            attn_metadata: AttentionMetadata,
            latent_cache: Optional[torch.Tensor] = None,
            output: Optional[torch.Tensor] = None) -> torch.Tensor:
        global FLASH_DECODING
        if FLASH_DECODING == "1":
            return self.forward_generation_flash_decoding(q, compressed_kv, k_pe,
                                              attn_metadata, latent_cache, output)
        else:
            return self.forward_generation_default(q, compressed_kv, k_pe,
                                              attn_metadata, latent_cache, output)

    def forward_generation_flash_decoding(
            self,
            q: torch.Tensor,
            compressed_kv: torch.Tensor,
            k_pe: torch.Tensor,
            attn_metadata: AttentionMetadata,
            latent_cache: Optional[torch.Tensor] = None,
            output: Optional[torch.Tensor] = None) -> torch.Tensor:

        global czq_idx
        # czq_idx += 1
        # print(f"czq_idx: {czq_idx}")

        trtllm_attention = cast(TrtllmAttention, self.mha)
        # A. 更新 q_pe，并全存kv
        # apply RoPE, append compressed_kv + k_pe to paged kv cache and assign q_pe to q
        trtllm_attention.mla_rope_append_paged_kv_assign_q_flash_decoding(
            q, latent_cache, attn_metadata, self.tp_size)

        # bs > 1 时会读成别的seq的kv，待排查
        # 先默认bs==1
        # TODO 直接load chunked
        full_kv, full_k_pe = trtllm_attention.load_paged_kv_cache_for_mla_flash_decoding(
            attn_metadata, q.dtype)

        max_num_cached_tokens_per_seq = 1 + max(attn_metadata.kv_cache_params.num_cached_tokens_per_seq[-attn_metadata.num_generations:])
        tokens_per_block = attn_metadata.kv_cache_manager.tokens_per_block

        # use 1 instead of 2 in dim 1, because kv are the same in MLA
        paged_full_kv = torch.zeros([
            attn_metadata.num_generations, 1,
            (max_num_cached_tokens_per_seq + tokens_per_block - 1) //
            tokens_per_block * tokens_per_block,
            self.kv_lora_rank + self.qk_rope_head_dim
        ],
                              dtype=q.dtype,
                              device=q.device)

        # TODO: 改成用一个kernel
        # mla_kv_cache_block_offsets = trtllm_attention.set_chunked_kv_cache_for_mla_flash_decoding(
        #     paged_full_kv,
        #     full_kv,
        #     full_k_pe,
        #     cu_chunked_seq_len=None,
        #     # cached=False,
        #     metadata=attn_metadata)

        # padded_size = (max_num_cached_tokens_per_seq + tokens_per_block - 1) // tokens_per_block * tokens_per_block

        # max_num_cached_tokens_per_seq = 1 + max(attn_metadata.kv_cache_params.num_cached_tokens_per_seq[-attn_metadata.num_generations:])
        num_cached_tokens_cu_ori = 0
        for i in range(attn_metadata.num_generations):
            num_cached_tokens = 1 + attn_metadata.kv_cache_params.num_cached_tokens_per_seq[i - attn_metadata.num_generations]
            num_cached_tokens_cu = num_cached_tokens_cu_ori + num_cached_tokens
            # paged_full_kv[i, 0, :num_cached_tokens, :self.kv_lora_rank] = full_kv[num_cached_tokens_cu_ori:num_cached_tokens_cu, :]
            # paged_full_kv[i, 0, :num_cached_tokens, self.kv_lora_rank:] = full_k_pe[num_cached_tokens_cu_ori:num_cached_tokens_cu, :]
            full_kv_slice = full_kv[num_cached_tokens_cu_ori:num_cached_tokens_cu, :]
            full_kv_chunk = torch.chunk(full_kv_slice, self.tp_size, dim=0)[self.tp_rank]
            paged_full_kv[i, 0, :full_kv_chunk.shape[0], :self.kv_lora_rank] = full_kv_chunk
            full_k_pe_slice = full_k_pe[num_cached_tokens_cu_ori:num_cached_tokens_cu, :]
            full_k_pe_chunk = torch.chunk(full_k_pe_slice, self.tp_size, dim=0)[self.tp_rank]
            paged_full_kv[i, 0, :full_k_pe_chunk.shape[0], self.kv_lora_rank:] = full_k_pe_chunk
            num_cached_tokens_cu_ori = num_cached_tokens_cu

        max_cached_tokens_per_seq = 1 + max(attn_metadata.kv_cache_params.num_cached_tokens_per_seq[-attn_metadata.num_generations:])
        max_block_num = int((max_cached_tokens_per_seq + attn_metadata.kv_cache_manager.tokens_per_block - 1) / attn_metadata.kv_cache_manager.tokens_per_block)
        # repeat offsets for 2 times because kv are the same in MLA
        mla_kv_cache_block_offsets = torch.arange(0, attn_metadata.num_generations * 1 * max_block_num, dtype=torch.int32, device=q.device).view(-1, 1, max_block_num).repeat(1, 2, 1)

        # TODO:
        # v1[done]: 单卡，走通外置kv处理；在不切割kv对齐输出(可能需要打印原版kv看下)
        # v2: 裸切割kv，实现reduce，对齐输出(先allgather+split，再all2all)
        # v3: 参考mla_rope_append_paged_kv_assign_q_flash_decoding实现切割+只在一个rank写kv

        num_tokens = q.shape[0]
        q_nope, q_pe = q.view([-1, self.num_heads, self.qk_head_dim]).split(
            [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        # fused_q contains 1) the result of the following bmm with shape [num_tokens, num_heads, kv_lora_rank]
        # 2) rope(q_pe) with shape [num_tokens, num_heads, qk_rope_head_dim]. rope is applied inside AttentionOp
        fused_q = torch.empty(
            [
                num_tokens, self.num_heads,
                (self.kv_lora_rank + self.qk_rope_head_dim)
            ],
            dtype=q.dtype,
            device=q.device,
        )

        if self.k_b_proj_trans.dtype == torch.bfloat16:
            # [num_heads, num_tokens, self.qk_nope_head_dim]
            q_nope_t = q_nope.transpose(0, 1)
            # [num_heads, num_tokens, self.kv_lora_rank]
            q_nope_out = fused_q[..., :self.kv_lora_rank].transpose(0, 1)

            # [num_heads, num_tokens, self.qk_nope_head_dim] x [num_heads, kv_lora_rank, qk_nope_head_dim]
            # -> [num_heads, num_tokens, kv_lora_rank] -> [num_tokens, num_heads, kv_lora_rank]
            # The output of bmm is written directly into fused_q
            torch.ops.trtllm.bmm_out(q_nope_t,
                                     self.k_b_proj_trans.transpose(1, 2),
                                     q_nope_out)
        elif self.k_b_proj_trans.dtype == torch.float8_e4m3fn:
            # [num_heads, num_tokens, self.kv_lora_rank]
            q_nope_out = fused_q[..., :self.kv_lora_rank].transpose(0, 1)

            fp8_block_scaling_bmm_out(q_nope, self.k_b_proj_trans,
                                      self.k_b_proj_trans_scale, q_nope_out)
        else:
            raise NotImplementedError(
                f"Missing bmm impl for dtype: {self.k_b_proj_trans.dtype}.")

        # if self.apply_rotary_emb:
        fused_q[..., self.kv_lora_rank:] = q_pe
        fused_q = fused_q.view([
            num_tokens,
            self.num_heads,
            self.kv_lora_rank + self.qk_rope_head_dim
        ])

        # out_scale = getattr(self.o_proj, "inv_input_scale", None)
        out_scale = None  # Although we use FP8 MLA for generation phase, the output is still in BF16

        from ..distributed import allgather
        gathered_fused_q = allgather(fused_q, self.mapping, dim=1)

        # print(f"czq fused_q: {fused_q.shape} \n {fused_q[:, :32]}")

        partial_softmax_stats_tensor = torch.empty(
            (num_tokens, self.global_num_heads, 2),
            dtype=torch.float,
            device='cuda',
        )
        attn_out_latent = self.mqa.forward(
            gathered_fused_q.view(num_tokens, -1),
            None,
            None,
            attn_metadata,
            attention_input_type=AttentionInputType.generation_only,
            out_scale=out_scale,
            latent_cache=latent_cache,  # kvcache and k_pe
            q_pe=q_pe,  # used by `invokeMLARopeGeneration`
            mla_context_paged_kv=paged_full_kv,
            mla_context_kv_cache_block_offsets=mla_kv_cache_block_offsets,
            softmax_stats_tensor=partial_softmax_stats_tensor,
        )

        if czq_idx % 30 == 1:
            import pdb;pdb.set_trace()
        
        # 先allgather,之后改all2all
        if self.tp_size > 0:
            partial_attn_out_latent = attn_out_latent
            gathered_attn_out_latent = allgather(
                partial_attn_out_latent.view(1, *partial_attn_out_latent.shape),
                self.mapping,
                dim=0
            )
            head_start = self.num_heads * self.tp_rank
            head_end = head_start + self.num_heads
            gathered_attn_out_latent = gathered_attn_out_latent.view(self.tp_size, num_tokens, self.global_num_heads, self.kv_lora_rank)
            gathered_attn_out_latent = gathered_attn_out_latent[:, :, head_start: head_end, :]
            gathered_attn_out_latent = gathered_attn_out_latent.view(self.tp_size, num_tokens, -1)
            gathered_partial_softmax_stats_tensor = allgather(
                partial_softmax_stats_tensor.view(1, *partial_softmax_stats_tensor.shape),
                self.mapping,
                dim=0
            )
            gathered_partial_softmax_stats_tensor = gathered_partial_softmax_stats_tensor[:, :, head_start: head_end, :]

            attn_out_latent = gathered_attn_out_latent[0]
            reduced_softmax_stats_tensor = torch.empty_like(partial_softmax_stats_tensor)
            # 结果不对，和gt对一下
            merge_op = torch.ones(
                [num_tokens],
                dtype=torch.int64,
                device=q.device,
            )
            for i in range(1, self.tp_size):
                # print("czq attn_out_latent", attn_out_latent)
                # print("czq gathered_attn_out_latent", gathered_attn_out_latent[i])
                # print("czq reduced_softmax_stats_tensor", reduced_softmax_stats_tensor)
                # print("czq gathered_partial_softmax_stats_tensor", gathered_partial_softmax_stats_tensor[i])
                trtllm_attention.merge_attention_for_mla_flash_decoding(attn_out_latent, gathered_attn_out_latent[i],
                                                         reduced_softmax_stats_tensor,
                                                         gathered_partial_softmax_stats_tensor[i],
                                                         merge_op, attn_metadata, self.num_heads)

        if czq_idx % 30 == 1:
            import pdb;pdb.set_trace()

        # print(f"czq attn_out_latent: {attn_out_latent.shape} \n {attn_out_latent[:, :32]}")
        fused_q = None

        assert (attn_out_latent.shape[0] == q.shape[0] and
                attn_out_latent.shape[1] == self.num_heads * self.kv_lora_rank)

        # [seq, num_heads, kv_lora_rank]
        attn_out_latent = attn_out_latent.view(
            [-1, self.num_heads, self.kv_lora_rank])

        # [seq, num_heads * v_head_dim]
        output = output if output is not None else torch.empty(
            [num_tokens, self.num_heads * self.v_head_dim],
            dtype=attn_out_latent.dtype,
            device=attn_out_latent.device)

        attn_output = output.view([num_tokens, self.num_heads, self.v_head_dim])

        if self.v_b_proj.dtype == torch.bfloat16:
            # [num_heads, seq, kv_lora_rank] x [num_heads, kv_lora_rank, v_head_dim]
            # -> [num_heads, seq, v_head_dim]
            torch.ops.trtllm.bmm_out(attn_out_latent.transpose(0, 1),
                                     self.v_b_proj.transpose(1, 2),
                                     attn_output.transpose(0, 1))
        elif self.v_b_proj.dtype == torch.float8_e4m3fn:
            fp8_block_scaling_bmm_out(attn_out_latent, self.v_b_proj,
                                      self.v_b_proj_scale,
                                      attn_output.transpose(0, 1))
        else:
            raise NotImplementedError(
                f"Missing bmm impl for dtype: {self.v_b_proj.dtype}.")

        return output

    def forward_generation_default(
            self,
            q: torch.Tensor,
            compressed_kv: torch.Tensor,
            k_pe: torch.Tensor,
            attn_metadata: AttentionMetadata,
            latent_cache: Optional[torch.Tensor] = None,
            output: Optional[torch.Tensor] = None) -> torch.Tensor:
        num_tokens = q.shape[0]
        q_nope, q_pe = q.view([-1, self.num_heads, self.qk_head_dim]).split(
            [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        # fused_q contains 1) the result of the following bmm with shape [num_tokens, num_heads, kv_lora_rank]
        # 2) rope(q_pe) with shape [num_tokens, num_heads, qk_rope_head_dim]. rope is applied inside AttentionOp
        fused_q = torch.empty(
            [
                num_tokens, self.num_heads,
                (self.kv_lora_rank + self.qk_rope_head_dim)
            ],
            dtype=q.dtype,
            device=q.device,
        )

        if self.k_b_proj_trans.dtype == torch.bfloat16:
            # [num_heads, num_tokens, self.qk_nope_head_dim]
            q_nope_t = q_nope.transpose(0, 1)
            # [num_heads, num_tokens, self.kv_lora_rank]
            q_nope_out = fused_q[..., :self.kv_lora_rank].transpose(0, 1)

            # [num_heads, num_tokens, self.qk_nope_head_dim] x [num_heads, kv_lora_rank, qk_nope_head_dim]
            # -> [num_heads, num_tokens, kv_lora_rank] -> [num_tokens, num_heads, kv_lora_rank]
            # The output of bmm is written directly into fused_q
            torch.ops.trtllm.bmm_out(q_nope_t,
                                     self.k_b_proj_trans.transpose(1, 2),
                                     q_nope_out)
        elif self.k_b_proj_trans.dtype == torch.float8_e4m3fn:
            # [num_heads, num_tokens, self.kv_lora_rank]
            q_nope_out = fused_q[..., :self.kv_lora_rank].transpose(0, 1)

            fp8_block_scaling_bmm_out(q_nope, self.k_b_proj_trans,
                                      self.k_b_proj_trans_scale, q_nope_out)
        else:
            raise NotImplementedError(
                f"Missing bmm impl for dtype: {self.k_b_proj_trans.dtype}.")

        if self.apply_rotary_emb:
            fused_q[..., self.kv_lora_rank:] = q_pe
        fused_q = fused_q.view([
            num_tokens,
            self.num_heads * (self.kv_lora_rank + self.qk_rope_head_dim)
        ])

        # out_scale = getattr(self.o_proj, "inv_input_scale", None)
        out_scale = None  # Although we use FP8 MLA for generation phase, the output is still in BF16

        # print(f"czq fused_q: {fused_q.shape} \n {fused_q[:, ]}")
        # torch.cuda.synchronize()
        attn_out_latent = self.mqa.forward(
            fused_q,
            None,
            None,
            attn_metadata,
            attention_input_type=AttentionInputType.generation_only,
            out_scale=out_scale,
            latent_cache=latent_cache,  # kvcache and k_pe
            q_pe=q_pe,  # used by `invokeMLARopeGeneration`
        )
        torch.cuda.synchronize()
        print(f"czq attn_out_latent: {attn_out_latent.shape} \n {attn_out_latent[:, :32]}")
        fused_q = None

        assert (attn_out_latent.shape[0] == q.shape[0] and
                attn_out_latent.shape[1] == self.num_heads * self.kv_lora_rank)

        # [seq, num_heads, kv_lora_rank]
        attn_out_latent = attn_out_latent.view(
            [-1, self.num_heads, self.kv_lora_rank])

        # [seq, num_heads * v_head_dim]
        output = output if output is not None else torch.empty(
            [num_tokens, self.num_heads * self.v_head_dim],
            dtype=attn_out_latent.dtype,
            device=attn_out_latent.device)

        attn_output = output.view([num_tokens, self.num_heads, self.v_head_dim])

        if self.v_b_proj.dtype == torch.bfloat16:
            # [num_heads, seq, kv_lora_rank] x [num_heads, kv_lora_rank, v_head_dim]
            # -> [num_heads, seq, v_head_dim]
            torch.ops.trtllm.bmm_out(attn_out_latent.transpose(0, 1),
                                     self.v_b_proj.transpose(1, 2),
                                     attn_output.transpose(0, 1))
        elif self.v_b_proj.dtype == torch.float8_e4m3fn:
            fp8_block_scaling_bmm_out(attn_out_latent, self.v_b_proj,
                                      self.v_b_proj_scale,
                                      attn_output.transpose(0, 1))
        else:
            raise NotImplementedError(
                f"Missing bmm impl for dtype: {self.v_b_proj.dtype}.")

        return output

    def forward(
        self,
        position_ids: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        all_reduce_params: Optional[AllReduceParams] = None,
    ) -> torch.Tensor:

        attn_output = self.create_output(hidden_states)
        if self.register_to_config:
            torch.ops.trtllm.mla_custom_op_inplace(hidden_states, position_ids,
                                                   self.layer_idx_str,
                                                   attn_output)
        else:
            self.forward_impl(position_ids,
                              hidden_states,
                              attn_metadata,
                              output=attn_output)
        attn_output = self.o_proj(attn_output,
                                  all_reduce_params=all_reduce_params)
        return attn_output
