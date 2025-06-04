import math
from typing import Optional

import torch
from torch import nn

from tensorrt_llm.mapping import Mapping

from ..attention_backend import AttentionInputType, AttentionMetadata
from ..attention_backend.interface import (PositionalEmbeddingParams,
                                           PredefinedAttentionMask)
from ..attention_backend.utils import create_attention, get_attention_backend
from ..distributed import AllReduceParams, allgather, reducescatter
from ..model_config import ModelConfig
from ..peft.lora.layer import LoraLayer, LoraModuleType
from .linear import Linear, TensorParallelMode, WeightMode, WeightsLoadingConfig
from .multi_stream_utils import maybe_execute_in_parallel
from .rms_norm import RMSNorm
from .rotary_embedding import RotaryEmbedding


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
        layer_idx: Optional[int] = None,
        dtype: torch.dtype = None,
        dense_bias: Optional[bool] = None,
        config: Optional[ModelConfig] = None,
    ):
        super().__init__()
        self.layer_idx = layer_idx

        config = config or ModelConfig()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        if config:
            self.head_dim = getattr(config.pretrained_config, "head_dim",
                                    self.hidden_size // self.num_heads)
        else:
            self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = max_position_embeddings
        self.pos_embd_params = pos_embd_params
        self.dense_bias = dense_bias

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
        )
        self.o_proj = Linear(
            tp_size * self.q_size,
            self.hidden_size,
            bias=self.dense_bias,
            dtype=dtype,
            mapping=mapping,
            tensor_parallel_mode=TensorParallelMode.ROW,
            quant_config=config.get_quant_config(),
            skip_create_weights_in_init=config.skip_create_weights_in_init,
        )
        self.quant_config = config.get_quant_config()
        self.mapping_config = config.mapping
        self.attn_backend = config.attn_backend
        self.pos_embd_params = pos_embd_params

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

        self.use_qk_norm = (
            config.pretrained_config
            and (config.pretrained_config.model_type == 'qwen3'
                 or config.pretrained_config.model_type == 'qwen3_moe'))
        attn_cls = get_attention_backend(self.attn_backend)
        self.enable_rope_fusion = attn_cls.support_fused_rope(
        ) and not self.use_qk_norm
        self.attn = create_attention(
            self.attn_backend,
            self.layer_idx,
            self.num_heads,
            self.head_dim,
            self.num_key_value_heads,
            pos_embd_params=self.pos_embd_params
            if self.enable_rope_fusion else None,
            quant_config=self.quant_config,
            skip_create_weights_in_init=config.skip_create_weights_in_init,
            mapping=self.mapping_config,
        )

        self.support_fused_qkv = self.attn.support_fused_qkv()

        self.rotary_emb = None
        self.apply_rotary_emb = (not self.enable_rope_fusion
                                 and pos_embd_params is not None)
        if self.apply_rotary_emb:
            self.rotary_emb = RotaryEmbedding(
                pos_embd_params.rope,
                head_dim=self.head_dim,
                is_neox=pos_embd_params.is_neox,
            )

        if not config.skip_create_weights_in_init:
            self.create_weights()

    def create_weights(self):
        # self.attn has no weights but has states that are related to quant_config,
        # which could be modified after __init__
        self.attn.update_quant_config(self.quant_config)

    def convert_qkv(self, q, k, v):
        if k is None and v is None and not self.support_fused_qkv:
            q, k, v = q.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        elif k is not None and v is not None and self.support_fused_qkv:
            qkv = torch.concat([q, k, v], dim=-1)
            q, k, v = qkv, None, None
        return q, k, v

    def forward(
        self,
        position_ids: Optional[torch.LongTensor],
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        attention_mask: PredefinedAttentionMask = PredefinedAttentionMask.
        CAUSAL,
        mrope_config: Optional[dict] = None,
        all_reduce_params: Optional[AllReduceParams] = None,
        lora_params: Optional[dict] = None,
        **kwargs,
    ) -> torch.Tensor:
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
        if self.apply_rotary_emb and position_ids is not None:
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size],
                                dim=-1)
            if hasattr(self, 'q_norm') and hasattr(self, 'k_norm'):
                # Add qk-norm
                if hasattr(self, 'ln_events'):
                    q_l2norm = lambda: self.q_norm(q.reshape(-1, self.head_dim)
                                                   ).reshape(-1, self.q_size)
                    k_l2norm = lambda: self.k_norm(k.reshape(-1, self.head_dim)
                                                   ).reshape(-1, self.kv_size)
                    q, k = maybe_execute_in_parallel(
                        q_l2norm,
                        k_l2norm,
                        self.ln_events[0],
                        self.ln_events[1],
                        self.aux_stream,
                    )
                else:
                    q_by_head = q.reshape(-1, self.head_dim)
                    q_by_head = self.q_norm(q_by_head)
                    q = q_by_head.view(q.shape)
                    k_by_head = k.reshape(-1, self.head_dim)
                    k_by_head = self.k_norm(k_by_head)
                    k = k_by_head.view(k.shape)

            q, k = self.rotary_emb(position_ids, [q, k])
        out_scale = None

        if self.o_proj.has_fp8_qdq or self.o_proj.has_nvfp4 or self.o_proj.has_fp8_block_scales:
            out_scale = self.o_proj.inv_input_scale

        q, k, v = self.convert_qkv(q, k, v)
        attn_output = self.attn.forward(q,
                                        k,
                                        v,
                                        attn_metadata,
                                        out_scale=out_scale,
                                        attention_mask=attention_mask,
                                        mrope_config=mrope_config)
        hidden_states = attn_output
        attn_output = self.o_proj(attn_output,
                                  all_reduce_params=all_reduce_params)
        if bool(lora_params):
            attn_lora_output = self.o_lora(hidden_states, lora_params,
                                           self.layer_idx)
            if attn_lora_output is not None:
                attn_output = attn_output + attn_lora_output

        return attn_output


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
        super().__init__()
        self.layer_idx = layer_idx
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

        assert self.num_heads % tp_size == 0
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
                use_custom_cublas_mm=True)

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
                skip_create_weights_in_init=config.skip_create_weights_in_init)
        else:
            self.fused_a = Linear(
                hidden_size,
                self.kv_lora_rank + self.qk_rope_head_dim,
                bias=bias,
                dtype=dtype,
                quant_config=quant_config,
                skip_create_weights_in_init=config.skip_create_weights_in_init,
                use_custom_cublas_mm=True)

            self.q_proj = Linear(
                self.q_lora_rank,
                tp_size * self.num_heads * self.qk_head_dim,
                bias=bias,
                dtype=dtype,
                mapping=mapping,
                tensor_parallel_mode=TensorParallelMode.COLUMN,
                quant_config=quant_config,
                skip_create_weights_in_init=config.skip_create_weights_in_init,
            )
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
            skip_create_weights_in_init=config.skip_create_weights_in_init)
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
        )

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
            mapping=config.mapping,
        )

        self.mqa = create_attention(
            config.attn_backend,
            self.layer_idx,
            self.num_heads,
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
            mapping=config.mapping,
        )

        self.aux_stream = aux_stream
        self.ln_events = [torch.cuda.Event(), torch.cuda.Event()]

        self.enable_rope_fusion = self.mha.support_fused_rope()
        self.support_fused_qkv = self.mha.support_fused_qkv()
        self.rotary_emb = None
        self.apply_rotary_emb = not self.enable_rope_fusion
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

        has_fp8_block_scales = self.quant_config and self.quant_config.quant_mode.has_fp8_block_scales(
        )

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

    def forward(
        self,
        position_ids: Optional[torch.LongTensor],
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        all_reduce_params: Optional[AllReduceParams] = None,
    ) -> torch.Tensor:
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

            attn_output_context = self.forward_context(q_ctx, compressed_kv_ctx,
                                                       k_pe_ctx, attn_metadata,
                                                       latent_cache_ctx)
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

            attn_output_gen = self.forward_generation(q_gen, compressed_kv_gen,
                                                      k_pe_gen, attn_metadata,
                                                      latent_cache_gen)
        else:
            attn_output_gen = None

        # release pytorch activation memory
        q = None
        compressed_kv = None
        k_pe = None

        # merge context and gen batches
        if attn_output_context is not None and attn_output_gen is not None:
            assert (
                len(attn_output_context.shape) == 2
            ), f"attn_output_context must be rank 2, not {len(attn_output_context.shape)}"
            assert (
                len(attn_output_gen.shape) == 2
            ), f"attn_output_gen must be rank 2, not {len(attn_output_gen.shape)}"
            attn_output = torch.cat([attn_output_context, attn_output_gen],
                                    dim=0)
            # release pytorch activation memory
            attn_output_context = None
            attn_output_gen = None
        elif attn_output_gen is None:
            attn_output = attn_output_context
        else:
            attn_output = attn_output_gen

        attn_output = self.o_proj(attn_output,
                                  all_reduce_params=all_reduce_params)
        return attn_output

    def _maybe_concat_qkv(self, q, k, v):
        if k is not None and v is not None and self.support_fused_qkv:
            qkv = torch.concat([q, k, v], dim=-1)
            q, k, v = qkv, None, None
        return q, k, v

    def forward_context(
        self,
        q: torch.Tensor,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        attn_metadata: AttentionMetadata,
        latent_cache: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
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
        print("czq q.shape", q.shape)
        print("czq k.shape", k.shape)
        print("czq v.shape", v.shape)
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
        )
        print("czq attn_output.shape", attn_output.shape)

        return attn_output
        # return torch.ones((16,2048))

    def forward_generation(
        self,
        q: torch.Tensor,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        attn_metadata: AttentionMetadata,
        latent_cache: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
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
            q_nope_fp8, q_nope_scales = torch.ops.trtllm.fp8_batched_quantize_1x128_permute102(
                q_nope)
            # [num_heads, num_tokens, self.kv_lora_rank]
            q_nope_out = fused_q[..., :self.kv_lora_rank].transpose(0, 1)

            torch.ops.trtllm.fp8_block_scaling_bmm_out(
                q_nope_fp8, self.k_b_proj_trans, q_nope_scales,
                self.k_b_proj_trans_scale, q_nope_out)
            q_nope_scales = None
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
        fused_q = None

        assert (attn_out_latent.shape[0] == q.shape[0] and
                attn_out_latent.shape[1] == self.num_heads * self.kv_lora_rank)

        # [seq, num_heads, kv_lora_rank]
        attn_out_latent = attn_out_latent.view(
            [-1, self.num_heads, self.kv_lora_rank])

        attn_output = torch.empty([num_tokens, self.num_heads, self.v_head_dim],
                                  dtype=attn_out_latent.dtype,
                                  device=attn_out_latent.device)

        if self.v_b_proj.dtype == torch.bfloat16:
            # [num_heads, seq, kv_lora_rank] x [num_heads, kv_lora_rank, v_head_dim]
            # -> [num_heads, seq, v_head_dim]
            torch.ops.trtllm.bmm_out(attn_out_latent.transpose(0, 1),
                                     self.v_b_proj.transpose(1, 2),
                                     attn_output.transpose(0, 1))
        elif self.v_b_proj.dtype == torch.float8_e4m3fn:
            attn_out_latent, attn_out_latent_scales = torch.ops.trtllm.fp8_batched_quantize_1x128_permute102(
                attn_out_latent)

            torch.ops.trtllm.fp8_block_scaling_bmm_out(
                attn_out_latent, self.v_b_proj, attn_out_latent_scales,
                self.v_b_proj_scale, attn_output.transpose(0, 1))
            attn_out_latent_scales = None
        else:
            raise NotImplementedError(
                f"Missing bmm impl for dtype: {self.v_b_proj.dtype}.")

        # [seq, num_heads * v_head_dim]
        return attn_output.flatten(1, 2)


class VanillaMLA(nn.Module):

    def __init__(self,
                 *,
                 hidden_size: int,
                 num_attention_heads: int,
                 num_key_value_heads: int,
                 qk_nope_head_dim: int,
                 qk_rope_head_dim: int,
                 v_head_dim: int,
                 q_lora_rank: int,
                 kv_lora_rank: int,
                 max_position_embeddings: int,
                 bias: bool,
                 pos_embd_params: Optional[PositionalEmbeddingParams] = None,
                 rotary_emb: Optional[RotaryEmbedding] = None,
                 layer_idx: Optional[int] = None,
                 dtype: torch.dtype = None,
                 dense_bias: Optional[bool] = None,
                 config: Optional[ModelConfig] = None,
                 aux_stream: Optional[torch.cuda.Stream] = None):
        super().__init__()
        self.layer_idx = layer_idx
        self.dim = hidden_size
        self.n_heads = num_attention_heads
        self.n_local_heads = num_attention_heads // config.mapping.tp_size
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.dtype = dtype
        self.dense_bias = dense_bias
        if dense_bias is None:
            self.dense_bias = bias

        # tensor parallel
        device = torch.device('cuda')
        config = config or ModelConfig()
        tp_size = config.mapping.tp_size
        pp_size = config.mapping.pp_size
        if config.mapping.enable_attention_dp:
            tp_size = 1
        self.tp_size = tp_size
        self.tp_rank = config.mapping.tp_rank
        self.mapping = config.mapping
        if config.mapping.enable_attention_dp:
            self.tp_size = 1
            self.tp_rank = 0

        mapping = Mapping(
            world_size=tp_size * pp_size,
            tp_size=tp_size,
            pp_size=pp_size,
            rank=config.mapping.rank,
            gpus_per_node=config.mapping.gpus_per_node,
            enable_attention_dp=config.mapping.enable_attention_dp,
        )

        # quantization
        quant_config = config.get_quant_config()
        quant_mode = quant_config.quant_mode

        if quant_mode.has_fp8_block_scales():
            self.mla_weight_dtype = torch.float8_e4m3fn
        else:
            self.mla_weight_dtype = dtype

        if self.q_lora_rank is None:
            self.q_lora_rank = hidden_size
            self.is_lite = True
        else:
            self.is_lite = False

        if self.is_lite:
            self.wq = Linear(
                self.dim,
                self.n_heads * self.qk_head_dim,
                bias=bias,
                dtype=dtype,
                mapping=mapping,
                tensor_parallel_mode=TensorParallelMode.COLUMN,
                quant_config=config.get_quant_config(),
                skip_create_weights_in_init=config.skip_create_weights_in_init,
            )
        else:
            self.wq_a = Linear(
                self.dim,
                self.q_lora_rank,
                bias=bias,
                dtype=dtype,
                quant_config=config.get_quant_config(),
                skip_create_weights_in_init=config.skip_create_weights_in_init)
            self.q_norm = RMSNorm(hidden_size=self.q_lora_rank,
                                  eps=config.pretrained_config.rms_norm_eps,
                                  dtype=dtype)
            self.wq_b = Linear(
                self.q_lora_rank,
                self.n_heads * self.qk_head_dim,
                bias=bias,
                dtype=dtype,
                mapping=mapping,
                tensor_parallel_mode=TensorParallelMode.COLUMN,
                quant_config=config.get_quant_config(),
                skip_create_weights_in_init=config.skip_create_weights_in_init,
            )
        self.wkv_a = Linear(
            self.dim,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=bias,
            dtype=dtype,
            quant_config=config.get_quant_config(),
            skip_create_weights_in_init=config.skip_create_weights_in_init)
        self.kv_norm = RMSNorm(hidden_size=self.kv_lora_rank,
                               eps=config.pretrained_config.rms_norm_eps,
                               dtype=dtype)
        if quant_mode.has_fp8_block_scales():
            self.wkv_b = None
            self.kv_b_proj = Linear(
                self.kv_lora_rank,
                self.n_heads * (self.qk_nope_head_dim + self.v_head_dim),
                bias=bias,
                dtype=dtype,
                mapping=mapping,
                tensor_parallel_mode=TensorParallelMode.COLUMN,
                quant_config=config.get_quant_config(),
                skip_create_weights_in_init=config.skip_create_weights_in_init,
            )
            self.k_b_proj_trans = nn.Parameter(
                torch.empty(
                    (self.n_heads // tp_size, self.kv_lora_rank,
                     self.qk_nope_head_dim),
                    dtype=self.mla_weight_dtype,
                    device=device,
                ))
            self.k_b_proj_trans_scale = nn.Parameter(
                torch.empty(
                    (
                        self.n_heads // tp_size,
                        int(self.kv_lora_rank / 128),
                        int(self.qk_nope_head_dim / 128),
                    ),
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            self.v_b_proj = None  # view into self.kv_b_proj.weight
            self.v_b_proj_scale = None  # view into self.kv_b_proj.weight_scale
        else:
            self.wkv_b = Linear(
                self.kv_lora_rank,
                self.n_heads * (self.qk_nope_head_dim + self.v_head_dim),
                bias=bias,
                dtype=dtype,
                mapping=mapping,
                tensor_parallel_mode=TensorParallelMode.COLUMN,
                quant_config=config.get_quant_config(),
                skip_create_weights_in_init=config.skip_create_weights_in_init,
            )
            self.k_b_proj_trans = None
            self.k_b_proj_trans_scale = None
            self.v_b_proj = None
            self.v_b_proj_scale = None

        self.wo = Linear(
            self.n_heads * self.v_head_dim,
            self.dim,
            bias=self.dense_bias,
            dtype=dtype,
            mapping=mapping,
            tensor_parallel_mode=TensorParallelMode.ROW,
            quant_config=config.get_quant_config(),
            skip_create_weights_in_init=config.skip_create_weights_in_init,
        )

        # rope
        def yarn_get_mscale(scale=1, mscale=1):
            if scale <= 1:
                return 1.0
            return 0.1 * mscale * math.log(scale) + 1.0

        self.softmax_scale = self.qk_head_dim**-0.5
        rope_scaling = getattr(config.pretrained_config, 'rope_scaling', None)
        self.rope_params = {
            "qk_rope_head_dim": config.pretrained_config.qk_rope_head_dim,
            "rope_theta": config.pretrained_config.rope_theta,
        }
        if rope_scaling is not None:
            self.rope_params.update({
                "beta_fast":
                rope_scaling.get("beta_fast", 32),
                "beta_slow":
                rope_scaling.get("beta_slow", 1),
                "original_seq_len":
                rope_scaling.get("original_max_position_embeddings", 1024),
                "rope_factor":
                rope_scaling.get("factor", 1.0),
            })
            mscale_all_dim = rope_scaling.get("mscale_all_dim", 0.0)
            scaling_factor = rope_scaling.get("factor", 1.0)
            if mscale_all_dim:
                mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
                self.softmax_scale = self.softmax_scale * mscale * mscale
        self.freqs_cis = None
        self.flash_decoding = True

    def apply_rotary_emb(self, x: torch.Tensor,
                         freqs_cis: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))
        freqs_cis = freqs_cis.view(x.size(0), 1, x.size(-1))
        y = torch.view_as_real(x * freqs_cis).flatten(2)
        return y.to(dtype)

    def precompute_freqs_cis(
        self,
        max_seq_len: int,
        qk_rope_head_dim: int,
        beta_fast: int = 32,
        beta_slow: int = 1,
        original_seq_len: int = 4096,
        rope_factor: float = 40,
        rope_theta: float = 10000,
    ) -> torch.Tensor:
        dim = qk_rope_head_dim
        seqlen = max_seq_len
        beta_fast = beta_fast
        beta_slow = beta_slow
        base = rope_theta
        factor = rope_factor

        import math

        def find_correction_dim(num_rotations, dim, base, max_seq_len):
            return dim * math.log(
                max_seq_len /
                (num_rotations * 2 * math.pi)) / (2 * math.log(base))

        def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
            low = math.floor(
                find_correction_dim(low_rot, dim, base, max_seq_len))
            high = math.ceil(
                find_correction_dim(high_rot, dim, base, max_seq_len))
            return max(low, 0), min(high, dim - 1)

        def linear_ramp_factor(min, max, dim):
            if min == max:
                max += 0.001
            linear_func = (torch.arange(dim, dtype=torch.float32) -
                           min) / (max - min)
            ramp_func = torch.clamp(linear_func, 0, 1)
            return ramp_func

        freqs = 1.0 / (base
                       **(torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        if seqlen > original_seq_len:
            low, high = find_correction_range(beta_fast, beta_slow, dim, base,
                                              original_seq_len)
            smooth = 1 - linear_ramp_factor(low, high, dim // 2)
            freqs = freqs / factor * (1 - smooth) + freqs * smooth

        t = torch.arange(seqlen)
        freqs = torch.outer(t, freqs)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        return freqs_cis

    def _single_request_update_kv_cache(self, kv, k_pe, kv_cache_tensor,
                                        cache_idx, start_pos, end_pos):
        kv_cache = kv_cache_tensor[cache_idx,
                                   0, :, :, :self.kv_lora_rank].squeeze()
        pe_cache = kv_cache_tensor[cache_idx, 0, :, :,
                                   self.kv_lora_rank:].squeeze()
        kv_cache[start_pos:end_pos] = kv
        pe_cache[start_pos:end_pos] = k_pe
        return kv_cache[:end_pos, :], pe_cache[:end_pos, :]

    def _single_request_forward(self, x: torch.Tensor,
                                kv_cache_tensor: torch.Tensor, start_pos: int,
                                cache_idx: int):
        seqlen, _ = x.size()
        end_pos = start_pos + seqlen
        # rope param
        if end_pos > self.freqs_cis.shape[0]:
            self.freqs_cis = self.precompute_freqs_cis(max_seq_len=end_pos,
                                                       **self.rope_params).to(
                                                           x.device)
        # get mask
        mask = None
        if seqlen > 1:
            mask = torch.full((end_pos, end_pos), float("-inf"),
                              device='cuda').triu_(1)
            mask = mask[-seqlen:]
        # proj
        if self.is_lite:
            q = self.wq(x)
        else:
            # x: 1, 64, 128
            qnorm = self.q_norm(self.wq_a(x))
            # qnorm: 1, 1536
            q = self.wq_b(qnorm)
        # q rope
        q = q.view(seqlen, self.n_local_heads, self.qk_head_dim)
        # print("q.shape2", q.shape) # 1, 128/tp, 192
        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_pe = self.apply_rotary_emb(q_pe, self.freqs_cis[start_pos:end_pos])
        # kv proj a
        # 1, 512
        kv = self.wkv_a(x)
        # kv rope
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim],
                               dim=-1)
        k_pe = self.apply_rotary_emb(k_pe.unsqueeze(1),
                                     self.freqs_cis[start_pos:end_pos])
        # q_nope proj
        if self.mla_weight_dtype == torch.bfloat16:
            wkv_b = self.wkv_b.weight.view(self.n_local_heads, -1,
                                           self.kv_lora_rank)
            # 1, 128/tp, 512
            q_nope = torch.einsum("shd,hdc->shc", q_nope,
                                  wkv_b[:, :self.qk_nope_head_dim])
        elif self.mla_weight_dtype == torch.float8_e4m3fn:
            q_nope, q_nope_scales = torch.ops.trtllm.fp8_batched_quantize_1x128_permute102(
                q_nope)
            q_nope = torch.ops.trtllm.fp8_block_scaling_bmm(
                q_nope, self.k_b_proj_trans, q_nope_scales,
                self.k_b_proj_trans_scale)
            q_nope = q_nope.transpose(0, 1)
        else:
            raise NotImplementedError(
                f"Unsupported dtype: {self.mla_weight_dtype}")
        # flash decoding A: 
        # v1: allgather q_nope and q_pe
        # v2: overlap with computation / combine q_nope and q_pe
        if self.flash_decoding and seqlen == 1:
            q_nope = allgather(q_nope, self.mapping, gather_dim=1)
            q_nope = q_nope.view(seqlen, self.n_heads, self.kv_lora_rank)
            q_pe = allgather(q_pe, self.mapping, gather_dim=1)
            q_pe = q_pe.view(seqlen, self.n_heads, self.qk_rope_head_dim)
        #####################
        # [done] 先抛开 lse，做完整attention 然后split看正确性
        # [done] 计算lse
        # [done] 计算部分kv + reduce
        # 
        #####################

        # update kv cache
        kv_states, pe_states = self._single_request_update_kv_cache(
            self.kv_norm(kv), k_pe.squeeze(1), kv_cache_tensor, cache_idx,
            start_pos, end_pos)
        # flash decoding B
        # v1，kv/pe sp直接取一段
        if self.flash_decoding and seqlen == 1:
            part = end_pos // self.tp_size
            sp_start = part * self.tp_rank
            sp_end = part * (self.tp_rank + 1) if self.tp_rank + 1 < self.tp_size else end_pos
            kv_states = kv_states[sp_start:sp_end, :]
            pe_states = pe_states[sp_start:sp_end, :]
        # v2，TODO: kv只存部分

        # attention
        # flash decoding C
        # 如何得到lse，并reduceScatter
        scores = (torch.einsum("shc,tc->sht", q_nope, kv_states) + torch.einsum(
            "shr,tr->sht", q_pe, pe_states)) * self.softmax_scale
        if mask is not None:
            scores += mask.unsqueeze(1)

        # if False and self.flash_decoding:
        if self.flash_decoding and seqlen == 1:
            # 会有一些数值diff
            scores_fp32 = scores.float()
            lse = torch.logsumexp(scores_fp32, dim=-1, keepdim=True)
            scores = torch.exp(scores_fp32 - lse).type_as(x)
        else:
            scores = scores.softmax(dim=-1, dtype=torch.float32).type_as(x)

        x = torch.einsum("sht,tc->shc", scores, kv_states)

        if self.flash_decoding and seqlen == 1:
            # v1: all gather + split
            head_start = self.n_local_heads * self.tp_rank
            head_end = head_start + self.n_local_heads
            x_blocks = allgather(x, self.mapping, gather_dim=0).float().view(self.tp_size, *x.shape)
            lse_blocks = allgather(lse, self.mapping, gather_dim=0).view(self.tp_size, *lse.shape)
            x_reduce = x_blocks[0]
            lse_reduce = lse_blocks[0]
            for i in range(1, self.tp_size):
                x_block = x_blocks[i]
                lse_block = lse_blocks[i]
                lse_new = lse_reduce + torch.log(1 + torch.exp(lse_block - lse_reduce))
                x_reduce = torch.exp(lse_reduce - lse_new) * x_reduce + torch.exp(
                    lse_block - lse_new) * x_block
                lse_reduce = lse_new
            x = x_reduce[:, head_start:head_end, ...].type_as(x)

        # if self.flash_decoding and seqlen == 1:
        #     # v2: TODO 改成gather
        #     x_blocks = []
        #     lse_blocks = []
        #     for i in range(self.tp_size):
        #         head_start = self.n_local_heads * i
        #         head_end = head_start + self.n_local_heads
        #         # TODO: 需要初始化
        #         torch.distributed.gather(x[:, head_start:head_end, ...], gather_list=x_blocks, dst=i)
        #         torch.distributed.gather(lse[:, head_start:head_end, ...], gather_list=lse_blocks, dst=i)
        #     x_reduce = x_blocks[0]
        #     lse_reduce = lse_blocks[0]
        #     for i in range(1, self.tp_size):
        #         x_block = x_blocks[i]
        #         lse_block = lse_blocks[i]
        #         lse_new = lse_reduce + torch.log(1 + torch.exp(lse_block - lse_reduce))
        #         x_reduce = torch.exp(lse_reduce - lse_new) * x_reduce + torch.exp(
        #             lse_block - lse_new) * x_block
        #         lse_reduce = lse_new
        #     x = x_reduce.type_as(x)
        #     print("x.shape final", x.shape)

        # v proj
        if self.mla_weight_dtype == torch.bfloat16:
            x = torch.einsum("shc,hdc->shd", x, wkv_b[:, -self.v_head_dim:])
        elif self.mla_weight_dtype == torch.float8_e4m3fn:
            x, x_scales = torch.ops.trtllm.fp8_batched_quantize_1x128_permute102(
                x)
            x = torch.ops.trtllm.fp8_block_scaling_bmm(x, self.v_b_proj,
                                                       x_scales,
                                                       self.v_b_proj_scale)
            x = x.transpose(0, 1)
        else:
            raise NotImplementedError(
                f"Unsupported dtype: {self.mla_weight_dtype}")
        # proj
        x = self.wo(x.flatten(1))
        return x

    def dummy_forward(self, x: torch.Tensor):
        seqlen, hidden_dim = x.size()
        end_pos = seqlen
        # rope param
        if self.freqs_cis is None or seqlen > self.freqs_cis.shape[0]:
            self.freqs_cis = self.precompute_freqs_cis(max_seq_len=seqlen,
                                                       **self.rope_params).to(
                                                           x.device)
        # get mask
        mask = None
        if seqlen > 1:
            mask = torch.full((end_pos, end_pos), float("-inf"),
                              device='cuda').triu_(1)
            mask = mask[-seqlen:]
        # proj
        if self.is_lite:
            q = self.wq(x)
        else:
            q = self.wq_a(x.view(-1, hidden_dim))
            q = self.q_norm(q).type_as(x)
            q = self.wq_b(q).view(seqlen, -1)
        # q rope
        q = q.view(seqlen, self.n_local_heads, self.qk_head_dim)
        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_pe = self.apply_rotary_emb(q_pe, self.freqs_cis[0:end_pos])
        # kv proj a
        kv = self.wkv_a(x.view(-1, hidden_dim)).view(seqlen, -1)
        # kv rope
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim],
                               dim=-1)
        k_pe = self.apply_rotary_emb(k_pe.unsqueeze(1),
                                     self.freqs_cis[0:end_pos])
        # q_nope proj
        if self.mla_weight_dtype == torch.bfloat16:
            wkv_b = self.wkv_b.weight.view(self.n_local_heads, -1,
                                           self.kv_lora_rank)
            q_nope = torch.einsum("shd,hdc->shc", q_nope,
                                  wkv_b[:, :self.qk_nope_head_dim])
        elif self.mla_weight_dtype == torch.float8_e4m3fn:
            q_nope, q_nope_scales = torch.ops.trtllm.fp8_batched_quantize_1x128_permute102(
                q_nope)
            q_nope = torch.ops.trtllm.fp8_block_scaling_bmm(
                q_nope, self.k_b_proj_trans, q_nope_scales,
                self.k_b_proj_trans_scale)
            q_nope = q_nope.transpose(0, 1)
        else:
            raise NotImplementedError(
                f"Unsupported dtype: {self.mla_weight_dtype}")
        # get kv and pe states
        kv_states = self.kv_norm(kv).type(self.dtype)
        pe_states = k_pe.squeeze(1)
        # attention
        scores = (torch.einsum("shc,tc->sht", q_nope, kv_states) + torch.einsum(
            "shr,tr->sht", q_pe, pe_states)) * self.softmax_scale
        if mask is not None:
            scores += mask.unsqueeze(1)
        scores = scores.softmax(dim=-1, dtype=torch.float32).type_as(x)
        x = torch.einsum("sht,tc->shc", scores, kv_states)
        # v proj
        if self.mla_weight_dtype == torch.bfloat16:
            x = torch.einsum("shc,hdc->shd", x, wkv_b[:, -self.v_head_dim:])
        elif self.mla_weight_dtype == torch.float8_e4m3fn:
            x, x_scales = torch.ops.trtllm.fp8_batched_quantize_1x128_permute102(
                x)
            x = torch.ops.trtllm.fp8_block_scaling_bmm(x, self.v_b_proj,
                                                       x_scales,
                                                       self.v_b_proj_scale)
            x = x.transpose(0, 1)
        else:
            raise NotImplementedError(
                f"Unsupported dtype: {self.mla_weight_dtype}")
        # proj
        x = self.wo(x.flatten(1))
        return x

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.LongTensor],
        attn_metadata: Optional[AttentionMetadata],
        all_reduce_params: Optional[AllReduceParams],
        **kwargs,
    ) -> torch.Tensor:
        # Control reduce output.
        self.wo.reduce_output = all_reduce_params.enable_allreduce

        # import pdb; pdb.set_trace()
        if attn_metadata is None or attn_metadata.kv_cache_manager is None:
            return self.dummy_forward(hidden_states)

        print("MLA Vanilla forward start", hidden_states.shape)
        # print("attn_metadata.seq_lens", attn_metadata.seq_lens) # e.g. [4,5,1]
        max_seq_len = attn_metadata.kv_cache_manager.max_seq_len
        if self.freqs_cis is None or max_seq_len > self.freqs_cis.shape[0]:
            self.freqs_cis = self.precompute_freqs_cis(max_seq_len=max_seq_len,
                                                       **self.rope_params).to(
                                                           hidden_states.device)

        past_seen_tokens = attn_metadata.kv_cache_params.num_cached_tokens_per_seq
        cache_indices = [
            block_ids[0] for block_ids in attn_metadata.block_ids_per_seq
        ]
        kv_cache_tensor = attn_metadata.kv_cache_manager.get_buffers(
            self.layer_idx)

        assert len(cache_indices) == len(past_seen_tokens)
        assert len(cache_indices) == attn_metadata.seq_lens.nelement()

        # Ulysses preprocess
        if getattr(self, 'count', None) is None:
            self.count = 0
        # if self.count % 4 == 0:
        # import pdb;pdb.set_trace()
        self.count += 1

        offset = 0
        attn_outputs = []
        for i, seq_len in enumerate(attn_metadata.seq_lens):
            # allGather for gen, 先确保正确性，后续优化overlap
            single_hidden_state = hidden_states[offset:offset + seq_len, :]
            past_seen_token = past_seen_tokens[i]
            cache_idx = cache_indices[i]
            attn_output = self._single_request_forward(single_hidden_state,
                                                       kv_cache_tensor,
                                                       past_seen_token,
                                                       cache_idx)
            attn_outputs.append(attn_output)
            offset += seq_len

        attn_output = torch.cat(attn_outputs, dim=0).contiguous()

        # Ulysses postprocess

        return attn_output
