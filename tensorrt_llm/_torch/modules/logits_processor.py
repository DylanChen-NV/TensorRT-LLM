import os

import torch
import torch.nn as nn

from ..attention_backend import AttentionMetadata
from .linear import Linear

dump_count = 0


class LogitsProcessor(nn.Module):

    def __init__(self):
        super().__init__()
        self.return_hidden_states = os.getenv("RETURN_HIDDEN_STATES",
                                              default=None)

    def forward(self,
                hidden_states: torch.Tensor,
                lm_head: Linear,
                attn_metadata: AttentionMetadata,
                return_context_logits: bool = False) -> torch.Tensor:

        hidden_states_origin = None
        if self.return_hidden_states:
            hidden_states_origin = hidden_states

        ##############  debug
        # print("debug attn_metadata.seq_lens", attn_metadata.seq_lens)
        # temp = torch.split(hidden_states[:,0], attn_metadata.seq_lens.tolist(), dim=0)
        # for i, t in enumerate(temp):
        #     print(f"debug {i}th hidden_states: {t.tolist()}")
        ##############  debug

        if not return_context_logits:
            if attn_metadata is not None:
                last_tokens = torch.cumsum(
                    attn_metadata.seq_lens_cuda,
                    dim=0,
                    dtype=torch.long,
                ) - 1
                hidden_states = hidden_states[last_tokens]
            else:
                hidden_states = hidden_states[-1]

        logits = lm_head(hidden_states)
        logits = logits.float()
        if not self.return_hidden_states:
            return logits
        else:
            return logits, hidden_states_origin, attn_metadata.seq_lens
