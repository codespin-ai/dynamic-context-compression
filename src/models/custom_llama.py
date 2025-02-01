from transformers import LlamaForCausalLM, LlamaConfig
import torch
from typing import Optional
from transformers.modeling_outputs import CausalLMOutputWithPast


class CustomLlamaConfig(LlamaConfig):
    # You can add custom configuration parameters here if needed
    pass


class CustomLlamaForCausalLM(LlamaForCausalLM):
    config_class = CustomLlamaConfig

    def __init__(self, config: CustomLlamaConfig):
        super().__init__(config)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        # Simply call the parent class's forward method
        return super().forward(
            input_ids=input_ids, attention_mask=attention_mask, **kwargs
        )
