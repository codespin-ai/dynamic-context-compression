from transformers import LlamaForCausalLM, LlamaConfig
import torch
from typing import Optional, Tuple
from transformers.modeling_outputs import CausalLMOutputWithPast


class CustomLlamaConfig(LlamaConfig):
    """
    Extended Llama configuration to support reasoning tokens.

    Adds special tokens for:
    - Reasoning start/end
    - Success/failure markers
    """

    def __init__(
        self,
        reasoning_start_token: str = "<REASONING_START>",
        reasoning_failure_start_token: str = "<REASONING_FAILURE_START>",
        reasoning_failure_end_token: str = "<REASONING_FAILURE_END>",
        reasoning_success_start_token: str = "<REASONING_SUCCESS_START>",
        reasoning_success_end_token: str = "<REASONING_SUCCESS_END>",
        reasoning_end_token: str = "<REASONING_END>",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.reasoning_start_token = reasoning_start_token
        self.reasoning_failure_start_token = reasoning_failure_start_token
        self.reasoning_failure_end_token = reasoning_failure_end_token
        self.reasoning_success_start_token = reasoning_success_start_token
        self.reasoning_success_end_token = reasoning_success_end_token
        self.reasoning_end_token = reasoning_end_token


class CustomLlamaForCausalLM(LlamaForCausalLM):
    """
    Custom Llama model with advanced reasoning capabilities.

    Key Features:
    - Dynamic reasoning context management
    - Precise KV cache slicing
    - Success/failure reasoning detection
    """

    config_class = CustomLlamaConfig

    def __init__(self, config: CustomLlamaConfig):
        super().__init__(config)
        # Custom reasoning-related attributes
        self.reasoning_cache = None

    def replace_reasoning_context(
        self,
        input_ids: torch.Tensor,
        tokenizer,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Replace reasoning context with summary.

        Precise steps:
        1. Detect reasoning markers
        2. Extract summary
        3. Slice input and KV cache
        4. Prepare for next generation phase

        Args:
            input_ids (torch.Tensor): Input token IDs
            tokenizer: Tokenizer for encoding/decoding

        Returns:
            Tuple of modified input_ids and attention_mask
        """
        # Convert special tokens to IDs for precise detection
        token_ids = {
            "start": tokenizer.convert_tokens_to_ids(self.config.reasoning_start_token),
            "failure_start": tokenizer.convert_tokens_to_ids(
                self.config.reasoning_failure_start_token
            ),
            "failure_end": tokenizer.convert_tokens_to_ids(
                self.config.reasoning_failure_end_token
            ),
            "success_start": tokenizer.convert_tokens_to_ids(
                self.config.reasoning_success_start_token
            ),
            "success_end": tokenizer.convert_tokens_to_ids(
                self.config.reasoning_success_end_token
            ),
            "end": tokenizer.convert_tokens_to_ids(self.config.reasoning_end_token),
        }

        # Find precise indices of all markers
        marker_indices = {
            name: torch.where(input_ids == token_id)[1]
            for name, token_id in token_ids.items()
        }

        # Validate marker presence and sequence
        if not all(len(indices) > 0 for indices in marker_indices.values()):
            return input_ids, torch.ones_like(input_ids, dtype=torch.long)

        # Determine reasoning outcome and extract summary
        if marker_indices["failure_start"]:
            start_idx = marker_indices["failure_start"][0]
            end_idx = marker_indices["failure_end"][0]
            summary_tokens = input_ids[0, start_idx + 1 : end_idx]
        elif marker_indices["success_start"]:
            start_idx = marker_indices["success_start"][0]
            end_idx = marker_indices["success_end"][0]
            summary_tokens = input_ids[0, start_idx + 1 : end_idx]
        else:
            return input_ids, torch.ones_like(input_ids, dtype=torch.long)

        # Precise input reconstruction
        new_input_ids = torch.cat(
            [
                input_ids[:, : marker_indices["start"][0]],  # Before reasoning
                summary_tokens,  # Extracted summary
                input_ids[:, marker_indices["end"][0] + 1 :],  # After reasoning
            ],
            dim=1,
        )

        # Precise KV cache slicing
        if hasattr(self, "past_key_values") and self.past_key_values is not None:
            self.past_key_values = tuple(
                tuple(
                    layer_cache[:, : marker_indices["start"][0], :]  # Slice precisely
                    for layer_cache in layer_pair
                )
                for layer_pair in self.past_key_values
            )

        # Create corresponding attention mask
        new_attention_mask = torch.ones_like(new_input_ids, dtype=torch.long)

        return new_input_ids, new_attention_mask

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        # Simply call the parent class's forward method
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            **kwargs,
        )
