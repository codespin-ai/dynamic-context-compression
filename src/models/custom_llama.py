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

        The function:
        1. Decodes the token ids into text (preserving special tokens).
        2. Locates the reasoning block defined by <REASONING_START> and <REASONING_END>.
        3. Within that block, if a failure or success marker is present,
           extracts the text between the corresponding markers as the summary.
        4. Removes the entire reasoning block and, if the summary is non-empty, inserts it.
        5. Re-tokenizes the resulting text and prunes the KV cache accordingly.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            tokenizer: Tokenizer for encoding/decoding.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Modified input_ids and corresponding attention_mask.
        """
        # Decode the input while preserving special tokens.
        decoded = tokenizer.decode(input_ids[0], skip_special_tokens=False)

        start_token = self.config.reasoning_start_token
        end_token = self.config.reasoning_end_token

        start_idx = decoded.find(start_token)
        end_idx = decoded.find(end_token)
        if start_idx == -1 or end_idx == -1:
            # If the reasoning markers are not found, return original tensors.
            return input_ids, torch.ones_like(input_ids, dtype=torch.long)

        # The reasoning block (from start token up to and including end token).
        block = decoded[start_idx : end_idx + len(end_token)]

        summary = ""
        # Check for a failure summary.
        if (
            self.config.reasoning_failure_start_token in block
            and self.config.reasoning_failure_end_token in block
        ):
            fs = block.find(self.config.reasoning_failure_start_token)
            fe = block.find(self.config.reasoning_failure_end_token)
            summary = block[
                fs + len(self.config.reasoning_failure_start_token) : fe
            ].strip()
        # Otherwise, check for a success summary.
        elif (
            self.config.reasoning_success_start_token in block
            and self.config.reasoning_success_end_token in block
        ):
            ss = block.find(self.config.reasoning_success_start_token)
            se = block.find(self.config.reasoning_success_end_token)
            summary = block[
                ss + len(self.config.reasoning_success_start_token) : se
            ].strip()

        # Reconstruct the text by removing the reasoning block and inserting the summary (if any).
        new_decoded = (
            decoded[:start_idx]
            + (summary if summary else "")
            + decoded[end_idx + len(end_token) :]
        )

        # Re-tokenize the new text.
        new_input_ids = tokenizer(
            new_decoded, add_special_tokens=False, return_tensors="pt"
        ).input_ids.to(input_ids.device)
        new_attention_mask = torch.ones_like(new_input_ids, dtype=torch.long)

        # Prune the KV cache up to the number of tokens before the reasoning block.
        # Tokenize the text before the reasoning block to determine the new cutoff.
        prefix_text = decoded[:start_idx]
        prefix_ids = tokenizer(
            prefix_text, add_special_tokens=False, return_tensors="pt"
        ).input_ids
        cutoff = prefix_ids.shape[1]

        if hasattr(self, "past_key_values") and self.past_key_values is not None:
            self.past_key_values = tuple(
                tuple(layer_cache[:, :cutoff, :] for layer_cache in layer_pair)
                for layer_pair in self.past_key_values
            )

        return new_input_ids, new_attention_mask

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        # Simply call the parent class's forward method.
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            **kwargs,
        )
