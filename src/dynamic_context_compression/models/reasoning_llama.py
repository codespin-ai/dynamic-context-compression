from transformers import LlamaForCausalLM, LlamaConfig
import torch
from typing import Optional, Tuple, Union
from transformers.modeling_outputs import CausalLMOutputWithPast


# =============================================================================
# ReasoningLlamaConfig: Extending the default LlamaConfig to support additional
# special tokens used for reasoning context management.
# =============================================================================
class ReasoningLlamaConfig(LlamaConfig):
    """
    Extended Llama configuration to support reasoning tokens.

    This configuration adds several special tokens to denote parts of a reasoning
    block within the text. These tokens include markers for:
      - The start and end of a reasoning block.
      - Failure and success markers within the reasoning block.

    Example usage:
      config = ReasoningLlamaConfig(
          reasoning_start_token="<REASONING_START>",
          reasoning_end_token="<REASONING_END>",
          reasoning_failure_start_token="<REASONING_FAILURE_START>",
          reasoning_failure_end_token="<REASONING_FAILURE_END>",
          reasoning_success_start_token="<REASONING_SUCCESS_START>",
          reasoning_success_end_token="<REASONING_SUCCESS_END>"
      )
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
        # Initialize the parent configuration with other provided keyword arguments.
        super().__init__(**kwargs)
        # Store the reasoning tokens in the configuration.
        self.reasoning_start_token = reasoning_start_token
        self.reasoning_failure_start_token = reasoning_failure_start_token
        self.reasoning_failure_end_token = reasoning_failure_end_token
        self.reasoning_success_start_token = reasoning_success_start_token
        self.reasoning_success_end_token = reasoning_success_end_token
        self.reasoning_end_token = reasoning_end_token


# =============================================================================
# ReasoningLlamaForCausalLM: A custom causal language model that adds reasoning
# context management to the base Llama model.
# =============================================================================
class ReasoningLlamaForCausalLM(LlamaForCausalLM):
    """
    Custom Llama model with advanced reasoning capabilities.

    This model builds upon the base LlamaForCausalLM to enable dynamic reasoning
    context management. It includes:
      - A method to detect and replace detailed reasoning blocks with a concise
        summary.
      - KV cache slicing so that the model's past key values remain consistent
        after modifications to the input sequence.

    Example usage:
      >>> # Assume `tokenizer` and `config` have been initialized properly.
      >>> model = ReasoningLlamaForCausalLM(config)
      >>> input_text = (
      ...    "Problem: What is 2+2? <REASONING_START> Detailed reasoning... "
      ...    "<REASONING_SUCCESS_START>4<REASONING_SUCCESS_END> <REASONING_END> Final answer is 4."
      ... )
      >>> input_ids = tokenizer(input_text, return_tensors="pt").input_ids
      >>> new_input_ids, attention_mask = model.replace_reasoning_context(input_ids, tokenizer)
    """

    # Bind this model to the extended configuration class.
    config_class = ReasoningLlamaConfig

    def __init__(self, config: ReasoningLlamaConfig):
        # Initialize the parent LlamaForCausalLM with the provided configuration.
        super().__init__(config)
        # Optionally, store additional reasoning-related data here.
        self.reasoning_cache = None

    def replace_reasoning_context(
        self,
        input_ids: torch.Tensor,
        tokenizer,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple[Tuple[torch.Tensor]]]]:
        """
        Replace the reasoning context within the input with a summary.

        This method performs the following steps:
          1. Decodes the input token IDs into text while preserving special tokens.
          2. Searches for the reasoning block defined by the tokens <REASONING_START>
             and <REASONING_END>.
          3. Within the block, it checks for failure markers. If found, it extracts the
             summary between <REASONING_FAILURE_START> and <REASONING_FAILURE_END>.
          4. Otherwise, it checks for success markers and extracts the text between
             <REASONING_SUCCESS_START> and <REASONING_SUCCESS_END>.
          5. It then removes the entire reasoning block and, if a summary was found,
             inserts the summary in its place.
          6. The new text is re-tokenized to generate updated input IDs and a corresponding
             attention mask.
          7. If a past_key_values is provided, it prunes the cache to keep only
             tokens preceding the reasoning block.

        Args:
            input_ids (torch.Tensor): Tensor containing the token IDs.
            tokenizer: The tokenizer used to convert between text and tokens.
            past_key_values (Optional[Tuple[Tuple[torch.Tensor]]]): Optional past key values cache.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Optional[Tuple[Tuple[torch.Tensor]]]]:
            A tuple of (new_input_ids, new_attention_mask, pruned_past_key_values).
        """
        # ---------------------------------------------------------------------
        # Step 1: Decode the input_ids to a text string while keeping special tokens.
        # ---------------------------------------------------------------------
        decoded = tokenizer.decode(input_ids[0], skip_special_tokens=False)

        # ---------------------------------------------------------------------
        # Step 2: Locate the reasoning block using the start and end markers.
        # ---------------------------------------------------------------------
        start_token = self.config.reasoning_start_token
        end_token = self.config.reasoning_end_token
        start_idx = decoded.find(start_token)
        end_idx = decoded.find(end_token)
        if start_idx == -1 or end_idx == -1:
            # If the reasoning block is not found, simply return the original tensors and past_key_values.
            return (
                input_ids,
                torch.ones_like(input_ids, dtype=torch.long),
                past_key_values,
            )

        # Extract the complete reasoning block (from <REASONING_START> to <REASONING_END>).
        block = decoded[start_idx : end_idx + len(end_token)]

        # ---------------------------------------------------------------------
        # Step 3: Extract the summary from the reasoning block.
        #         Check for failure markers first, then success markers.
        # ---------------------------------------------------------------------
        summary = ""
        # Check if the failure markers exist.
        if (
            self.config.reasoning_failure_start_token in block
            and self.config.reasoning_failure_end_token in block
        ):
            fs = block.find(self.config.reasoning_failure_start_token)
            fe = block.find(self.config.reasoning_failure_end_token)
            summary = block[
                fs + len(self.config.reasoning_failure_start_token) : fe
            ].strip()
        # Otherwise, check if the success markers exist.
        elif (
            self.config.reasoning_success_start_token in block
            and self.config.reasoning_success_end_token in block
        ):
            ss = block.find(self.config.reasoning_success_start_token)
            se = block.find(self.config.reasoning_success_end_token)
            summary = block[
                ss + len(self.config.reasoning_success_start_token) : se
            ].strip()

        # ---------------------------------------------------------------------
        # Step 4: Reconstruct the text by removing the reasoning block and
        #         inserting the summary (if any) in its place.
        # ---------------------------------------------------------------------
        new_decoded = (
            decoded[:start_idx]  # Text before the reasoning block.
            + (summary if summary else "")  # Insert the summary or nothing.
            + decoded[end_idx + len(end_token) :]  # Text after the reasoning block.
        )

        # ---------------------------------------------------------------------
        # Step 5: Re-tokenize the new text to create updated input_ids.
        # ---------------------------------------------------------------------
        new_input_ids = tokenizer(
            new_decoded, add_special_tokens=False, return_tensors="pt"
        ).input_ids.to(input_ids.device)

        # ---------------------------------------------------------------------
        # Step 6: Create a new attention mask (all ones) matching the new tokens.
        # ---------------------------------------------------------------------
        new_attention_mask = torch.ones_like(new_input_ids, dtype=torch.long)

        # ---------------------------------------------------------------------
        # Step 7: Prune the past_key_values (KV cache) if provided
        # ---------------------------------------------------------------------
        # Determine the cutoff point BEFORE the reasoning start token
        reasoning_start_token_id = tokenizer.convert_tokens_to_ids(start_token)
        start_token_indices = torch.where(input_ids[0] == reasoning_start_token_id)[0]

        # Cut exactly BEFORE the reasoning start token
        if len(start_token_indices) > 0:
            cutoff = start_token_indices[0].item()
        else:
            # Fallback if token is not found
            cutoff = input_ids.shape[1]

        # Prune past_key_values if provided
        pruned_past_key_values = None
        if past_key_values is not None:
            pruned_past_key_values = tuple(
                tuple(layer_cache[:, :cutoff, :] for layer_cache in layer_pair)
                for layer_pair in past_key_values
            )

        # Return the updated token IDs, attention mask, and pruned past_key_values
        return new_input_ids, new_attention_mask, pruned_past_key_values

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """
        Forward pass for the model.

        This method delegates the forward pass to the parent LlamaForCausalLM.
        It accepts input_ids, attention_mask, and an optional KV cache (past_key_values)
        to support efficient autoregressive generation.

        Args:
            input_ids (torch.Tensor, optional): Input token IDs.
            attention_mask (torch.Tensor, optional): Mask indicating which tokens to attend to.
            past_key_values (tuple, optional): Cached key/value tensors from previous passes.
            kwargs: Additional keyword arguments.

        Returns:
            CausalLMOutputWithPast: Model output including logits and past key values.
        """
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            **kwargs,
        )
