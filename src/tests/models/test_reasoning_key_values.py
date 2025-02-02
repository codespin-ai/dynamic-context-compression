import pytest
import torch


@pytest.mark.usefixtures("llama_model_setup")
class TestReasoningKeyValues:
    """
    Unit tests for reasoning context replacement with past key values.
    """

    def test_replace_with_past_key_values(self):
        """
        Test replacing reasoning context with past key values.
        """
        prefix_text = "Problem: What is 2+2?"
        full_text = (
            f"{prefix_text} <REASONING_START> "
            "<REASONING_SUCCESS_START> 4 <REASONING_SUCCESS_END> <REASONING_END> The answer is 4."
        )

        input_ids = self.tokenizer(full_text, return_tensors="pt").input_ids
        reasoning_start_token = "<REASONING_START>"
        reasoning_start_token_id = self.tokenizer.convert_tokens_to_ids(
            reasoning_start_token
        )

        mock_past_key_values = tuple(
            (
                torch.rand(1, 20, self.model.config.hidden_size),  # keys
                torch.rand(1, 20, self.model.config.hidden_size),  # values
            )
            for _ in range(self.model.config.num_hidden_layers)
        )

        new_input_ids, new_attention_mask, pruned_past_key_values = (
            self.model.replace_reasoning_context(
                input_ids, self.tokenizer, past_key_values=mock_past_key_values
            )
        )

        assert pruned_past_key_values is not None
        assert len(pruned_past_key_values) == len(mock_past_key_values)

        start_indices = torch.where(input_ids[0] == reasoning_start_token_id)[0]
        expected_pruned_length = (
            start_indices[0].item() if len(start_indices) > 0 else input_ids.shape[1]
        )

        for i, (layer_keys, layer_values) in enumerate(pruned_past_key_values):
            assert (
                layer_keys.shape[1] == expected_pruned_length
            ), f"Layer {i}: Unexpected pruned length"
