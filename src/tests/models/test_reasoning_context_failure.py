import pytest
import torch


@pytest.mark.usefixtures("llama_model_setup")
class TestReasoningContextFailure:
    """
    Unit tests for failure reasoning context replacements.
    """

    def test_reasoning_failure_replacement(self):
        """
        Test that a reasoning block containing failure markers is replaced correctly.
        """
        input_text = (
            "Problem: What is 2+2? <REASONING_START> I don't know. "
            "<REASONING_FAILURE_START> Could not solve. <REASONING_FAILURE_END> "
            "<REASONING_END> I give up."
        )
        # Convert the input text to token IDs.
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids

        # Invoke the method to replace the reasoning context.
        new_input_ids, new_attention_mask, past_key_values = (
            self.model.replace_reasoning_context(input_ids, self.tokenizer)
        )

        # Define the expected output text after replacement.
        expected_text = "Problem: What is 2+2? Could not solve. I give up."
        # Decode the new token IDs into text (skipping special tokens).
        decoded_text = self.tokenizer.decode(new_input_ids[0], skip_special_tokens=True)

        # Assert that the decoded text matches the expected text.
        assert decoded_text == expected_text
        # Assert that the attention mask has the correct shape.
        assert new_attention_mask.shape == new_input_ids.shape
        # Assert that past_key_values is None since no cache was provided
        assert past_key_values is None
