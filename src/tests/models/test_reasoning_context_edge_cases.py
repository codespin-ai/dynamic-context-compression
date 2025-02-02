import pytest
import torch


@pytest.mark.usefixtures("llama_model_setup")
class TestReasoningContextEdgeCases:
    """
    Unit tests for edge cases in reasoning context replacement.
    """

    def test_no_reasoning_markers(self):
        """
        Test that if there are no reasoning markers, the original input is returned unchanged.
        """
        input_text = "Problem: What is 2+2? The answer is 4."
        # Convert the input text to token IDs.
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids

        # Invoke the method.
        new_input_ids, new_attention_mask, past_key_values = (
            self.model.replace_reasoning_context(input_ids, self.tokenizer)
        )

        # Decode the token IDs back into text.
        decoded_text = self.tokenizer.decode(new_input_ids[0], skip_special_tokens=True)
        # Assert that the text remains unchanged.
        assert decoded_text == "Problem: What is 2+2? The answer is 4."
        # Assert that the returned token IDs are identical to the input token IDs.
        assert torch.equal(new_input_ids, input_ids)

    def test_empty_summary(self):
        """
        Test that a reasoning block with an empty summary is removed properly.
        """
        input_text = (
            "Problem: What is 2+2? <REASONING_START> "
            "<REASONING_SUCCESS_START> <REASONING_SUCCESS_END> <REASONING_END> The answer is 4."
        )
        # Convert the input text to token IDs.
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids

        # Invoke the method to replace the reasoning context.
        new_input_ids, new_attention_mask, past_key_values = (
            self.model.replace_reasoning_context(input_ids, self.tokenizer)
        )

        # Define the expected output text. Note the extra space where the block was removed.
        expected_text = "Problem: What is 2+2?  The answer is 4."
        # Decode the new token IDs into text.
        decoded_text = self.tokenizer.decode(new_input_ids[0], skip_special_tokens=True)

        # Assert that the decoded text matches the expected output.
        assert decoded_text == expected_text
