import pytest


@pytest.mark.usefixtures("llama_model_setup")
class TestReasoningContextSuccess:
    """
    Unit tests for successful reasoning context replacements.
    """

    def test_simple_success_replacement(self):
        """
        Test basic successful reasoning block replacement.
        """
        input_text = (
            "Problem: What is 2+2? <REASONING_START> The answer is 4. "
            "<REASONING_SUCCESS_START> 4 <REASONING_SUCCESS_END> <REASONING_END> The answer is 4."
        )
        # Convert the input text to token IDs.
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids

        # Invoke the method to replace the reasoning context.
        new_input_ids, new_attention_mask, past_key_values = (
            self.model.replace_reasoning_context(input_ids, self.tokenizer)
        )

        # Define the expected output text after replacement.
        expected_text = "Problem: What is 2+2? 4 The answer is 4."
        # Decode the new token IDs into text (skipping any special tokens).
        decoded_text = self.tokenizer.decode(new_input_ids[0], skip_special_tokens=True)

        # Assert that the decoded text matches the expected text.
        assert decoded_text == expected_text
        # Assert that the attention mask has the same shape as the new input IDs.
        assert new_attention_mask.shape == new_input_ids.shape
        # Assert that past_key_values is None since no cache was provided
        assert past_key_values is None

    def test_complex_success_replacement(self):
        """
        Test a more complex successful reasoning block replacement.
        """
        input_text = (
            "Problem: Solve the equation x + 5 = 10. <REASONING_START> "
            "To solve this, I'll subtract 5 from both sides. "
            "<REASONING_SUCCESS_START> x = 5 <REASONING_SUCCESS_END> <REASONING_END> "
            "Therefore, the solution is x = 5."
        )

        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids

        new_input_ids, new_attention_mask, past_key_values = (
            self.model.replace_reasoning_context(input_ids, self.tokenizer)
        )

        expected_text = "Problem: Solve the equation x + 5 = 10. x = 5 Therefore, the solution is x = 5."
        decoded_text = self.tokenizer.decode(new_input_ids[0], skip_special_tokens=True)

        assert decoded_text == expected_text
