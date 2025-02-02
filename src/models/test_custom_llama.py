import unittest
import torch
from transformers import AutoTokenizer
from .custom_llama import CustomLlamaForCausalLM, CustomLlamaConfig


# =============================================================================
# TestReplaceReasoningContext: Unit tests for the replace_reasoning_context method.
# =============================================================================
class TestReplaceReasoningContext(unittest.TestCase):
    """
    Unit tests for the 'replace_reasoning_context' method of CustomLlamaForCausalLM.

    These tests verify that:
      - A reasoning block with success markers is replaced correctly.
      - A reasoning block with failure markers is replaced correctly.
      - If no reasoning markers are present, the original input is returned.
      - An empty summary within a reasoning block is handled properly.
    """

    def setUp(self):
        """
        Initialize the tokenizer and model for testing.

        This method performs the following steps:
          - Loads the tokenizer for a given model.
          - Adds the additional special tokens required for reasoning.
          - Loads the custom model with its configuration.
          - Resizes the token embeddings to account for the newly added tokens.

        Example:
          >>> test_instance = TestReplaceReasoningContext()
          >>> test_instance.setUp()
        """
        # Specify the model name. (For testing, consider using a smaller model if necessary.)
        self.model_name = "meta-llama/Llama-3.2-3B"  # Change to a smaller model if faster testing is needed.
        # Load the tokenizer from the pre-trained model.
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # Set the pad token to be the same as the end-of-sequence token.
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Add special tokens for reasoning context to the tokenizer.
        special_tokens_dict = {
            "additional_special_tokens": [
                "<REASONING_START>",
                "<REASONING_FAILURE_START>",
                "<REASONING_FAILURE_END>",
                "<REASONING_SUCCESS_START>",
                "<REASONING_SUCCESS_END>",
                "<REASONING_END>",
            ]
        }
        self.tokenizer.add_special_tokens(special_tokens_dict)

        # Load the custom configuration and model.
        config = CustomLlamaConfig.from_pretrained(self.model_name)
        self.model = CustomLlamaForCausalLM.from_pretrained(
            self.model_name, config=config, device_map="cpu", torch_dtype=torch.float32
        )
        # Resize model embeddings to include the additional special tokens.
        self.model.resize_token_embeddings(len(self.tokenizer))

    def test_replace_reasoning_context_success(self):
        """
        Test that a reasoning block containing success markers is replaced correctly.

        Input example:
          "Problem: What is 2+2? <REASONING_START> The answer is 4.
           <REASONING_SUCCESS_START> 4 <REASONING_SUCCESS_END> <REASONING_END> The answer is 4."

        Expected output:
          "Problem: What is 2+2? 4 The answer is 4."
        """
        input_text = (
            "Problem: What is 2+2? <REASONING_START> The answer is 4. "
            "<REASONING_SUCCESS_START> 4 <REASONING_SUCCESS_END> <REASONING_END> The answer is 4."
        )
        # Convert the input text to token IDs.
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids

        # Invoke the method to replace the reasoning context.
        new_input_ids, new_attention_mask = self.model.replace_reasoning_context(
            input_ids, self.tokenizer
        )

        # Define the expected output text after replacement.
        expected_text = "Problem: What is 2+2? 4 The answer is 4."
        # Decode the new token IDs into text (skipping any special tokens).
        decoded_text = self.tokenizer.decode(new_input_ids[0], skip_special_tokens=True)

        # Assert that the decoded text matches the expected text.
        self.assertEqual(decoded_text, expected_text)
        # Assert that the attention mask has the same shape as the new input IDs.
        self.assertEqual(new_attention_mask.shape, new_input_ids.shape)

    def test_replace_reasoning_context_failure(self):
        """
        Test that a reasoning block containing failure markers is replaced correctly.

        Input example:
          "Problem: What is 2+2? <REASONING_START> I don't know.
           <REASONING_FAILURE_START> Could not solve. <REASONING_FAILURE_END>
           <REASONING_END> I give up."

        Expected output:
          "Problem: What is 2+2? Could not solve. I give up."
        """
        input_text = (
            "Problem: What is 2+2? <REASONING_START> I don't know. "
            "<REASONING_FAILURE_START> Could not solve. <REASONING_FAILURE_END> "
            "<REASONING_END> I give up."
        )
        # Convert the input text to token IDs.
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids

        # Invoke the method to replace the reasoning context.
        new_input_ids, new_attention_mask = self.model.replace_reasoning_context(
            input_ids, self.tokenizer
        )

        # Define the expected output text after replacement.
        expected_text = "Problem: What is 2+2? Could not solve. I give up."
        # Decode the new token IDs into text (skipping special tokens).
        decoded_text = self.tokenizer.decode(new_input_ids[0], skip_special_tokens=True)

        # Assert that the decoded text matches the expected text.
        self.assertEqual(decoded_text, expected_text)
        # Assert that the attention mask has the correct shape.
        self.assertEqual(new_attention_mask.shape, new_input_ids.shape)

    def test_replace_reasoning_context_no_markers(self):
        """
        Test that if there are no reasoning markers, the original input is returned unchanged.

        Input example:
          "Problem: What is 2+2? The answer is 4."

        Expected output:
          The same text as the input.
        """
        input_text = "Problem: What is 2+2? The answer is 4."
        # Convert the input text to token IDs.
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids

        # Invoke the method.
        new_input_ids, new_attention_mask = self.model.replace_reasoning_context(
            input_ids, self.tokenizer
        )

        # Decode the token IDs back into text.
        decoded_text = self.tokenizer.decode(new_input_ids[0], skip_special_tokens=True)
        # Assert that the text remains unchanged.
        self.assertEqual(decoded_text, "Problem: What is 2+2? The answer is 4.")
        # Assert that the returned token IDs are identical to the input token IDs.
        self.assertTrue(torch.equal(new_input_ids, input_ids))
        # Confirm that the attention mask is correctly set (all ones).
        self.assertTrue(
            torch.equal(
                new_attention_mask, torch.ones_like(input_ids, dtype=torch.long)
            )
        )

    def test_replace_reasoning_context_empty_summary(self):
        """
        Test that a reasoning block with an empty summary is removed properly.

        Input example:
          "Problem: What is 2+2? <REASONING_START>
           <REASONING_SUCCESS_START> <REASONING_SUCCESS_END>
           <REASONING_END> The answer is 4."

        Expected output:
          "Problem: What is 2+2?  The answer is 4."
          (Note: Extra spaces may occur where the reasoning block was removed.)
        """
        input_text = (
            "Problem: What is 2+2? <REASONING_START> "
            "<REASONING_SUCCESS_START> <REASONING_SUCCESS_END> <REASONING_END> The answer is 4."
        )
        # Convert the input text to token IDs.
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids

        # Invoke the method to replace the reasoning context.
        new_input_ids, new_attention_mask = self.model.replace_reasoning_context(
            input_ids, self.tokenizer
        )

        # Define the expected output text. Note the extra space where the block was removed.
        expected_text = "Problem: What is 2+2?  The answer is 4."
        # Decode the new token IDs into text.
        decoded_text = self.tokenizer.decode(new_input_ids[0], skip_special_tokens=True)

        # Assert that the decoded text matches the expected output.
        self.assertEqual(decoded_text, expected_text)
        # Assert that the attention mask's shape is correct.
        self.assertEqual(new_attention_mask.shape, new_input_ids.shape)


if __name__ == "__main__":
    # Run all the unit tests when this script is executed.
    unittest.main()
