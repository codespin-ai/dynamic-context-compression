import unittest
import torch
from transformers import AutoTokenizer
from .custom_llama import CustomLlamaForCausalLM, CustomLlamaConfig


class TestReplaceReasoningContext(unittest.TestCase):

    def setUp(self):
        # Initialize tokenizer and model for testing
        self.model_name = (
            "meta-llama/Llama-3.2-3B"  # Or a smaller model for testing if needed
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

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

        config = CustomLlamaConfig.from_pretrained(self.model_name)
        self.model = CustomLlamaForCausalLM.from_pretrained(
            self.model_name, config=config, device_map="cpu", torch_dtype=torch.float32
        )
        self.model.resize_token_embeddings(len(self.tokenizer))

    def test_replace_reasoning_context_success(self):
        input_text = "Problem: What is 2+2? <REASONING_START> The answer is 4. <REASONING_SUCCESS_START> 4 <REASONING_SUCCESS_END> <REASONING_END> The answer is 4."
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids

        new_input_ids, new_attention_mask = self.model.replace_reasoning_context(
            input_ids, self.tokenizer
        )

        expected_text = "Problem: What is 2+2? 4 The answer is 4."
        decoded_text = self.tokenizer.decode(new_input_ids[0], skip_special_tokens=True)

        self.assertEqual(decoded_text, expected_text)
        self.assertEqual(new_attention_mask.shape, new_input_ids.shape)

    def test_replace_reasoning_context_failure(self):
        input_text = "Problem: What is 2+2? <REASONING_START> I don't know. <REASONING_FAILURE_START> Could not solve. <REASONING_FAILURE_END> <REASONING_END> I give up."
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids

        new_input_ids, new_attention_mask = self.model.replace_reasoning_context(
            input_ids, self.tokenizer
        )

        expected_text = "Problem: What is 2+2? Could not solve. I give up."
        decoded_text = self.tokenizer.decode(new_input_ids[0], skip_special_tokens=True)

        self.assertEqual(decoded_text, expected_text)
        self.assertEqual(new_attention_mask.shape, new_input_ids.shape)

    def test_replace_reasoning_context_no_markers(self):
        input_text = "Problem: What is 2+2? The answer is 4."
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids

        new_input_ids, new_attention_mask = self.model.replace_reasoning_context(
            input_ids, self.tokenizer
        )

        decoded_text = self.tokenizer.decode(new_input_ids[0], skip_special_tokens=True)
        self.assertEqual(
            decoded_text, "Problem: What is 2+2? The answer is 4."
        )  # Should return original input if no markers
        self.assertTrue(torch.equal(new_input_ids, input_ids))
        self.assertTrue(
            torch.equal(
                new_attention_mask, torch.ones_like(input_ids, dtype=torch.long)
            )
        )

    def test_replace_reasoning_context_empty_summary(self):
        input_text = "Problem: What is 2+2? <REASONING_START> <REASONING_SUCCESS_START> <REASONING_SUCCESS_END> <REASONING_END> The answer is 4."
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids

        new_input_ids, new_attention_mask = self.model.replace_reasoning_context(
            input_ids, self.tokenizer
        )

        expected_text = "Problem: What is 2+2?  The answer is 4."  # Two spaces because of the empty summary
        decoded_text = self.tokenizer.decode(new_input_ids[0], skip_special_tokens=True)

        self.assertEqual(decoded_text, expected_text)
        self.assertEqual(new_attention_mask.shape, new_input_ids.shape)


if __name__ == "__main__":
    unittest.main()
