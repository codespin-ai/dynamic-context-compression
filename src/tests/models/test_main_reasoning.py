import pytest


@pytest.mark.usefixtures("llama_model_setup")
class TestMainReasoning:
    """
    Integration tests for the reasoning mechanism in the main workflow.
    """

    def test_reasoning_block_generation(self):
        """
        Test that the model can generate a meaningful reasoning block.
        """
        prompt = """Solve this problem step by step. 
Use these special tokens:
- Start reasoning with <REASONING_START>
- Mark successful solution with <REASONING_SUCCESS_START> and </REASONING_SUCCESS_END>
- End reasoning with <REASONING_END>

Problem: If a train travels 120 miles in 2 hours, what is its speed?

Begin your reasoning:
<REASONING_START>"""

        inputs = self.tokenizer(prompt, return_tensors="pt")

        # Generate with some randomness to explore reasoning
        outputs = self.model.generate(
            inputs.input_ids,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
        )

        decoded_output = self.tokenizer.decode(outputs[0], skip_special_tokens=False)

        # Validate reasoning block structure
        assert "<REASONING_START>" in decoded_output, "Reasoning block should start"
        assert "<REASONING_END>" in decoded_output, "Reasoning block should end"

        # Check for either success or failure markers
        assert any(
            marker in decoded_output
            for marker in ["<REASONING_SUCCESS_START>", "<REASONING_FAILURE_START>"]
        ), "Reasoning block should have a conclusive marker"

        # Optional: Check that the reasoning block contains some problem-solving steps
        reasoning_start = decoded_output.find("<REASONING_START>")
        reasoning_end = decoded_output.find("<REASONING_END>")
        reasoning_block = decoded_output[reasoning_start:reasoning_end]

        # Ensure the reasoning block contains computational or logical steps
        assert (
            len(reasoning_block.split()) > 10
        ), "Reasoning block should have detailed steps"

    def test_reasoning_context_replacement(self):
        """
        Test the replacement of reasoning context after generation.
        """
        prompt = "What is 17 multiplied by 6?"

        inputs = self.tokenizer(prompt, return_tensors="pt")

        # Generate with reasoning
        outputs = self.model.generate(
            inputs.input_ids,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
        )

        decoded_output = self.tokenizer.decode(outputs[0], skip_special_tokens=False)

        # Convert to input_ids
        output_ids = self.tokenizer(decoded_output, return_tensors="pt").input_ids

        # Test context replacement
        new_input_ids, new_attention_mask, past_key_values = (
            self.model.replace_reasoning_context(output_ids, self.tokenizer)
        )

        # Decode the new input
        simplified_output = self.tokenizer.decode(
            new_input_ids[0], skip_special_tokens=True
        )

        # Check that reasoning context is replaced with a concise result
        assert len(simplified_output) < len(decoded_output)
        assert any(
            str(num) in simplified_output for num in range(1, 200)
        ), "Output should contain a numeric result"
