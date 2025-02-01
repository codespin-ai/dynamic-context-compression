import os
import torch
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoTokenizer
from models.custom_llama import CustomLlamaForCausalLM, CustomLlamaConfig


def is_newly_generated_special_token(
    decoded_output: str, original_prompt: str, special_tokens: list
) -> bool:
    """
    Determine if a newly generated token is a special token
    and not part of the original prompt.

    Args:
        decoded_output (str): Newly decoded output
        original_prompt (str): Original input prompt
        special_tokens (list): List of special tokens to check

    Returns:
        bool: Whether the token is a newly generated special token
    """
    # Remove original prompt from decoded output
    generated_portion = decoded_output[len(original_prompt) :]

    # Check if any special token is in the generated portion
    return any(
        token in generated_portion
        and generated_portion.index(token)
        == 0  # Must be at the start of new generation
        for token in special_tokens
    )


def main():
    # Load environment variables
    load_dotenv()

    # Authenticate with Hugging Face
    token = os.getenv("HUGGINGFACE_TOKEN")
    login(token=token)

    # Model details
    model_name = "meta-llama/Llama-3.2-3B"

    # Initialize tokenizer with explicit padding
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # Special reasoning tokens to track
    special_tokens = [
        "<REASONING_SUCCESS_END>",
        "<REASONING_FAILURE_END>",
        "<REASONING_END>",
    ]

    # Add special reasoning tokens
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
    tokenizer.add_special_tokens(special_tokens_dict)

    # Create custom config
    config = CustomLlamaConfig.from_pretrained(model_name)

    # Initialize custom model for CPU
    model = CustomLlamaForCausalLM.from_pretrained(
        model_name, config=config, device_map="cpu", torch_dtype=torch.float32
    )

    # Resize token embeddings after adding special tokens
    model.resize_token_embeddings(len(tokenizer))

    # Argparse for command-line prompt
    import argparse

    parser = argparse.ArgumentParser(description="Reasoning-based Problem Solver")
    parser.add_argument(
        "--prompt", type=str, required=True, help="Problem prompt to solve"
    )
    args = parser.parse_args()

    # Construct initial prompt with reasoning context
    full_prompt = f"""Task: {args.prompt}

Your task is to reason about the solution:
1. Break down the problem, and solve it step by step.
2. Otherwise you must try all possibilities and combinations
3. Use these EXACT special tokens:
   - <REASONING_START> at reasoning beginning
   - <REASONING_SUCCESS_START> if solution found
   - <REASONING_SUCCESS_END> after solution
   - <REASONING_FAILURE_START> if unsolvable
   - <REASONING_FAILURE_END> after failure explanation
   - <REASONING_END> at reasoning end

Let's start.
<REASONING_START>
"""

    # Tokenize initial input
    inputs = tokenizer(
        full_prompt, return_tensors="pt", padding=True, add_special_tokens=True
    )

    # Initialize generation variables
    generated_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    original_prompt = full_prompt

    # Failure recovery mechanism
    failure_recovery_attempts = 0
    max_failure_recovery_attempts = 3

    while failure_recovery_attempts < max_failure_recovery_attempts:
        print(f"\nReasoning Attempt {failure_recovery_attempts + 1}:")

        # Generation loop
        for _ in range(500):  # Limit generation to prevent infinite loops
            outputs = model.generate(
                generated_ids,
                max_new_tokens=1,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
                attention_mask=attention_mask,
            )

            # Get the newly generated token
            new_token = outputs[0, generated_ids.shape[1] :]

            # Decode and print the new token
            new_token_text = tokenizer.decode(new_token, skip_special_tokens=False)
            print(new_token_text, end="", flush=True)

            # Update generated_ids for next iteration
            generated_ids = outputs

            # Decode full output
            decoded_output = tokenizer.decode(generated_ids[0])

            # Advanced special token detection
            if is_newly_generated_special_token(
                decoded_output, original_prompt, special_tokens
            ):
                if "<REASONING_SUCCESS_END>" in decoded_output:
                    # Extract solution between success markers
                    solution_start = decoded_output.find(
                        "<REASONING_SUCCESS_START>"
                    ) + len("<REASONING_SUCCESS_START>")
                    solution_end = decoded_output.find("<REASONING_SUCCESS_END>")
                    solution = decoded_output[solution_start:solution_end].strip()
                    print(f"\n\nFinal Solution:\n{solution}")
                    return  # Exit main function

                elif "<REASONING_FAILURE_END>" in decoded_output:
                    # Failure Recovery Mechanism
                    failure_start = decoded_output.find(
                        "<REASONING_FAILURE_START>"
                    ) + len("<REASONING_FAILURE_START>")
                    failure_end = decoded_output.find("<REASONING_FAILURE_END>")
                    failure_explanation = decoded_output[
                        failure_start:failure_end
                    ].strip()

                    print(
                        f"\n\nReasoning Failure (Attempt {failure_recovery_attempts + 1}):\n{failure_explanation}"
                    )

                    # Prepare recovery prompt
                    recovery_prompt = f"""
Previous attempt failed. Here's the failure explanation:
{failure_explanation}

Let's try a different approach to solve: {args.prompt}

<REASONING_START>
Recovering from previous failure. New strategy:
"""

                    # Use model's custom method to prune and reset context
                    recovered_input_ids, recovered_attention_mask = (
                        model.replace_reasoning_context(generated_ids, tokenizer)
                    )

                    # Append recovery prompt to pruned context
                    recovery_input = tokenizer(
                        recovery_prompt, return_tensors="pt", add_special_tokens=True
                    )

                    # Combine pruned context with recovery prompt
                    generated_ids = torch.cat(
                        [recovered_input_ids, recovery_input.input_ids], dim=1
                    )
                    attention_mask = torch.cat(
                        [recovered_attention_mask, recovery_input.attention_mask], dim=1
                    )

                    failure_recovery_attempts += 1
                    break  # Restart generation loop

            # Stop if EOS token is generated
            if tokenizer.eos_token_id in outputs[0]:
                break

        # Exit if max recovery attempts reached
        if failure_recovery_attempts >= max_failure_recovery_attempts:
            print("\nFailed to solve the problem after multiple attempts.")
            break

    print("\n")  # New line after response


if __name__ == "__main__":
    main()
