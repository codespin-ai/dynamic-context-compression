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

    # Construct prompt with reasoning context
    full_prompt = f"""Task: {args.prompt}

You must solve this systematically:
1. Break down the problem
2. Develop a step-by-step solution strategy
3. Use these EXACT special tokens:
   - <REASONING_START> at reasoning beginning
   - <REASONING_SUCCESS_START> if solution found
   - <REASONING_SUCCESS_END> after solution
   - <REASONING_FAILURE_START> if unsolvable
   - <REASONING_FAILURE_END> after failure explanation
   - <REASONING_END> at reasoning end

<REASONING_START>
Let's solve: {args.prompt}
"""

    # Tokenize input
    inputs = tokenizer(
        full_prompt, return_tensors="pt", padding=True, add_special_tokens=True
    )

    # Generate text with streaming
    generated_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    # Store original prompt for special token detection
    original_prompt = full_prompt

    print("Reasoning Process:")
    for _ in range(200):  # Limit generation to prevent infinite loops
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
            # Determine which special token triggered the end
            if "<REASONING_SUCCESS_END>" in decoded_output:
                # Extract solution between success markers
                solution_start = decoded_output.find("<REASONING_SUCCESS_START>") + len(
                    "<REASONING_SUCCESS_START>"
                )
                solution_end = decoded_output.find("<REASONING_SUCCESS_END>")
                solution = decoded_output[solution_start:solution_end].strip()
                print(f"\n\nFinal Solution:\n{solution}")
                break

            elif "<REASONING_FAILURE_END>" in decoded_output:
                # Extract failure explanation
                failure_start = decoded_output.find("<REASONING_FAILURE_START>") + len(
                    "<REASONING_FAILURE_START>"
                )
                failure_end = decoded_output.find("<REASONING_FAILURE_END>")
                failure_explanation = decoded_output[failure_start:failure_end].strip()
                print(f"\n\nReasoning Failure:\n{failure_explanation}")
                break

        # Stop if EOS token is generated
        if tokenizer.eos_token_id in outputs[0]:
            break

    print("\n")  # New line after response


if __name__ == "__main__":
    main()
