import os
import torch
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoTokenizer
from dynamic_context_compression.models.reasoning_mllama import (
    ReasoningMllamaForCausalLM,
    ReasoningMllamaConfig,
)
from textwrap import dedent
import argparse
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# CPU Optimization Environment Variables
os.environ["OMP_NUM_THREADS"] = "16"  # Match your CPU core count
os.environ["MKL_NUM_THREADS"] = "16"
torch.set_float32_matmul_precision("high")


def is_newly_generated_special_token(
    decoded_output: str, original_prompt: str, special_tokens: list
) -> bool:
    """
    Efficiently determine if a newly generated token is a special token
    and not part of the original prompt.
    """
    generated_portion = decoded_output[len(original_prompt) :]

    for token in special_tokens:
        if token in generated_portion and generated_portion.index(token) == 0:
            return True
    return False


def setup_tokenizer_and_model(model_name: str):
    """
    Set up the tokenizer and model with special reasoning tokens.

    Args:
        model_name (str): Name of the model to load

    Returns:
        Tuple[AutoTokenizer, ReasoningMllamaForCausalLM]
    """
    # Optimization for CPU
    device = torch.device("cpu")
    dtype = torch.bfloat16  # Use bfloat16 for better performance

    logger.info(f"Using device: {device}, dtype: {dtype}")

    # Initialize tokenizer with explicit padding
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

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
    config = ReasoningMllamaConfig.from_pretrained(model_name)

    # Initialize custom model with CPU optimizations
    model = ReasoningMllamaForCausalLM.from_pretrained(
        model_name,
        config=config,
        device_map="cpu",
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )

    # Resize token embeddings after adding special tokens
    model.resize_token_embeddings(len(tokenizer))

    return tokenizer, model


def generate_reasoning_solution(model, tokenizer, prompt: str, max_tokens: int = 300):
    """
    Generate a solution using incremental token generation with reasoning mechanism.
    """
    device = torch.device("cpu")

    # Special reasoning tokens to track
    special_tokens = [
        "<REASONING_SUCCESS_END>",
        "<REASONING_FAILURE_END>",
        "<REASONING_END>",
    ]

    # Construct initial prompt with reasoning context
    full_prompt = dedent(
        f"""
        Task: {prompt}
        Your job is to print all reasoning steps for this.

        You must include your reasoning between the <REASONING_START> and <REASONING_END> tags.
        <REASONING_START>
        """
    ).strip()

    logger.info(f"Initial Prompt: {full_prompt}")

    # Tokenize initial input
    inputs = tokenizer(
        full_prompt, return_tensors="pt", padding=True, add_special_tokens=True
    ).to(device)

    # Generation parameters
    generation_config = {
        "max_new_tokens": 20,
        "num_return_sequences": 1,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50,
        "repetition_penalty": 1.1,
        "no_repeat_ngram_size": 2,
        "use_cache": True,
    }

    # Track total generated tokens and full generated text
    total_tokens_generated = 0
    full_generated_text = full_prompt

    while total_tokens_generated < max_tokens:
        # Generate next batch of tokens
        outputs = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            **generation_config,
        )

        # Decode the newly generated tokens
        new_tokens = outputs[0, inputs.input_ids.shape[1] :]
        decoded_new_tokens = tokenizer.decode(new_tokens, skip_special_tokens=False)

        # Print newly generated tokens
        print(decoded_new_tokens)

        # Append new tokens to full text
        full_generated_text += decoded_new_tokens

        # Update inputs for next iteration
        inputs = tokenizer(
            full_generated_text,
            return_tensors="pt",
            padding=True,
            add_special_tokens=True,
        ).to(device)

        # Update token count
        total_tokens_generated += generation_config["max_new_tokens"]

        # Check for termination conditions
        if (
            "<REASONING_SUCCESS_END>" in full_generated_text
            or "<REASONING_FAILURE_END>" in full_generated_text
        ):
            break

    # Extract solution if found
    if "<REASONING_SUCCESS_END>" in full_generated_text:
        solution_start = full_generated_text.find("<REASONING_SUCCESS_START>") + len(
            "<REASONING_SUCCESS_START>"
        )
        solution_end = full_generated_text.find("<REASONING_SUCCESS_END>")
        solution = full_generated_text[solution_start:solution_end].strip()
        logger.info(f"Solution Found: {solution}")
        return solution

    logger.warning("No clear solution generated.")
    return "Unable to solve the problem completely."


def main():
    # Load environment variables
    load_dotenv()

    # Authenticate with Hugging Face
    token = os.getenv("HUGGINGFACE_TOKEN")
    login(token=token)

    # Model details
    model_name = "meta-llama/Llama-3.2-11B-Vision"

    # Argument parsing
    parser = argparse.ArgumentParser(description="Reasoning-based Problem Solver")
    parser.add_argument(
        "--prompt", type=str, required=True, help="Problem prompt to solve"
    )
    args = parser.parse_args()

    # Setup tokenizer and model
    tokenizer, model = setup_tokenizer_and_model(model_name)

    # Generate and print solution
    solution = generate_reasoning_solution(model, tokenizer, args.prompt)
    print(f"Final Solution: {solution}")


if __name__ == "__main__":
    main()
