import os
import torch
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoTokenizer
from dynamic_context_compression.models.reasoning_llama import (
    ReasoningLlamaForCausalLM,
    ReasoningLlamaConfig,
)
from textwrap import dedent
import argparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def is_newly_generated_special_token(
    decoded_output: str, original_prompt: str, special_tokens: list
) -> bool:
    """
    Efficiently determine if a newly generated token is a special token
    and not part of the original prompt.
    """
    generated_portion = decoded_output[len(original_prompt):]

    for token in special_tokens:
        if token in generated_portion and generated_portion.index(token) == 0:
            return True
    return False


def setup_tokenizer_and_model(model_name: str, use_gpu: bool = False):
    """
    Set up the tokenizer and model with special reasoning tokens.

    Args:
        model_name (str): Name of the model to load
        use_gpu (bool): Whether to use GPU acceleration

    Returns:
        Tuple[AutoTokenizer, ReasoningLlamaForCausalLM]
    """
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

    # Determine device
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create custom config
    config = ReasoningLlamaConfig.from_pretrained(model_name)

    # Initialize custom model with device and dtype optimization
    model = ReasoningLlamaForCausalLM.from_pretrained(
        model_name, 
        config=config, 
        device_map="auto" if use_gpu else "cpu", 
        torch_dtype=torch.float16 if use_gpu else torch.float32
    )

    # Move model to the selected device
    model = model.to(device)

    # Resize token embeddings after adding special tokens
    model.resize_token_embeddings(len(tokenizer))

    return tokenizer, model


def generate_reasoning_solution(
    model, tokenizer, prompt: str, use_gpu: bool = False, max_recovery_attempts: int = 3
):
    """
    Generate a solution using reasoning mechanism with advanced token generation.
    """
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    
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

        Reasoning Framework:
        1. Break down the problem systematically
        2. Generate multiple solution approaches
        3. Critically evaluate each approach
        4. Select and justify the most promising solution

        Example Problem Solving:
        Problem: Calculate 17 * 23
        <REASONING_START>
        - Multiply systematically 
        - 17 * 23 = (10 + 7) * (20 + 3)
        - Distribute: 200 + 30 + 140 + 21
        - Solution: 391
        <REASONING_SUCCESS_START>391<REASONING_SUCCESS_END>
        <REASONING_END>

        Problem Solving Guidelines:
        - Be thorough and logical
        - Explain reasoning step-by-step
        - Use special tokens precisely

        Your Task:
        <REASONING_START>
        """
    ).strip()

    logger.info(f"Initial Prompt: {full_prompt}")

    # Tokenize initial input
    inputs = tokenizer(
        full_prompt, return_tensors="pt", padding=True, add_special_tokens=True
    ).to(device)

    # Generation parameters optimized for better performance
    generation_config = {
        "max_new_tokens": 200,  # Increased to allow more complex reasoning
        "num_return_sequences": 1,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50,
        "repetition_penalty": 1.1,
        "no_repeat_ngram_size": 2,  # Prevent repeating n-grams
    }

    # Failure recovery attempts with more intelligent generation
    for attempt in range(max_recovery_attempts):
        logger.info(f"Reasoning Attempt {attempt + 1}")

        # Advanced generation with more sophisticated parameters
        outputs = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            **generation_config
        )

        # Decode the entire output
        decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=False)
        logger.info(f"Generated Output: {decoded_output}")

        # Check for reasoning markers and extract solution or handle failure
        if "<REASONING_SUCCESS_END>" in decoded_output:
            solution_start = decoded_output.find("<REASONING_SUCCESS_START>") + len("<REASONING_SUCCESS_START>")
            solution_end = decoded_output.find("<REASONING_SUCCESS_END>")
            solution = decoded_output[solution_start:solution_end].strip()
            logger.info(f"Solution Found: {solution}")
            return solution

        elif "<REASONING_FAILURE_END>" in decoded_output:
            failure_start = decoded_output.find("<REASONING_FAILURE_START>") + len("<REASONING_FAILURE_START>")
            failure_end = decoded_output.find("<REASONING_FAILURE_END>")
            failure_explanation = decoded_output[failure_start:failure_end].strip()
            logger.warning(f"Reasoning Failure (Attempt {attempt + 1}): {failure_explanation}")

    logger.error("Failed to solve the problem after multiple attempts.")
    return "Unable to solve the problem."


def main():
    # Load environment variables
    load_dotenv()

    # Authenticate with Hugging Face
    token = os.getenv("HUGGINGFACE_TOKEN")
    login(token=token)

    # Model details
    model_name = "meta-llama/Llama-3.2-1B"

    # Argument parsing with GPU flag
    parser = argparse.ArgumentParser(description="Reasoning-based Problem Solver")
    parser.add_argument("--prompt", type=str, required=True, help="Problem prompt to solve")
    parser.add_argument("--gpu", action="store_true", help="Use GPU acceleration")
    args = parser.parse_args()

    # Setup tokenizer and model with optional GPU
    tokenizer, model = setup_tokenizer_and_model(model_name, use_gpu=args.gpu)

    # Generate and print solution
    solution = generate_reasoning_solution(model, tokenizer, args.prompt, use_gpu=args.gpu)
    print(f"Final Solution: {solution}")


if __name__ == "__main__":
    main()