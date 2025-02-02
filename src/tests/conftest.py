import pytest
import torch
from transformers import AutoTokenizer
from dynamic_context_compression.models.reasoning_llama import (
    ReasoningLlamaForCausalLM,
    ReasoningLlamaConfig,
)
import os

# Set up multi-core processing for tests
os.environ['OMP_NUM_THREADS'] = '16'  # Match your CPU core count
os.environ['MKL_NUM_THREADS'] = '16'
torch.set_float32_matmul_precision('high')

@pytest.fixture(scope="class")
def llama_model_setup(request):
    """
    Fixture to set up the Llama model and tokenizer for testing.

    Args:
        request: pytest fixture for managing test class lifecycle

    Returns:
        Dict containing tokenizer, model, and other necessary components
    """
    # Specify the model name. (For testing, consider using a smaller model if necessary.)
    model_name = "meta-llama/Llama-3.2-1B"  # Change to a smaller model if faster testing is needed.

    # Load the tokenizer from the pre-trained model.
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Set the pad token to be the same as the end-of-sequence token.
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

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
    tokenizer.add_special_tokens(special_tokens_dict)

    # Load the custom configuration and model.
    config = ReasoningLlamaConfig.from_pretrained(model_name)
    model = ReasoningLlamaForCausalLM.from_pretrained(
        model_name, config=config, device_map="cpu", torch_dtype=torch.bfloat16
    )

    # Resize model embeddings to include the additional special tokens.
    model.resize_token_embeddings(len(tokenizer))

    # Attach to the class for access in tests
    request.cls.tokenizer = tokenizer
    request.cls.model = model

    return {"tokenizer": tokenizer, "model": model, "model_name": model_name}
