import torch
from transformers import AutoTokenizer
from models.custom_llama import CustomLlamaForCausalLM, CustomLlamaConfig
from dotenv import load_dotenv
from huggingface_hub import login
import os


def main():
    # Load environment variables
    load_dotenv()

    # Authenticate with Hugging Face
    token = os.getenv("HUGGINGFACE_TOKEN")
    login(token=token)

    # Model details
    model_name = "meta-llama/Llama-3.2-3B"

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Create custom config
    config = CustomLlamaConfig.from_pretrained(model_name)

    # Initialize custom model for CPU
    model = CustomLlamaForCausalLM.from_pretrained(
        model_name, config=config, device_map="cpu", torch_dtype=torch.float32
    )

    # List of prompts to generate
    prompts = [
        "Explain quantum physics",
        "Tell me about artificial intelligence",
        "Describe the future of technology",
        "What is the meaning of life?",
    ]

    # Generate text for each prompt
    for i, prompt in enumerate(prompts, 1):
        print(f"Prompt {i}: {prompt}")

        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt")

        # Generate text
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

        # Decode and print response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Response {i}: {response}\n")


if __name__ == "__main__":
    main()
