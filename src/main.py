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

    # Initialize tokenizer with explicit padding
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

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
        print("Response:", end=" ", flush=True)

        # Tokenize input with padding and attention mask
        inputs = tokenizer(
            prompt, return_tensors="pt", padding=True, add_special_tokens=True
        )

        # Generate text with streaming
        generated_ids = inputs.input_ids
        for _ in range(50):  # Generate up to 50 new tokens
            outputs = model.generate(
                generated_ids,
                max_new_tokens=1,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
                attention_mask=inputs.attention_mask,
            )

            # Get the newly generated token
            new_token = outputs[0, generated_ids.shape[1] :]

            # Decode and print the new token
            new_token_text = tokenizer.decode(new_token, skip_special_tokens=True)
            print(new_token_text, end="", flush=True)

            # Update generated_ids for next iteration
            generated_ids = outputs

            # Optional: Stop generation if EOS token is generated
            if tokenizer.eos_token_id in outputs[0]:
                break

        print("\n")  # New line after each response


if __name__ == "__main__":
    main()
