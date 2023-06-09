import json

from llama_cpp import Llama

# load the model
print("Loading model...")
# llm = Llama("./models/gpt4all_ggml/gpt4all-lora-quantized-ggml.bin")
llm = Llama("./models/ggml-vicuna-13b-4bit-rev1.bin", n_ctx=4048,
            last_n_tokens_size=4048)
print("Model loaded.")
output = llm(
    "Question: Tell me about Liechtenstein. Answer:",
    max_tokens=200,
    stop=["\n", "Question:", "Q:"],
    echo=True
)
# print the output
print(json.dumps(output, indent=2))