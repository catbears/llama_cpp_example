import json

from llama_cpp import Llama

# load the model
print("Loading model...")
# llm = Llama("./models/gpt4all_ggml/gpt4all-lora-quantized-ggml.bin")
llm = Llama("./models/ggml-vicuna-13b-1.1-q4_2.bin")
print("Model loaded.")
output = llm(
    "Question: Tell me about Liechtenstein. Answer:",
    max_tokens=100,
    stop=["\n", "Question:", "Q:"],
    echo=True
)
# print the output
print(json.dumps(output, indent=2))