import copy
from llama_cpp import Llama

# load the model
print("Loading model...")
# llm = Llama("./models/gpt4all_ggml/gpt4all-lora-quantized-ggml.bin")
llm = Llama("./models/ggml-vicuna-13b-4bit-rev1.bin", n_ctx=1024)
print("Model loaded.")
stream = llm(
    "Question: Tell me about Liechtenstein. Answer:",
    max_tokens=200,
    stop=["\n", "Question:", "Q:"],
    stream=True,
)
# print the output
for output in stream:
    completionFragment = copy.deepcopy(output)
    print(completionFragment["choices"][0]["text"])