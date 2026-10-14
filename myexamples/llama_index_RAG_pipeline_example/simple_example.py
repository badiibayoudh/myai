
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.llms import ChatMessage

# Path to your downloaded Hugging Face model
model_path = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

llm = HuggingFaceLLM(
    model_name="Qwen/Qwen2.5-Coder-1.5B-Instruct",
    tokenizer_name="Qwen/Qwen2.5-Coder-1.5B-Instruct",
    context_window=4096,
    max_new_tokens=100,
    device_map="auto",
    model_kwargs={
        "dtype": "auto"#,
        #"local_files_only": True
    },
    generate_kwargs={
        "temperature": 0.7,
        "do_sample": True
    },
    is_chat_model=True
)

response = llm.chat([
    ChatMessage(role="user", content="Hello, how are you?")
])

print(response.message.content)
