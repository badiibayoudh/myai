import os
from smolagents import GradioUI, CodeAgent, TransformersModel

# Test login with Hugging Face
#from huggingface_hub import login
# login()

# Use inference client model with Hugging Face token
HF_TOKEN = os.environ.get("HF_TOKEN")
model = TransformersModel(
    model_id="Qwen/Qwen2.5-Coder-7B-Instruct",
    device="cpu",
    max_new_tokens=2048,
)

# Test agent
agent = CodeAgent(model=model, tools=[DuckDuckGoSearchTool()])
agent.run("Search for the best music recommendations for a party at the Wayne's mansion.")

