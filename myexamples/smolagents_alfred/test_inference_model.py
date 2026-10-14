import os
from smolagents import CodeAgent, DuckDuckGoSearchTool, InferenceClientModel

# Test login with Hugging Face
#from huggingface_hub import login
# login()

# Use inference client model with Hugging Face token
HF_TOKEN = os.environ.get("HF_TOKEN")
model = InferenceClientModel(
model_id='Qwen/Qwen2.5-Coder-32B-Instruct',
provider=None,
token=HF_TOKEN,
)

# Test agent
agent = CodeAgent(model=model, tools=[DuckDuckGoSearchTool()])
agent.run("Search for the best music recommendations for a party at the Wayne's mansion.")

