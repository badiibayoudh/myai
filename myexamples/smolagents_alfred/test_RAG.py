from smolagents import CodeAgent, DuckDuckGoSearchTool

# Initialize the search tool
search_tool = DuckDuckGoSearchTool()

# Initialize the model
from smolagents import TransformersModel
model = TransformersModel(
    model_id= "Qwen/Qwen2.5-0.5B-Instruct", #"Qwen/Qwen2.5-Coder-7B-Instruct",
    device_map="cpu",
    max_new_tokens=2048,
)

agent = CodeAgent(
    model=model,
    tools=[search_tool],
    additional_authorized_imports=['requests', 'urllib.parse'],
)

# Example usage
response = agent.run(
    "Search for luxury superhero-themed party ideas, including decorations, entertainment, and catering."
)
print(response)