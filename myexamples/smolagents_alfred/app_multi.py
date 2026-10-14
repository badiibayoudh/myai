import yaml
import os
from smolagents import GradioUI, CodeAgent, TransformersModel

from PIL import Image

from smolagents.utils import encode_image_base64, make_image_url
from smolagents import OpenAIServerModel

# Get current directory path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

from tools.web_search import DuckDuckGoSearchTool as WebSearch
from tools.calculate_cargo_travel_time import calculateCargoTravelTimeTool as calculateCargoTravelTime
from tools.visit_webpage import VisitWebpageTool as VisitWebpage
from tools.final_answer import FinalAnswerTool as FinalAnswer

# Langfuse configuration
#---------------------
# Get keys for your project from the project settings page: https://cloud.langfuse.com
os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-432b5f77-8611-4256-ba6b-70571083dfa4" 
os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-4f0e9c76-7da1-44fa-a1d5-9ac250ff3fc3" 
os.environ["LANGFUSE_HOST"] = "https://cloud.langfuse.com" # 🇪🇺 EU region

# Instrumentation for Langfuse
#-----------------------------
from langfuse import get_client
langfuse = get_client()
# Verify connection
if langfuse.auth_check():
    print("Langfuse client is authenticated and ready!")
else:
    print("Authentication failed. Please check your credentials and host.")
from openinference.instrumentation.smolagents import SmolagentsInstrumentor
SmolagentsInstrumentor().instrument()


# Tools initialization
#---------------------
web_search = WebSearch()
visit_webpage = VisitWebpage()
final_answer = FinalAnswer()

# Model initialization
#---------------------
model_light = TransformersModel(
    model_id= "Qwen/Qwen2.5-Coder-1.5B-Instruct", #"Qwen/Qwen2.5-Coder-7B-Instruct", #"HuggingFaceTB/SmolLM3-3B", #"Qwen/Qwen2.5-0.5B-Instruct", #"Qwen/Qwen2.5-Coder-7B-Instruct",
    device_map="cpu",
    max_new_tokens=2048,
)
model = TransformersModel(
    model_id= "Qwen/Qwen2.5-Coder-3B-Instruct", #"Qwen/Qwen2.5-Coder-7B-Instruct", #"HuggingFaceTB/SmolLM3-3B", #"Qwen/Qwen2.5-0.5B-Instruct", #"Qwen/Qwen2.5-Coder-7B-Instruct",
    device_map="cpu",
    max_new_tokens=2048,
)

# System prompt
#--------------
# with open(os.path.join(CURRENT_DIR, "prompts.yaml"), 'r') as stream:
#    prompt_templates = yaml.safe_load(stream)


def check_reasoning_and_plot(final_answer, agent_memory):
    multimodal_model = OpenAIServerModel("gpt-4o", max_tokens=8096)
    filepath = "saved_map.png"
    assert os.path.exists(filepath), "Make sure to save the plot under saved_map.png!"
    image = Image.open(filepath)
    prompt = (
        f"Here is a user-given task and the agent steps: {agent_memory.get_succinct_steps()}. Now here is the plot that was made."
        "Please check that the reasoning process and plot are correct: do they correctly answer the given task?"
        "First list reasons why yes/no, then write your final decision: PASS in caps lock if it is satisfactory, FAIL if it is not."
        "Don't be harsh: if the plot mostly solves the task, it should pass."
        "To pass, a plot should be made using px.scatter_map and not any other method (scatter_map looks nicer)."
    )
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt,
                },
                {
                    "type": "image_url",
                    "image_url": {"url": make_image_url(encode_image_base64(image))},
                },
            ],
        }
    ]
    output = multimodal_model(messages).content
    print("Feedback: ", output)
    if "FAIL" in output:
        raise Exception(output)
    return True

# Agent initialization
#----------------------
web_agent = CodeAgent(
    model=model_light,
    tools=[
        web_search,
        visit_webpage,
        calculateCargoTravelTime,
    ],
    name="web_agent",
    description="Browses the web to find information",
    verbosity_level=0,
    max_steps=10,
)

agent = CodeAgent(
    model=model,
    tools=[calculateCargoTravelTime, web_search, visit_webpage],
    managed_agents=[],
    max_steps=20,
    verbosity_level=2,
    planning_interval=None,
    name=None,
    description=None,
    # prompt_templates=prompt_templates,
    additional_authorized_imports=['requests', 'pandas']
)
agent.planning_interval = 4

manager_agent = CodeAgent(
    model=model,
    tools=[calculateCargoTravelTime],
    managed_agents=[web_agent],
    additional_authorized_imports=[
        "geopandas",
        "plotly",
        "shapely",
        "json",
        "pandas",
        "numpy",
    ],
    planning_interval=5,
    verbosity_level=2,
    #final_answer_checks=[check_reasoning_and_plot],
    max_steps=15,
)

# Prompts
#----------
# task = """Find all Batman filming locations in the world, calculate the time to transfer via cargo plane to here (we're in Gotham, 40.7128° N, 74.0060° W), and return them to me as a pandas dataframe.
# Also give me some supercar factories with the same cargo plane transfer time."""

# detailed_report = agent.run(f"""
# You're an expert analyst. You make comprehensive reports after visiting many websites.
# Don't hesitate to search for many queries at once in a for loop.
#For each data point that you find, visit the source url to confirm numbers.

# {task}
# """)
#print(detailed_report)


manager_agent.visualize()

manager_agent.run("""
Find all Batman filming locations in the world, calculate the time to transfer via cargo plane to here (we're in Gotham, 40.7128° N, 74.0060° W).
Also give me some supercar factories with the same cargo plane transfer time. You need at least 6 points in total.
Represent this as spatial map of the world, with the locations represented as scatter points with a color that depends on the travel time, and save it to saved_map.png!

Here's an example of how to plot and return a map:
import plotly.express as px
df = px.data.carshare()
fig = px.scatter_map(df, lat="centroid_lat", lon="centroid_lon", text="name", color="peak_hour", size=100,
     color_continuous_scale=px.colors.sequential.Magma, size_max=15, zoom=1)
fig.show()
fig.write_image("saved_image.png")
final_answer(fig)

Never try to process strings using code: when you have a string to read, just print it and you'll see it.
""")

# if __name__ == "__main__":
#    GradioUI(agent).launch()
