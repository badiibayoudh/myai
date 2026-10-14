import yaml
import os
from smolagents import GradioUI, CodeAgent, TransformersModel

# Get current directory path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

from tools.web_search import DuckDuckGoSearchTool as WebSearch
from tools.visit_webpage import VisitWebpageTool as VisitWebpage
from tools.suggest_menu import SimpleTool as SuggestMenu
from tools.catering_service_tool import SimpleTool as CateringServiceTool
from tools.superhero_party_theme_generator import SuperheroPartyThemeTool as SuperheroPartyThemeGenerator
from tools.final_answer import FinalAnswerTool as FinalAnswer



model = TransformersModel(
    model_id= "Qwen/Qwen2.5-Coder-1.5B-Instruct", #"HuggingFaceTB/SmolLM3-3B", #"Qwen/Qwen2.5-0.5B-Instruct", #"Qwen/Qwen2.5-Coder-7B-Instruct",
    device_map="cpu",
    max_new_tokens=2048,
)


# Get keys for your project from the project settings page: https://cloud.langfuse.com
os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-432b5f77-8611-4256-ba6b-70571083dfa4" 
os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-4f0e9c76-7da1-44fa-a1d5-9ac250ff3fc3" 
os.environ["LANGFUSE_HOST"] = "https://cloud.langfuse.com" # 🇪🇺 EU region

from langfuse import get_client
langfuse = get_client()
# Verify connection
if langfuse.auth_check():
    print("Langfuse client is authenticated and ready!")
else:
    print("Authentication failed. Please check your credentials and host.")

from openinference.instrumentation.smolagents import SmolagentsInstrumentor
SmolagentsInstrumentor().instrument()



web_search = WebSearch()
visit_webpage = VisitWebpage()
suggest_menu = SuggestMenu()
catering_service_tool = CateringServiceTool()
superhero_party_theme_generator = SuperheroPartyThemeGenerator()
final_answer = FinalAnswer()


with open(os.path.join(CURRENT_DIR, "prompts.yaml"), 'r') as stream:
    prompt_templates = yaml.safe_load(stream)

agent = CodeAgent(
    model=model,
    tools=[web_search, visit_webpage, suggest_menu, catering_service_tool, superhero_party_theme_generator],
    managed_agents=[],
    max_steps=10,
    verbosity_level=2,
    planning_interval=None,
    name=None,
    description=None,
    prompt_templates=prompt_templates,
    additional_authorized_imports=['requests']
)
if __name__ == "__main__":
    GradioUI(agent).launch()
