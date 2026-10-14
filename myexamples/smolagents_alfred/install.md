# Virtual environment
## Create Virtual environment
python -m venv alfred

## Activate Virtual environment
alfred\scripts\activate

## Deactivate Virtual environment
deactivate

# Verbindung zu extern system
## Set Hugging Faces Key
create environment variable HF_TOKEN

# Dependencies
requests
duckduckgo_search
markdownify
pip install requests duckduckgo_search  markdownify

smolagents
pip install smolagents -U

transformers
pip install "smolagents[transformers]"

gradio
python -m pip install "smolagents[gradio]"
http://127.0.0.1:7860

langfuse
pip install opentelemetry-sdk opentelemetry-exporter-otlp openinference-instrumentation-smolagents langfuse


# Credits
To use the Hugging Faces inference model, credits are required.