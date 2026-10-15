# create temporary environment
python -m venv llama_index

# activate temporary environment
llama_index\scripts\activate

# install llama_index components
pip install llama-index-llms-huggingface-api llama-index-embeddings-huggingface

pip install -U llama-index-llms-huggingface transformers torch accelerate


# install chroma database
pip install llama-index-vector-stores-chroma


pip install llama-index-callbacks-arize-phoenix
pip install -q arize-phoenix


pip install llama-index-utils-workflow