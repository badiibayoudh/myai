
import chromadb

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.vector_stores.chroma import ChromaVectorStore

# 1. Load local Hugging Face LLM
llm = HuggingFaceLLM(
    model_name="Qwen/Qwen2.5-Coder-1.5B-Instruct",
    tokenizer_name="Qwen/Qwen2.5-Coder-1.5B-Instruct",
    context_window=4096,
    max_new_tokens=256,
    device_map="auto",
    model_kwargs={"dtype": "auto"},
    generate_kwargs={
        "temperature": 0.7,
        "do_sample": True
    },
    is_chat_model=True
)

# 2. Load documents
documents = SimpleDirectoryReader(
    input_dir=r"C:\Data\Git\myai\myexamples\llama_index_RAG_pipeline_example\data"
).load_data()

print(f"Loaded {len(documents)} documents")

# 3. Initialize ChromaDB
db = chromadb.PersistentClient(path="./alfred_chroma_db")
collection = db.get_or_create_collection("alfred")

vector_store = ChromaVectorStore(
    chroma_collection=collection
)

# 4. Initialize local embedding model
embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

# 5. Create ingestion pipeline
pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(
            chunk_size=512,
            chunk_overlap=50
        ),
        embed_model
    ],
    vector_store=vector_store,
)

# 6. Generate and store embeddings
# Skip ingestion if the collection already contains data
if collection.count() == 0:
    nodes = pipeline.run(documents=documents)
    print(f"Created {len(nodes)} nodes")

print(f"ChromaDB contains {collection.count()} nodes")

# 7. Connect LlamaIndex to existing ChromaDB
index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store,
    embed_model=embed_model
)

# 8. Create RAG query engine using local Qwen
query_engine = index.as_query_engine(
    llm=llm,
    similarity_top_k=3,
    response_mode="tree_summarize"
)

# 9. Ask a question
response = query_engine.query(
    "Summarize the documents"
)
print(response)

# 10. Evaluate the response quality
from llama_index.core.evaluation import FaithfulnessEvaluator

evaluator = FaithfulnessEvaluator(llm=llm)
eval_result = evaluator.evaluate_response(response=response)
print(f"\nEvaluation Result: {eval_result}\n")
eval_result.passing

#11. trace
import llama_index
import os

os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"api_key={PHOENIX_API_KEY}"
llama_index.core.set_global_handler(
    "arize_phoenix",
    endpoint="https://llamatrace.com/v1/traces"
)

# Use Agents now
# ------------------------------------------

# define tools
def multiply(a: int, b: int) -> int:
    """Multiplies two integers and returns the resulting integer"""
    return a * b

from llama_index.core.tools import QueryEngineTool
query_engine_tool = QueryEngineTool.from_defaults(
    query_engine=query_engine,
    name="name",
    description="a specific description",
    return_direct=False,
)

def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

def subtract(a: int, b: int) -> int:
    """Subtract two numbers."""
    return a - b

# initialize agents
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.core.tools import FunctionTool
agent_calc = AgentWorkflow.from_tools_or_functions(
    [FunctionTool.from_defaults(multiply)],
    llm=llm
)

query_engine_agent = AgentWorkflow.from_tools_or_functions(
    [query_engine_tool],
    llm=llm,
    system_prompt="You are a helpful assistant that has access to a database containing persona descriptions. "
)


from llama_index.core.agent.workflow import (
    AgentWorkflow,
    FunctionAgent,
    ReActAgent,
)

calculator_agent = ReActAgent(
    name="calculator",
    description="Performs basic arithmetic operations",
    system_prompt="You are a calculator assistant. Use your tools for any math operation.",
    tools=[add, subtract],
    llm=llm,
)

query_agent = ReActAgent(
    name="info_lookup",
    description="Looks up information about XYZ",
    system_prompt="Use your tool to query a RAG system to answer information about XYZ",
    tools=[query_engine_tool],
    llm=llm
)

# Create and run the workflow
agent = AgentWorkflow(
    agents=[calculator_agent, query_agent], root_agent="calculator"
)

# Run the agent asynchronously
import asyncio

async def main():
    response = await agent.run(
        user_msg="Can you add 5 and 3?"
    )
    print(response)

if __name__ == "__main__":
    asyncio.run(main())
