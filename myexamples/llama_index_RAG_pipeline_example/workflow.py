from llama_index.core.workflow import StartEvent, StopEvent, Workflow, step


# Basic workflow example
#########################
class MyWorkflow(Workflow):
    @step
    async def my_step(self, ev: StartEvent) -> StopEvent:
        # do something here
        return StopEvent(result="Hello, world!")
w1 = MyWorkflow(timeout=10, verbose=False)


# Multi-step workflow example
##############################
from llama_index.core.workflow import Event

class ProcessingEvent(Event):
    intermediate_result: str

class MultiStepWorkflow(Workflow):
    @step
    async def step_one(self, ev: StartEvent) -> ProcessingEvent:
        # Process initial data
        return ProcessingEvent(intermediate_result="Step 1 complete")

    @step
    async def step_two(self, ev: ProcessingEvent) -> StopEvent:
        # Use the intermediate result
        final_result = f"Finished processing: {ev.intermediate_result}"
        return StopEvent(result=final_result)

w2 = MultiStepWorkflow(timeout=10, verbose=False)


# Multi-step workflow with loop example
##############################
import random
class LoopEvent(Event):
    loop_output: str

class MultiStepLoopWorkflow(Workflow):
    @step
    async def step_one(self, ev: StartEvent | LoopEvent) -> ProcessingEvent | LoopEvent:
        if random.randint(0, 1) == 0:
            print("Bad thing happened")
            return LoopEvent(loop_output="Back to step one.")
        else:
            print("Good thing happened")
            return ProcessingEvent(intermediate_result="First step complete.")

    @step
    async def step_two(self, ev: ProcessingEvent) -> StopEvent:
        # Use the intermediate result
        final_result = f"Finished processing: {ev.intermediate_result}"
        return StopEvent(result=final_result)


w3 = MultiStepLoopWorkflow(verbose=False)

# draw workflow
#######################
from llama_index.utils.workflow import draw_all_possible_flows
# draw_all_possible_flows(w3, "flow.html")


# Define agents workflow
#########################
from llama_index.core.agent.workflow import AgentWorkflow, ReActAgent
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.workflow import Context

# Define some tools
async def add(ctx: Context, a: int, b: int) -> int:
    """Add two numbers."""
    # update our count
    cur_state = await ctx.store.get("state")
    cur_state["num_fn_calls"] += 1
    await ctx.store.set("state", cur_state)

    return a + b

async def multiply(ctx: Context, a: int, b: int) -> int:
    """Multiply two numbers."""
    # update our count
    cur_state = await ctx.store.get("state")
    cur_state["num_fn_calls"] += 1
    await ctx.store.set("state", cur_state)

    return a * b

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

# we can pass functions directly without FunctionTool -- the fn/docstring are parsed for the name/description
multiply_agent = ReActAgent(
    name="multiply_agent",
    description="Is able to multiply two integers",
    system_prompt="A helpful assistant that can use a tool to multiply numbers.",
    tools=[multiply],
    llm=llm,
)

addition_agent = ReActAgent(
    name="add_agent",
    description="Is able to add two integers",
    system_prompt="A helpful assistant that can use a tool to add numbers.",
    tools=[add],
    llm=llm,
)

# Create the workflow
workflow = AgentWorkflow(
    agents=[multiply_agent, addition_agent],
    root_agent="multiply_agent",
    initial_state={"num_fn_calls": 0},
    state_prompt="Current state: {state}. User message: {msg}",
)

ctx = Context(workflow)


# Run the workflow asynchronously
import asyncio
async def main():
    #result = await w3.run()
    result = await workflow.run(user_msg="Can you add 5 and 3?", ctx=ctx)
    print(result)

    # pull out and inspect the state
    state = await ctx.store.get("state")
    print(state["num_fn_calls"])

if __name__ == "__main__":
    asyncio.run(main())