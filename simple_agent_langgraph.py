import os
import json

from typing import Annotated, Literal, TypedDict
from dotenv import load_dotenv

from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.messages import ToolMessage, BaseMessage
from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv()
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

class BasicToolNode:
    """A node that runs the tools requested in the last AIMessage"""
    def __init__(self, tools : list):
        self.tools_by_name = {tool.name : tool for tool in tools}
        
    def __call__(self, inputs : dict):
        
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No messages in inputs")
        
        outputs = []
        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(
                tool_call['args']
            )
            outputs.append(
                ToolMessage(
                    content = json.dumps(tool_result),
                    name = tool_call['name'],
                    tool_call_id = tool_call['id']
                )
            )
        return {"messages" : outputs}


class State(TypedDict):
    messages : Annotated[list, add_messages]
    
def bot(state: State):
    # print(f"-----------------> {state['messages']}")
    return {"messages" : [model_with_tools.invoke(state["messages"])]}

def route_tools(state: State)-> Literal["tools", "__end__"]:
    """
    Use in the conditional_edge to route to the ToolNode if the last message contains a tool call
    Otherwise, route to the end
    """
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError("No messages found in input state")
    
    if hasattr(ai_message, "tool_calls") and ai_message.tool_calls:
        return "tools"
    
    return "__end__"
    

# Add tool node
tool = TavilySearchResults(max_results = 2)

# Method 1
tools = [tool]
# tool_node = BasicToolNode(tools)

# Method 2
tool_node = ToolNode(tools = [tool])


# Define model
model = ChatOllama(model = "MFDoom/deepseek-r1-tool-calling:8b")
# model = ChatOpenAI(model = "gpt-4o-mini")
model_with_tools = model.bind_tools(tools)


# Add memory note
memory = MemorySaver()

# Construct graph
graph_builder = StateGraph(State)
graph_builder.add_node("bot", bot)
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
   "bot",
   # Method 1
   route_tools,
   {"tools" : "tools", "__end__" : END} 
    # Method 2
    # tools_condition
)

graph_builder.add_edge("tools", "bot")

graph_builder.set_entry_point("bot")
graph_builder.set_finish_point("bot")

 
# Compile graph
graph = graph_builder.compile(
    checkpointer=memory,
    # interrupt_before = ["tools"]
)


# Run the AI Agent
config = {
    "configurable" : {"thread_id" : 1}
}

while True:
    user_input = input("User : ")
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break
    for event in graph.stream({"messages" : ("user", user_input)}, config=config):
        for value in event.values():
            if isinstance(value['messages'][-1], BaseMessage):
                print("Assistant : ", value["messages"][-1].content, "\n----------------------------------------------------\n")
            


# user_input = "What is the capital of Malaysia?"
# events = graph.stream({"messages" : [("user", user_input)]}, config=config, stream_mode = "values")
# for event in events:
#     event['messages'][-1].pretty_print()            

# snapshot = graph.get_state(config)
# next_step = (
#     snapshot.next
# )

# print(
#     "\n===>>>", next_step
# )

# existing_message = snapshot.values['messages'][-1]
# all_tools = existing_message.tool_calls

# print("Tools to be called :: ", all_tools)

# user_input = "What is the capital of Malaysia?"
# events = graph.stream({"messages" : [("user", user_input)]}, config=config, stream_mode = "values")
# for event in events:
#     event['messages'][-1].pretty_print()